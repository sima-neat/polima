"""Pi0.5 export: checkpoint + dataset -> ONNX graphs, calibration, fixtures, client files.

Runs in the Pi0.5 export interpreter (torch, transformers >= 5, LeRobot with pi05),
as the `export` stage of `polima compile --policy pi05`. Writes under OUT:

  onnx/<graph>.onnx              the eleven graphs of the spec
  calibration/<graph>.npz        boundary tensors captured from real observations
  constants/cond_XX.f32          adaRMS conditioning of Euler step XX
  fixtures/inputs/*.f32          one request (the wire tensors)
  fixtures/expected/normalized_actions.f32   LeRobot PI05Pytorch on that request
  client/                        client.json, tokenizer.json, embedding.f16, normalization_stats.npz
  profile.json                   the specialized spec's profile (camera keys, dims, task)
  verification.json              ONNX chain (onnxruntime) vs LeRobot, and our modules vs LeRobot
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from polima.policies.pi05 import blocks, client, modules, weights
from polima.policies.pi05 import runtime as rt
from polima.policies.pi05.profile import Geometry, build_spec, geometry_from_config

TOKENIZER = "google/paligemma-3b-pt-224"
#: Observations the build check runs (fixtures/set/sNN); it judges their mean.
FIXTURES = 8
CAMERA_ALIASES = {
    "base": ("cam_high", "top", "overhead", "front", "base", "exterior_image_1_left", "image"),
    "left_wrist": ("cam_left_wrist", "left_wrist", "wrist", "wrist_image"),
    "right_wrist": ("cam_right_wrist", "right_wrist", "exterior_image_2_left"),
    "wrist": ("wrist", "cam_wrist", "gripper", "wrist_cam", "cam_left_wrist"),
    "top": ("top", "overhead", "cam_high", "front", "top_cam"),
}


def log(text: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {text}", flush=True)


# ----------------------------------------------------------------- graph wrappers


class VisionGraph(nn.Module):
    def __init__(self, w: dict):
        super().__init__()
        self.vision = modules.Vision(w)

    def forward(self, patches):
        return self.vision(patches.reshape(1, rt.IMAGE_TOKENS, rt.PATCH_VALUES)).reshape(
            1, 1, rt.IMAGE_TOKENS, rt.WIDTH)


def build_modules(w: dict, tokens: int) -> dict[str, nn.Module]:
    graphs = {"vision": VisionGraph(w)}
    for index, name in enumerate(rt.block_names()):
        graphs[name] = blocks.PrefixBlock(w, index * rt.BLOCK_LAYERS, rt.BLOCK_LAYERS, tokens)
    graphs["expert"] = blocks.ExpertStep(w, rt.EXPERT_LAYERS, tokens)
    return {name: module.eval() for name, module in graphs.items()}


def conditioning(w: dict) -> list[torch.Tensor]:
    """silu(time_mlp_out(silu(time_mlp_in(emb(t))))) for t = 1.0, 0.9, ..., 0.1."""
    t_in = modules.Linear(w["time_mlp_in.weight"], w["time_mlp_in.bias"])
    t_out = modules.Linear(w["time_mlp_out.weight"], w["time_mlp_out.bias"])
    silu = torch.nn.functional.silu
    out = []
    for step in range(rt.DENOISE_STEPS):
        emb = modules.time_embedding(1.0 - step / rt.DENOISE_STEPS)[None]
        out.append(silu(t_out(silu(t_in(emb)))).reshape(rt.COND))
    return out


# ----------------------------------------------------------------- observations


def camera_map(geometry: Geometry, dataset_keys: list[str], given: dict[str, str] | None) -> dict[str, str]:
    """checkpoint camera key -> dataset camera key."""
    short = {key.rsplit(".", 1)[-1]: key for key in dataset_keys}
    chosen: dict[str, str] = {}
    for key in geometry.cameras:
        name = key.rsplit(".", 1)[-1]
        if given and (name in given or key in given):
            target = given.get(name) or given[key]
            chosen[key] = short.get(target, target)
            continue
        if name in short:
            chosen[key] = short[name]
            continue
        stem = name.replace("_0_rgb", "")
        for alias in CAMERA_ALIASES.get(stem, ()):
            if alias in short and short[alias] not in chosen.values():
                chosen[key] = short[alias]
                break
        else:
            raise ValueError(f"no dataset camera for {key}; dataset has {sorted(short)}. "
                             f"Pass --camera-map {name}=<dataset camera>")
    return chosen


def observations(dataset: Path, geometry: Geometry, count: int, mapping: dict[str, str] | None,
                 task: str | None) -> list[dict]:
    """`count` frames spread over the dataset: images (HWC float [0,1]), raw state, task."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # PyAV, not torchcodec: torchcodec wheels are tied to one torch minor version.
    ds = LeRobotDataset(repo_id=f"local/{dataset.name}", root=dataset, video_backend="pyav")
    keys = list(ds.meta.camera_keys)
    cams = camera_map(geometry, keys, mapping)
    log(f"cameras: {cams}")
    picks = np.linspace(0, len(ds) - 1, count).round().astype(int)
    out = []
    for index in picks:
        item = ds[int(index)]
        out.append({
            "images": [item[cams[key]].permute(1, 2, 0).numpy().astype(np.float32) for key in geometry.cameras],
            "state": item["observation.state"].numpy().astype(np.float32),
            "task": task or item.get("task", ""),
        })
    return out


def pad_state(state: np.ndarray, dim: int) -> np.ndarray:
    out = np.zeros(dim, np.float32)
    out[: min(dim, state.size)] = state[:dim]
    return out


def checkpoint_stats(checkpoint: Path) -> client.Stats:
    """QUANTILES statistics saved with a fine-tuned checkpoint's pre-processor, if any."""
    from safetensors.numpy import load_file

    stats = {}
    for path in checkpoint.glob("policy_preprocessor_step_*normalizer*.safetensors"):
        data = load_file(str(path))
        for prefix, name in (("observation.state", "state"), ("action", "action")):
            for q in ("q01", "q99"):
                if f"{prefix}.{q}" in data:
                    stats[f"{name}_{q}"] = data[f"{prefix}.{q}"].astype(np.float32)
    for path in checkpoint.glob("policy_postprocessor_step_*unnormalizer*.safetensors"):
        data = load_file(str(path))
        for q in ("q01", "q99"):
            if f"action.{q}" in data:
                stats[f"action_{q}"] = data[f"action.{q}"].astype(np.float32)
    return client.Stats(**stats)


# ----------------------------------------------------------------- the reference


class Reference:
    """Our static modules run as the plan runs them, capturing every graph input."""

    def __init__(self, graphs: dict[str, nn.Module], cond: list[torch.Tensor], tokens: int):
        self.graphs, self.cond, self.tokens = graphs, cond, tokens

    @torch.no_grad()
    def run(self, request: dict[str, np.ndarray], cameras: int, capture: dict | None = None) -> np.ndarray:
        t = {k: torch.from_numpy(v) for k, v in request.items()}
        p = self.tokens

        def keep(name, **inputs):
            if capture is not None:
                capture.setdefault(name, []).append({k: v.numpy().copy() for k, v in inputs.items()})

        vision = []
        for k in range(cameras):
            patches = t[f"image{k}"].reshape(1, 1, rt.IMAGE_TOKENS, rt.PATCH_VALUES)
            keep("vision", patches=patches)
            vision.append(self.graphs["vision"](patches).reshape(rt.IMAGE_TOKENS, rt.WIDTH))
        hidden = torch.cat([*vision, t["language"]]).reshape(1, 1, p, rt.WIDTH)
        bias = t["prefix_bias"].reshape(1, 1, 1, p)
        kv = []
        for name in rt.block_names():
            keep(name, hidden=hidden, bias=bias)
            out = self.graphs[name](hidden, bias).reshape(-1)
            hidden = out[: p * rt.WIDTH].reshape(1, 1, p, rt.WIDTH)
            kv.append(out[p * rt.WIDTH:])
        cache = torch.cat(kv).reshape(1, 1, rt.cache_rows(p), rt.ROW)
        rope = t["suffix_rope"].reshape(1, 1, 2 * rt.CHUNK, rt.ROW)
        sbias = torch.cat([t["prefix_bias"], torch.zeros(rt.CHUNK)]).reshape(1, 1, 1, p + rt.CHUNK)
        x = t["noise"].reshape(1, 1, rt.CHUNK, rt.ACTION_LANE).clone()
        for step in range(rt.DENOISE_STEPS):
            cond = self.cond[step].reshape(1, 1, 1, rt.COND)
            keep("expert", cache=cache, x=x, cond=cond, rope=rope, bias=sbias)
            x = x - rt.DT * self.graphs["expert"](cache, x, cond, rope, sbias)
        return x.reshape(rt.CHUNK, rt.ACTION_LANE).numpy()


def lerobot_reference(checkpoint: Path, request_images: list[np.ndarray], ids: np.ndarray, valid: int,
                      noise: np.ndarray) -> np.ndarray:
    """LeRobot PI05Pytorch.sample_actions on the same (already preprocessed) inputs."""
    return lerobot_references(checkpoint, [(request_images, ids, valid, noise)])[0]


def lerobot_references(checkpoint: Path, cases: list[tuple]) -> list[np.ndarray]:
    """lerobot_reference for several (images, ids, valid, noise) cases, loading the policy once."""
    import dataclasses
    import tempfile

    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    # A checkpoint saved by a newer LeRobot may carry config fields this one lacks
    # (e.g. disabled memory options); geometry_from_config refuses enabled ones.
    config = json.loads((checkpoint / "config.json").read_text())
    known = {f.name for f in dataclasses.fields(PI05Config)} | {"type"}
    unknown = sorted(set(config) - known)
    temp = tempfile.TemporaryDirectory()
    if unknown:
        log(f"LeRobot reference: dropping config fields this LeRobot lacks: {unknown}")
        source = Path(temp.name)
        for item in checkpoint.iterdir():
            if item.name != "config.json":
                (source / item.name).symlink_to(item.resolve())
        (source / "config.json").write_text(json.dumps({k: v for k, v in config.items() if k in known}))
    else:
        source = checkpoint
    policy = PI05Policy.from_pretrained(str(source)).eval()
    policy.model.float()
    results = []
    for request_images, ids, valid, noise in cases:
        images = [torch.from_numpy(img)[None] for img in request_images]  # [1,3,224,224] in [-1,1]
        masks = [torch.ones(1, dtype=torch.bool) for _ in images]
        tokens = torch.from_numpy(ids)[None]
        attention = torch.zeros(1, rt.LANGUAGE_TOKENS, dtype=torch.bool)
        attention[0, :valid] = True
        with torch.no_grad():
            actions = policy.model.sample_actions(images, masks, tokens, attention,
                                                  noise=torch.from_numpy(noise)[None].clone())
        results.append(actions[0].float().numpy())
    del policy
    temp.cleanup()
    return results


# ----------------------------------------------------------------- export


def export_onnx(module: nn.Module, inputs: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(module, tuple(inputs.values()), str(path), input_names=list(inputs),
                          output_names=["output"], opset_version=17, dynamo=False, do_constant_folding=True)


def verify_onnx(onnx_dir: Path, request: dict[str, np.ndarray], cameras: int, tokens: int,
                cond: list[torch.Tensor]) -> np.ndarray:
    """Run the plan's sequence with onnxruntime, one session at a time (they are large)."""
    import onnxruntime as ort

    def session(name):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        return ort.InferenceSession(str(onnx_dir / f"{name}.onnx"), options, providers=["CPUExecutionProvider"])

    p = tokens
    vision = session("vision")
    feats = [vision.run(None, {"patches": request[f"image{k}"].reshape(1, 1, 256, 588)})[0].reshape(256, 2048)
             for k in range(cameras)]
    del vision
    hidden = np.concatenate([*feats, request["language"]]).reshape(1, 1, p, 2048).astype(np.float32)
    bias = request["prefix_bias"].reshape(1, 1, 1, p)
    kv = []
    for name in rt.block_names():
        out = session(name).run(None, {"hidden": hidden, "bias": bias})[0].reshape(-1)
        hidden = out[: p * 2048].reshape(1, 1, p, 2048)
        kv.append(out[p * 2048:])
    cache = np.concatenate(kv).reshape(1, 1, rt.cache_rows(p), 256)
    expert = session("expert")
    rope = request["suffix_rope"].reshape(1, 1, 100, 256)
    sbias = np.concatenate([request["prefix_bias"], np.zeros(50, np.float32)]).reshape(1, 1, 1, p + 50)
    x = request["noise"].reshape(1, 1, 50, 32).copy()
    for step in range(rt.DENOISE_STEPS):
        v = expert.run(None, {"cache": cache, "x": x, "cond": cond[step].numpy().reshape(1, 1, 1, 1024),
                              "rope": rope, "bias": sbias})[0]
        x = x - rt.DT * v
    return x.reshape(50, 32)


def compare(ours: np.ndarray, ref: np.ndarray) -> dict:
    a, b = ours.astype(np.float64).ravel(), ref.astype(np.float64).ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return {"cosine": cos, "max_abs": float(np.abs(a - b).max()), "mean_abs": float(np.abs(a - b).mean())}


def calibration_rows(captured: list[dict], limit: int) -> dict[str, np.ndarray]:
    picks = np.linspace(0, len(captured) - 1, min(limit, len(captured))).round().astype(int)
    rows = [captured[i] for i in sorted(set(picks.tolist()))]
    # (N, *input_shape): one sample per row, each with the graph input's full shape.
    return {name: np.stack([r[name] for r in rows]).astype(np.float32) for name in rows[0]}


def export_build(checkpoint: Path, dataset: Path, out: Path, *, samples: int = 16, task: str | None = None,
                 mapping: dict[str, str] | None = None, calibrate: bool = True, client_files: bool = True,
                 seed: int = 0, backend: str = "llima", expert_form: str = "nchw", denoise: str = "device") -> dict:
    from transformers import AutoTokenizer

    started = time.time()
    out.mkdir(parents=True, exist_ok=True)
    geometry = geometry_from_config(checkpoint, backend=backend)
    # The reference (LeRobot, our modules, the fixture, calibration) runs at Pi0.5's own
    # prefix (256 per camera + 200); the LLiMa backend pads that to its 1024-token group.
    cameras, p = len(geometry.cameras), rt.prefix_tokens(len(geometry.cameras))
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    stats = checkpoint_stats(checkpoint)
    obs = observations(dataset, geometry, max(samples, 1), mapping, task)
    task_text = task or obs[0]["task"] or "do the task"
    geometry = Geometry(**{**geometry.__dict__, "task": task_text})
    spec = build_spec(geometry)
    log(f"Pi0.5 {cameras} cameras, state {geometry.state_dim}, action {geometry.action_dim}, P={p}, "
        f"task {task_text!r}, {len(obs)} observations")

    w = weights.load(checkpoint, skip=(weights.EMBED, "paligemma_with_expert.gemma_expert.lm_head.weight"))
    table = weights.embedding_table(checkpoint)
    graphs = build_modules(w, p)
    cond = conditioning(w)
    reference = Reference(graphs, cond, p)

    def make_request(o: dict, noise_seed: int):
        state = stats.normalize_state(pad_state(o["state"], geometry.state_dim))
        text = client.prompt(o["task"] or task_text, state)
        enc = tokenizer(text, max_length=rt.LANGUAGE_TOKENS, padding="max_length", truncation=True,
                        padding_side="right", return_tensors="np")
        ids, valid = enc["input_ids"][0].astype(np.int64), int(enc["attention_mask"][0].sum())
        language = np.zeros((rt.LANGUAGE_TOKENS, rt.WIDTH), np.float32)
        language[:valid] = table[torch.from_numpy(ids[:valid])].numpy()
        bias = np.zeros(p, np.float32)
        bias[256 * cameras + valid:] = modules.MASK
        chw = [client.resize_with_pad(img) for img in o["images"]]
        request = {f"image{k}": client.patchify(c) for k, c in enumerate(chw)}
        request.update(language=language, prefix_bias=bias, suffix_rope=client.rope_rows(256 * cameras + valid),
                       noise=np.random.default_rng(noise_seed).standard_normal((50, 32)).astype(np.float32))
        return request, chw, ids, valid

    # ---- constants
    (out / "constants").mkdir(parents=True, exist_ok=True)
    for step, vector in enumerate(cond):
        vector.numpy().astype("<f4").tofile(out / "constants" / f"cond_{step:02d}.f32")

    # ---- fixtures: LeRobot on FIXTURES observations spread over the dataset, each with its
    # own noise (the check judges their mean); the first is also the bundle's main fixture
    picks = [obs[round(i * (len(obs) - 1) / max(1, FIXTURES - 1))] for i in range(min(FIXTURES, len(obs)))]
    made = [make_request(o, seed if i == 0 else seed + 100 + i) for i, o in enumerate(picks)]
    request, _, ids, valid = made[0]
    ours = reference.run(request, cameras)
    log(f"reference (our modules) done; running LeRobot PI05Pytorch on {len(made)} fixtures")
    references = lerobot_references(checkpoint, [(c, i, v, r["noise"]) for r, c, i, v in made])
    lerobot = references[0]
    report = {"modules_vs_lerobot": compare(ours, lerobot)}
    log(f"our modules vs LeRobot: {report['modules_vs_lerobot']}")
    fixtures = out / "fixtures"

    def write_fixture(where: Path, req: dict, actions: np.ndarray) -> None:
        (where / "inputs").mkdir(parents=True, exist_ok=True)
        (where / "expected").mkdir(parents=True, exist_ok=True)
        for name, value in req.items():
            value.astype("<f4").tofile(where / "inputs" / f"{name}.f32")
        kept = actions[:, : geometry.action_dim] if geometry.action_dim != rt.ACTION_LANE else actions
        np.ascontiguousarray(kept, dtype="<f4").tofile(where / "expected" / "normalized_actions.f32")

    write_fixture(fixtures, request, lerobot)
    for index, ((req, *_), actions) in enumerate(zip(made, references, strict=True)):
        write_fixture(fixtures / "set" / f"s{index:02d}", req, actions)
    np.savez(fixtures / "pi05_fixture.npz", prompt=np.array(client.prompt(task_text, stats.normalize_state(
        pad_state(obs[0]["state"], geometry.state_dim)))), ids=ids, valid=valid, expected=lerobot)

    if backend == "llima":
        report.update(_export_llima(checkpoint, out, w, graphs, reference, obs, make_request, cameras, cond,
                                    calibrate, seed, expert_form, denoise))
        _finish_export(out, geometry, spec, stats, tokenizer, table, fixtures, valid, ids, task_text, client_files)
        report["seconds"] = round(time.time() - started)
        (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    # ---- ONNX
    onnx_dir = out / "onnx"
    examples = {"vision": {"patches": torch.from_numpy(request["image0"]).reshape(1, 1, 256, 588)}}
    for name in rt.block_names():
        examples[name] = {"hidden": torch.zeros(1, 1, p, rt.WIDTH), "bias": torch.zeros(1, 1, 1, p)}
    examples["expert"] = {"cache": torch.zeros(1, 1, rt.cache_rows(p), rt.ROW),
                          "x": torch.zeros(1, 1, 50, 32), "cond": cond[0].reshape(1, 1, 1, rt.COND),
                          "rope": torch.zeros(1, 1, 100, rt.ROW), "bias": torch.zeros(1, 1, 1, p + 50)}
    for name, module in graphs.items():
        target = onnx_dir / f"{name}.onnx"
        if not target.is_file():
            export_onnx(module, examples[name], target)
        log(f"onnx {name}: {target.stat().st_size >> 20} MiB")
    onnx_actions = verify_onnx(onnx_dir, request, cameras, p, cond)
    report["onnx_vs_lerobot"] = compare(onnx_actions, lerobot)
    log(f"ONNX chain vs LeRobot: {report['onnx_vs_lerobot']}")

    # ---- calibration: every graph's inputs over the observations
    if calibrate:
        captured: dict[str, list] = {}
        for index, o in enumerate(obs):
            reference.run(make_request(o, seed + 1 + index)[0], cameras, capture=captured)
            log(f"calibration observation {index + 1}/{len(obs)}")
        (out / "calibration").mkdir(exist_ok=True)
        for name in graphs:
            source = captured["vision" if name == "vision" else name]
            limit = 32 if name == "vision" else 16
            np.savez(out / "calibration" / f"{name}.npz", **calibration_rows(source, limit))
        log("calibration written")

    _finish_export(out, geometry, spec, stats, tokenizer, table, fixtures, valid, ids, task_text, client_files)
    report["seconds"] = round(time.time() - started)
    (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


# ------------------------------------------------------------------ LLiMa backend


def llima_freq_tables(tokens: int) -> tuple[np.ndarray, np.ndarray]:
    """RoPE inputs of LLiMa's `pre` graphs: cos and sin of positions 0..tokens-1 for the
    128 rotary frequencies (HF Gemma split-half convention), each [tokens, 128]."""
    inv = 1.0 / (modules.ROPE_THETA ** (np.arange(0, modules.HEAD_DIM, 2, dtype=np.float32) / modules.HEAD_DIM))
    angles = np.arange(tokens, dtype=np.float32)[:, None] * inv[None, :]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def llima_request(onnx_request: dict[str, np.ndarray], cameras: int, prefix: int) -> dict[str, np.ndarray]:
    """The LLiMa-backend wire request for the same observation as an ONNX-backend one:
    HWC images, prompt rows padded to the 1024-token group, padded keys masked."""
    request = {}
    for k in range(cameras):
        patches = onnx_request[f"image{k}"].reshape(16, 16, 3, 14, 14)  # (gy, gx, c, y, x)
        chw = patches.transpose(2, 0, 3, 1, 4).reshape(3, 224, 224)
        request[f"image{k}"] = np.ascontiguousarray(chw.transpose(1, 2, 0))
    rows = prefix - 256 * cameras
    language = np.zeros((rows, rt.WIDTH), np.float32)
    language[: onnx_request["language"].shape[0]] = onnx_request["language"]
    bias = np.full(prefix, modules.MASK, np.float32)
    bias[: onnx_request["prefix_bias"].shape[0]] = onnx_request["prefix_bias"]
    request.update(language=language, prefix_bias=bias, suffix_rope=onnx_request["suffix_rope"],
                   noise=onnx_request["noise"])
    return request


def write_llima_export(export: Path, out: Path, cameras: int) -> None:
    """LLiMa-backend fixture and constants derived from an ONNX-backend export of the
    same checkpoint (same observation, same LeRobot reference actions)."""
    prefix = rt.llima_prefix(cameras)
    names = [f"image{k}" for k in range(cameras)] + ["language", "prefix_bias", "suffix_rope", "noise"]
    shapes = {**{f"image{k}": (256 * 588,) for k in range(cameras)}, "language": (rt.LANGUAGE_TOKENS, rt.WIDTH),
              "prefix_bias": (-1,), "suffix_rope": (2 * rt.CHUNK, rt.ROW), "noise": (rt.CHUNK, rt.ACTION_LANE)}
    def convert(source: Path, target: Path) -> None:
        old = {n: np.fromfile(source / "inputs" / f"{n}.f32", "<f4").reshape(shapes[n]) for n in names}
        (target / "inputs").mkdir(parents=True, exist_ok=True)
        for name, value in llima_request(old, cameras, prefix).items():
            np.ascontiguousarray(value, dtype="<f4").tofile(target / "inputs" / f"{name}.f32")
        (target / "expected").mkdir(parents=True, exist_ok=True)
        (target / "expected/normalized_actions.f32").write_bytes(
            (source / "expected/normalized_actions.f32").read_bytes())

    convert(export / "fixtures", out / "fixtures")
    for sample in sorted((export / "fixtures" / "set").glob("s[0-9][0-9]")):
        convert(sample, out / "fixtures" / "set" / sample.name)
    constants = out / "constants"
    constants.mkdir(parents=True, exist_ok=True)
    for path in [*(export / "constants").glob("cond_*.f32"), *(export / "constants").glob("ada_*.f32")]:
        (constants / path.name).write_bytes(path.read_bytes())
    cos, sin = llima_freq_tables(prefix)
    cos.astype("<f4").tofile(constants / "freq_real.f32")
    sin.astype("<f4").tofile(constants / "freq_imag.f32")


def _finish_export(out, geometry, spec, stats, tokenizer, table, fixtures, valid, ids, task_text, client_files):
    """Client files for the robot side and the specialized profile."""
    # ---- client files (the robot client builds requests with these)
    if client_files:
        folder = out / "client"
        folder.mkdir(exist_ok=True)
        table.numpy().astype(np.float16).tofile(folder / "embedding.f16")
        tokenizer_json = Path(tokenizer.name_or_path)
        if not (tokenizer_json / "tokenizer.json").is_file():
            from huggingface_hub import hf_hub_download

            tokenizer_json = Path(hf_hub_download(TOKENIZER, "tokenizer.json")).parent
        (folder / "tokenizer.json").write_bytes((tokenizer_json / "tokenizer.json").read_bytes())
        np.savez(folder / "normalization_stats.npz",
                 **{k: v for k, v in stats.__dict__.items() if v is not None})
        (folder / "client.json").write_text(json.dumps({
            "cameras": len(geometry.cameras), "camera_keys": list(geometry.cameras), "state_dim": geometry.state_dim,
            "backend": geometry.backend, "prefix_tokens": geometry.prefix_tokens,
            "action_dim": geometry.action_dim, "task": task_text, "relative_actions": geometry.relative_actions,
            "relative_mask": [("gripper" not in n) for n in (geometry.joint_names or [""] * geometry.action_dim)],
        }, indent=2) + "\n")
        from tokenizers import Tokenizer

        fast = Tokenizer.from_file(str(folder / "tokenizer.json"))
        text = str(np.load(fixtures / "pi05_fixture.npz")["prompt"])
        if fast.encode(text).ids[:valid] != ids[:valid].tolist():
            raise RuntimeError("tokenizer.json does not reproduce the PaliGemma tokenizer's ids")
        log("client files written")

    profile = {"camera_keys": list(geometry.cameras), "state_dim": geometry.state_dim,
               "action_dim": geometry.action_dim, "camera_shape": list(geometry.camera_shape),
               "state_names": list(spec.dataset.state_names), "calibration_id": geometry.calibration_id,
               "camera_roles": [list(r) for r in geometry.roles],
               "runtime_context": dict(spec.runtime_context)}
    (out / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")


def _export_llima(checkpoint: Path, out: Path, w: dict, graphs: dict, reference: Reference, obs: list,
                  make_request, cameras: int, cond: list, calibrate: bool, seed: int,
                  expert_form: str = "nchw", denoise: str = "device") -> dict:
    """LLiMa backend: the PaliGemma folder LLiMa compiles, and only the expert as ONNX,
    sized for the 1024-token LLiMa group (padded K/V rows are masked keys).

    With the NCHW form and `denoise` "device" the ONNX is blocks.ExpertEulerNchw (it returns
    the next x; adaRMS as one row per norm), compiled with one persistent binding per input."""
    tokens = rt.llima_prefix(cameras)
    weights.paligemma_dir(checkpoint, out / "paligemma")
    log("PaliGemma folder for LLiMa written")
    step = blocks.ExpertStep(w, rt.EXPERT_LAYERS, tokens).eval()
    example = {"cache": torch.zeros(1, 1, rt.cache_rows(tokens), rt.ROW), "x": torch.zeros(1, 1, 50, 32),
               "cond": cond[0].reshape(1, 1, 1, rt.COND), "rope": torch.zeros(1, 1, 100, rt.ROW),
               "bias": torch.zeros(1, 1, 1, tokens + 50)}
    nchw = expert_form == "nchw"
    euler = nchw and denoise == "device"
    if nchw:  # GR00T N1.7's DiT form; the adaRMS modulation of each step becomes a sidecar
        expert = blocks.ExpertStepNchw(step).eval()
        example = dict(zip(rt.NCHW_EXPERT_INPUTS, blocks.to_nchw_inputs(*example.values(), step, tokens)))
        if euler:
            expert = blocks.ExpertEulerNchw(expert, rt.DT).eval()
            example["ada"] = blocks.euler_ada(example["ada"])
        (out / "constants").mkdir(exist_ok=True)
        for k, vector in enumerate(cond):
            blocks.ada_vectors(step, vector).numpy().astype("<f4").tofile(out / "constants" / f"ada_{k:02d}.f32")
    else:
        expert = step
    target = out / "onnx" / "expert.onnx"
    if not target.is_file():
        export_onnx(expert, example, target)
    log(f"onnx expert ({tokens}-row cache): {target.stat().st_size >> 20} MiB")
    captured: dict[str, list] = {}
    samples = obs if calibrate else obs[:1]
    for index, o in enumerate(samples):
        reference.run(make_request(o, seed + 1 + index)[0], cameras, capture=captured)
    rows = [pad_expert_row(r, tokens) for r in captured["expert"]]
    if nchw:
        rows = [dict(zip(rt.NCHW_EXPERT_INPUTS, (t.numpy() for t in blocks.to_nchw_inputs(
            *(torch.from_numpy(r[k]) for k in ("cache", "x", "cond", "rope", "bias")), step, tokens))))
            for r in rows]
        if euler:
            rows = [{**r, "ada": blocks.euler_ada(torch.from_numpy(r["ada"])).numpy()} for r in rows]
    import onnxruntime as ort

    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    small = graphs["expert"]
    first = captured["expert"][0]
    with torch.no_grad():
        want = small(*(torch.from_numpy(first[k]) for k in ("cache", "x", "cond", "rope", "bias"))).numpy()
    got = session.run(None, rows[0])[0]
    if euler:  # next x -> the velocity it implies
        got = (rows[0]["x"] - got) / rt.DT
    if nchw:  # [1, 32, 1, 50] -> [1, 1, 50, 32]
        got = got.reshape(32, 50).T.reshape(want.shape)
    report = {"expert_onnx_vs_modules": compare(got, want)}
    log(f"expert ONNX ({tokens} rows) vs modules ({reference.tokens} rows): {report['expert_onnx_vs_modules']}")
    if calibrate:
        (out / "calibration").mkdir(exist_ok=True)
        np.savez(out / "calibration" / "expert.npz", **calibration_rows(rows, 16))
        log(f"expert calibration: {min(16, len(rows))} rows from {len(samples)} observations")
    return report


def pad_expert_row(row: dict[str, np.ndarray], tokens: int) -> dict[str, np.ndarray]:
    """One expert input captured at the ONNX-backend prefix, padded to `tokens` rows per
    layer (zero K/V, masked keys) -- what the LLiMa prefill hands the expert."""
    p = row["bias"].shape[-1] - rt.CHUNK
    layers = row["cache"].shape[2] // p
    cache = np.zeros((1, 1, layers * tokens, rt.ROW), np.float32)
    cache.reshape(layers, tokens, rt.ROW)[:, :p] = row["cache"].reshape(layers, p, rt.ROW)
    bias = np.zeros((1, 1, 1, tokens + rt.CHUNK), np.float32)
    bias[..., :p] = row["bias"][..., :p]
    bias[..., p:tokens] = modules.MASK
    return {"cache": cache, "x": row["x"], "cond": row["cond"], "rope": row["rope"], "bias": bias}
