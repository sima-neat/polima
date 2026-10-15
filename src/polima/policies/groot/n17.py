"""GR00T N1.7 (Cosmos-Reason2 / Qwen3-VL backbone): static partitions, calibration and
the native interpreter plan, plus the compiled contract restored through the shared
GR00T policy (`specialize_spec`, `build_runtime_plan`).

`export_build` runs in the N1.7 GR00T environment, CPU only, against the official
policy (Isaac-GR00T N1.7 `Gr00tPolicy`):

1. samples: the dataset's first three episodes, 16 frames each with full history and
   horizon; the first two episodes are the calibration split;
2. reference: one official-policy call (sample 16, seed 123) with every partition
   boundary hooked -- trace inputs, plan constants and the bundle fixture;
3. partitions: vision (per camera, split at every deepstack tap), language (layers
   0, 1, 2 alone for the host deepstack adds, then pairs), vlln fed the last decoder
   layer as NVIDIA's Qwen3Backbone does (no final-norm graph), VL self-attention,
   state/action projections, 16 DiT pairs and the action tail, each checked against
   its PyTorch wrapper under onnxruntime;
4. calibration: 32 calibration-split observations x cameras x 4 denoising steps;
5. SmoothQuant (INT8-weight builds): alpha 0.6 on vision and language norms whose
   outputs feed only weight MatMul/Gemm consumers; FP32-checked;
6. runtime: plan, constants, fixture and robot profile for any group layout.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from polima.policies.base import GraphSpec, RuntimePlan, Step, TensorSpec


def processor_config(model_name: str) -> dict:
    """Read the backbone image processor from its directory or offline Hub cache."""
    model_name = os.environ.get("POLIMA_GROOT17_BACKBONE", "").strip() or model_name
    directory = Path(model_name).expanduser()
    path = directory / "preprocessor_config.json"
    if not path.is_file() and not directory.is_absolute():
        from huggingface_hub import hf_hub_download
        try:
            path = Path(hf_hub_download(model_name, "preprocessor_config.json", local_files_only=True))
        except (OSError, ValueError) as error:
            raise FileNotFoundError(
                f"Backbone processor unavailable for {model_name}. Download the complete backbone "
                "before compiling, using the same Hugging Face cache as Workbench."
            ) from error
    if not path.is_file():
        raise FileNotFoundError(f"Backbone processor not found: {path}. Restore the backbone assets referenced by the checkpoint or set POLIMA_GROOT17_BACKBONE to their local directory or Hub ID.")
    return json.loads(path.read_text())


@contextmanager
def checkpoint_assets(checkpoint: Path):
    """Override backbone references in a temporary view without copying weights."""
    checkpoint = checkpoint.resolve()
    backbone = os.environ.get("POLIMA_GROOT17_BACKBONE", "").strip()
    if not backbone:
        config = json.loads((checkpoint / "config.json").read_text())
        name = config.get("model_name", "")
        if name and Path(name).expanduser().is_absolute() and not Path(name).expanduser().is_dir():
            raise FileNotFoundError(
                "The checkpoint references a backbone directory that is unavailable on this host. "
                "Set POLIMA_GROOT17_BACKBONE to the matching local backbone directory or cached Hub ID."
            )
        yield checkpoint
        return
    if Path(backbone).expanduser().is_dir():
        backbone = str(Path(backbone).expanduser().absolute())
    elif Path(backbone).expanduser().is_absolute():
        raise FileNotFoundError(f"POLIMA_GROOT17_BACKBONE directory not found: {backbone}")
    with TemporaryDirectory(prefix="polima-groot17-") as temporary:
        view = Path(temporary) / "checkpoint"
        view.mkdir()
        if Path(backbone).is_dir() and not any(name in backbone for name in ("nvidia/Cosmos-Reason2", "Qwen/Qwen3-VL")):
            assets = json.loads((Path(backbone) / "config.json").read_text())
            if assets.get("model_type") != "qwen3_vl":
                raise ValueError("GR00T N1.7 requires a Qwen3-VL backbone directory.")
            alias = Path(temporary) / "Qwen/Qwen3-VL"
            alias.parent.mkdir()
            alias.symlink_to(backbone, target_is_directory=True)
            backbone = str(alias)
        for item in checkpoint.iterdir():
            (view / item.name).symlink_to(item, target_is_directory=item.is_dir())
        processor = Path("processor_config.json")
        if not (checkpoint / processor).is_file():
            processor = Path("processor/processor_config.json")
            (view / "processor").unlink()
            (view / "processor").mkdir()
            for item in (checkpoint / "processor").iterdir():
                (view / "processor" / item.name).symlink_to(item, target_is_directory=item.is_dir())
        for relative in (Path("config.json"), processor):
            target = view / relative
            config = json.loads(target.read_text())
            if relative.name == "processor_config.json":
                config.setdefault("processor_kwargs", {})["model_name"] = backbone
            else:
                config["model_name"] = backbone
            target.unlink()
            target.write_text(json.dumps(config))
        yield view


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------- compiled contract (plain)


def specialize_spec(spec, profile):
    context = dict(profile["runtime_context"])
    if context.get("model_version") != "1.7":
        raise ValueError("Expected an explicit N1.7 runtime contract")
    if context.get("image_preprocess", {}).get("format") != "groot-qwen3-image-v1":
        raise ValueError("N1.7 requires its Qwen3 image preprocessing contract")
    graphs = []
    for item in context["n17_graphs"]:
        hwc = item["name"].startswith(("vision_", "language_", "deepstack_")) or item["name"] == "vlln"
        graphs.append(GraphSpec(
            name=item["name"], builder="polima.policies.groot.n17:partition_export_required",
            inputs=tuple(TensorSpec(k, tuple(item["inputs"][k])) for k in item["input_order"]),
            outputs=(TensorSpec("output", tuple(item["output"])),), layout="NHWC",
            mla_tessellation=not hwc, external_dram_layout="HWC" if hwc else "compiler",
        ))
    names = tuple(profile["state_names"])
    cameras = tuple(profile["camera_keys"])
    width = int(profile["state_dim"])
    # The returned chunk length is the checkpoint's decoded horizon (16, 32, 40, 50 ...).
    horizon = int(profile["actions_per_chunk"])
    if len(names) != width or int(profile["action_dim"]) != width or not 0 < horizon <= 50 or not cameras:
        raise ValueError(f"inconsistent N1.7 robot profile: {profile}")
    plan = context["n17_plan"]
    if plan["buffers"].get(plan["result"]) != horizon * width:
        raise ValueError(f"N1.7 plan result {plan['result']} does not hold {horizon}x{width} actions")
    roles = tuple(k.rsplit(".", 1)[-1] for k in cameras)
    robot = replace(spec.robot, camera_roles=tuple((r, r) for r in roles), actions_per_chunk=horizon)
    if names != spec.robot.joint_names:  # any robot: joints and calibration from the profile
        robot = replace(robot, joint_names=names, calibration_id=profile["calibration_id"])
    return replace(
        spec,
        dataset=replace(spec.dataset, state_names=names, camera_keys=cameras, state_dim=width, action_dim=width),
        compile=replace(spec.compile, graphs=tuple(graphs),
                        export_entry="polima.policies.groot.n17:partition_export_required"),
        wire=replace(spec.wire, version=2, response_shape=(horizon, width), request_tensors=tuple(
            TensorSpec(v["name"], tuple(v["shape"])) for v in context["n17_request_tensors"])),
        robot=robot, runtime_context=context,
        runtime_plan_builder="polima.policies.groot.n17:build_runtime_plan",
    )


def build_runtime_plan(spec, context=None):
    raw = spec.runtime_context["n17_plan"]
    plan = RuntimePlan(buffers=raw["buffers"], steps=tuple(Step(v["op"], v["out"], v["args"]) for v in raw["steps"]),
                       result=raw["result"], sidecars=tuple(raw["sidecars"]))
    plan.validate("groot-n1.7", spec.compile)
    return plan


def partition_export_required(*args, **kwargs):
    raise ValueError("N1.7 partitions come from `polima compile --policy groot` (n17.export_build)")


# ------------------------------------------------------------ target modules (torch)


def _target_modules():
    import torch
    import torch.nn.functional as F
    from torch import nn

    class MatmulNorm(nn.Module):
        def __init__(self, source, rms=False):
            super().__init__()
            self.rms = rms
            width = source.weight.numel() if rms else source.normalized_shape[-1]
            self.epsilon = source.variance_epsilon if rms else source.eps
            device = next(source.parameters(), torch.empty(0)).device
            self.register_buffer("mean_column", torch.ones(width, 1, device=device) / width)
            self.register_buffer("weight", source.weight.detach().clone() if source.weight is not None else None)
            bias = getattr(source, "bias", None)
            self.register_buffer("bias", bias.detach().clone() if bias is not None else None)

        def forward(self, value):
            centered = value if self.rms else value - value @ self.mean_column
            normalized = centered / torch.sqrt((centered * centered) @ self.mean_column + self.epsilon)
            if self.weight is not None:
                normalized = normalized * self.weight
            if self.bias is not None:
                normalized = normalized + self.bias
            return normalized

    class HeadRMSNorm(nn.Module):
        """Per-head RMS norm in the ReduceMean form ModelSDK fuses into one BF16 kernel (the
        MatMul-mean form leaves a standalone Sqrt that ModelSDK 2.1.3 lowers to INT8)."""

        def __init__(self, source):
            super().__init__()
            self.epsilon = source.variance_epsilon
            self.register_buffer("weight", source.weight.detach().clone())

        def forward(self, value):
            return value / torch.sqrt((value * value).mean(-1, keepdim=True) + self.epsilon) * self.weight

    def replace_norms(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.LayerNorm):
                setattr(module, name, MatmulNorm(child))
            elif child.__class__.__name__ == "Qwen3VLTextRMSNorm":
                setattr(module, name, HeadRMSNorm(child) if name in ("q_norm", "k_norm") else MatmulNorm(child, rms=True))
            else:
                replace_norms(child)
        return module

    def clone(module):
        return replace_norms(copy.deepcopy(module)).float().eval().requires_grad_(False)

    class VisionPatch(nn.Module):
        def __init__(self, vision, grid):
            super().__init__()
            projection = vision.patch_embed.proj
            self.register_buffer("weight", projection.weight.detach().flatten(1).clone())
            self.register_buffer("bias", projection.bias.detach().clone())
            with torch.no_grad():
                position = vision.fast_pos_embed_interpolate(grid)
            self.register_buffer("position", position[None, None])

        def forward(self, patches):
            return F.linear(patches, self.weight, self.bias) + self.position

    class VisionGroup(nn.Module):
        def __init__(self, vision, start, stop, grid, helpers):
            super().__init__()
            self.blocks = nn.ModuleList([clone(block) for block in vision.blocks[start:stop]])
            with torch.no_grad():
                rotary = vision.rot_pos_emb(grid)
                frequencies = torch.cat((rotary, rotary), dim=-1)
            self.register_buffer("cos", frequencies.cos())
            self.register_buffer("sin", frequencies.sin())
            helpers._patch_vision_attention_for_export(self, chunk_sizes=[int(grid[0].prod())])

        def forward(self, hidden):
            hidden = hidden[0, 0]
            for block in self.blocks:
                hidden = block(hidden, cu_seqlens=None, position_embeddings=(self.cos, self.sin))
            return hidden[None, None]

    class VisionMerge(nn.Module):
        def __init__(self, merger):
            super().__init__()
            self.merger = clone(merger)

        def forward(self, hidden):
            return self.merger(hidden[0, 0])[None, None]

    class LanguageGroup(nn.Module):
        def __init__(self, language, start, stop):
            super().__init__()
            self.layers = nn.ModuleList([clone(layer) for layer in language.layers[start:stop]])
            for layer in self.layers:
                layer.self_attn.config._attn_implementation = "eager"

        def forward(self, hidden, attention_mask, cos, sin):
            hidden = hidden[:, 0]
            mask = attention_mask.permute(0, 3, 1, 2)
            for layer in self.layers:
                hidden = layer(hidden, attention_mask=mask, position_embeddings=(cos[:, 0], sin[:, 0]),
                               past_key_values=None)
            return hidden[:, None]

    class VLGroup(nn.Module):
        def __init__(self, blocks):
            super().__init__()
            self.blocks = nn.ModuleList([clone(block) for block in blocks])

        def forward(self, hidden):
            hidden = hidden[:, 0]
            for block in self.blocks:
                hidden = block(hidden)
            return hidden[:, None]

    class CategoryLinear(nn.Module):
        def __init__(self, linear, embodiment):
            super().__init__()
            self.register_buffer("weight", linear.W[embodiment].detach().clone())
            self.register_buffer("bias", linear.b[embodiment].detach().clone())

        def forward(self, value):
            return value @ self.weight + self.bias

    class StateProject(nn.Module):
        def __init__(self, head, embodiment):
            super().__init__()
            self.layer1 = CategoryLinear(head.state_encoder.layer1, embodiment)
            self.layer2 = CategoryLinear(head.state_encoder.layer2, embodiment)

        def forward(self, state):
            return self.layer2(F.relu(self.layer1(state)))

    class ActionProject(nn.Module):
        def __init__(self, head, embodiment):
            super().__init__()
            self.w1 = CategoryLinear(head.action_encoder.W1, embodiment)
            self.w2 = CategoryLinear(head.action_encoder.W2, embodiment)
            self.w3 = CategoryLinear(head.action_encoder.W3, embodiment)
            self.register_buffer("positions",
                                 head.position_embedding.weight[:head.action_horizon].detach()[None, None].clone())

        def forward(self, actions, tau):
            hidden = self.w2(torch.cat((self.w1(actions), tau), dim=-1))
            return self.w3(F.silu(hidden)) + self.positions

    class DiTPair(nn.Module):
        def __init__(self, dit, start):
            super().__init__()
            self.cross = clone(dit.transformer_blocks[start])
            self.self_attention = clone(dit.transformer_blocks[start + 1])

        def forward(self, hidden, backbone, attention_mask, temb):
            hidden = self.cross(hidden[:, 0], encoder_hidden_states=backbone[:, 0],
                                encoder_attention_mask=(1.0 - attention_mask[:, 0, :, 0]) * -10000.0,
                                temb=temb[:, 0, 0])
            return self.self_attention(hidden, temb=temb[:, 0, 0])[:, None]

    class ActionTail(nn.Module):
        def __init__(self, head, embodiment):
            super().__init__()
            self.norm = MatmulNorm(head.model.norm_out)
            self.proj1 = copy.deepcopy(head.model.proj_out_1)
            self.proj2 = copy.deepcopy(head.model.proj_out_2)
            self.layer1 = CategoryLinear(head.action_decoder.layer1, embodiment)
            self.layer2 = CategoryLinear(head.action_decoder.layer2, embodiment)
            padding = (-self.layer2.bias.shape[-1]) % 16  # pad the fixed weights: no target-side Pad
            self.layer2.weight = F.pad(self.layer2.weight, (0, padding))
            self.layer2.bias = F.pad(self.layer2.bias, (0, padding))
            self.horizon = head.action_horizon

        def forward(self, hidden, temb):
            shift, scale = self.proj1(F.silu(temb)).chunk(2, dim=-1)
            hidden = self.proj2(self.norm(hidden) * (1 + scale) + shift)
            return self.layer2(F.relu(self.layer1(hidden)))[:, :, -self.horizon:]

    return {"clone": clone, "MatmulNorm": MatmulNorm, "VisionPatch": VisionPatch, "VisionGroup": VisionGroup,
            "VisionMerge": VisionMerge, "LanguageGroup": LanguageGroup, "VLGroup": VLGroup,
            "StateProject": StateProject, "ActionProject": ActionProject, "DiTPair": DiTPair,
            "ActionTail": ActionTail}


def clone(module):
    return _target_modules()["clone"](module)


def language_ranges(depth: int):
    # Deepstack additions follow layers 0, 1, 2 and stay explicit host operations.
    return [(i, i + 1) for i in range(min(3, depth))] + [(i, min(i + 2, depth)) for i in range(3, depth, 2)]


# --------------------------------------------------------------------- policy (torch)


def load_policy(checkpoint: Path, source: Path, embodiment: str | None = None):
    for path in (str(source / "scripts/deployment"), str(source)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    from polima.policies.groot.profile import checkpoint_metadata

    embodiment = embodiment or checkpoint_metadata(checkpoint, image_preprocess=False)["embodiment_tag"]
    with checkpoint_assets(checkpoint) as model_path:
        return Gr00tPolicy(embodiment, str(model_path), device="cpu")


def _samples(dataset: Path, video_deltas, horizon: int) -> list[dict]:
    episodes = [json.loads(line) for line in (dataset / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    if len(episodes) < 3:
        raise ValueError("N1.7 calibration needs at least three episodes")
    history = max(0, -min(video_deltas))
    samples = []
    for number, episode in enumerate(episodes[:3]):
        last = episode["length"] - horizon
        if last - history + 1 < 16:
            raise ValueError(f"episode {episode['episode_index']} is too short for 16 samples")
        samples += [{"episode_index": episode["episode_index"], "frame_index": history + i * (last - history) // 15,
                     "split": "calibration" if number < 2 else "evaluation"} for i in range(16)]
    return samples


def _record(arrays: dict, prefix: str, value) -> None:
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        arrays[prefix] = (tensor.float() if tensor.dtype == torch.bfloat16 else tensor).numpy()
    elif isinstance(value, np.ndarray):
        arrays[prefix] = value
    elif isinstance(value, dict) or hasattr(value, "items"):
        for key, item in value.items():
            _record(arrays, f"{prefix}.{key}", item)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _record(arrays, f"{prefix}.{index}", item)


def _observation(policy, loader, sample, episodes):
    import numpy as np
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data

    episode = sample["episode_index"]
    if episode not in episodes:
        episodes[episode] = loader[episode]
    step = extract_step_data(episodes[episode], sample["frame_index"], policy.get_modality_config(),
                             policy.embodiment_tag)
    return step, {"video": {k: np.stack(v)[None] for k, v in step.images.items()},
                  "state": {k: v[None] for k, v in step.states.items()},
                  "language": {policy.language_key: [[step.text]]}}


def _stages(policy, *, everything: bool) -> dict:
    qwen, head = policy.model.backbone.model.model, policy.model.action_head
    stages = {"vision": qwen.visual, "language": qwen.language_model, "vlln": head.vlln,
              "state_encoder": head.state_encoder, "action_encoder": head.action_encoder,
              "vision_merger": qwen.visual.merger}
    if everything:
        stages.update({"vl_attention": head.vl_self_attention, "dit": head.model,
                       "action_decoder": head.action_decoder, "language_norm": qwen.language_model.norm})
    stages.update({f"vision_block_{i}": b for i, b in enumerate(qwen.visual.blocks)})
    stages.update({f"language_layer_{i}": b for i, b in enumerate(qwen.language_model.layers)})
    stages.update({f"vl_block_{i}": b for i, b in enumerate(head.vl_self_attention.transformer_blocks)})
    stages.update({f"dit_block_{i}": b for i, b in enumerate(head.model.transformer_blocks)})
    return stages


# --------------------------------------------------------------------- export (torch)


def export_build(checkpoint: Path, dataset: Path, out: Path, *, source: Path, smooth: bool) -> None:
    from polima.policies.groot.profile import checkpoint_metadata

    if out.exists():
        shutil.rmtree(out)  # the stage restarts from scratch
    out.mkdir(parents=True)
    meta = checkpoint_metadata(checkpoint, dataset, image_preprocess=False)
    policy = load_policy(checkpoint, source, meta["embodiment_tag"])
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

    modality = policy.get_modality_config()
    samples = _samples(dataset, list(modality["video"].delta_indices), meta["actions_per_chunk"])
    # A read-only view: the loader must not write into the customer's dataset.
    view = out / "dataset"
    (view / "meta").mkdir(parents=True)
    for path in sorted((dataset / "meta").iterdir()):
        (view / "meta" / path.name).symlink_to(path)
    for name in ("data", "videos"):
        (view / name).symlink_to(dataset / name, target_is_directory=True)
    loader = LeRobotEpisodeLoader(str(view), modality_configs=modality)
    episodes: dict = {}
    (out / "samples.json").write_text(json.dumps(samples, indent=2) + "\n")
    data = _capture_reference(policy, loader, samples[16], episodes, out / "reference")
    graphs = _export_partitions(policy, source, data, out, meta["embodiment_id"])
    _capture_calibration(policy, loader, samples, episodes, graphs, out / "calibration-multi", count=32)
    if smooth:
        smooth_partitions(out, out / "calibration-multi", out / "smooth", alpha=0.6,
                          pattern=r"language_\d\d_\d\d|vision_\d\d_\d\d")
    write_runtime(checkpoint, out, policy, graphs, data, meta)
    report = verify_export(out)
    print(f"ONNX validation: {'PASS' if report['ok'] else 'FAIL'}; "
          f"cosine={report['cosine']:.6f} (minimum {report['cosine_min']}), "
          f"mean_abs={report['mean_abs']:.6f} (maximum {report['mean_abs_max']}).", flush=True)
    if not report["ok"]:
        print("WARNING: Numerical validation failed. Compilation will continue; "
              "the failed result is saved in onnx_verification_report.json.", flush=True)


def verify_export(out: Path) -> dict:
    """Replay the complete ONNX policy against its recorded PyTorch action chunk."""
    import numpy as np
    import onnxruntime as ort

    from polima.deploy.smoke import compare_normalized_actions
    from polima.wire.server_stub import StubPlan

    root = out / "runtime"
    context = json.loads((root / "profile.json").read_text())["runtime_context"]
    raw = context["n17_plan"]
    for step in raw["steps"]:
        if step["op"] in ("normalize", "denormalize"):
            for key in ("mean", "std"):
                step["args"][key] += ".f32"
        for part in step.get("args", {}).get("parts", []):
            if part.get("sidecar"):
                part["src"] += ".f32"
    plan = StubPlan(raw["buffers"], raw["steps"], raw["result"], constants_dir=root / "constants")
    graphs = {graph["name"]: graph for graph in context["n17_graphs"]}
    sessions, calls = {}, 0
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 4, 1

    def run_graph(name, joined):
        nonlocal calls
        graph, feeds, offset = graphs[name], {}, 0
        for key in graph["input_order"]:
            shape = graph["inputs"][key]
            size = int(np.prod(shape))
            feeds[key] = joined[offset:offset + size].reshape(shape)
            offset += size
        if offset != joined.size:
            raise ValueError(f"{name}: ONNX input size differs from the runtime plan")
        if name not in sessions:
            sessions[name] = ort.InferenceSession(str(out / graph["file"]), sess_options=options,
                                                  providers=["CPUExecutionProvider"])
        calls += 1
        return sessions[name].run(None, feeds)[0]

    inputs = {item["name"]: np.fromfile(root / "fixtures/inputs" / f"{item['name']}.f32", "<f4")
              for item in context["n17_request_tensors"]}
    actual = plan.run(inputs, graph_fn=run_graph)
    expected = np.fromfile(root / "fixtures/expected/normalized_actions.f32", "<f4")
    with np.load(root / "normalization_stats.npz") as stats:
        report = compare_normalized_actions(
            actual, expected, stats["action_mean"], stats["action_std"], state=inputs["state"],
            relative_dim=context.get("relative_action_dim", 0),
            cosine_min=0.98, mean_abs_max=0.015).to_dict()
    error = np.abs(actual - expected)
    report.update(metric_space="normalized actions", graph_calls=calls,
                  physical_mean_abs=float(error.mean()), physical_max_abs=float(error.max()),
                  strict_physical_allclose=bool(np.allclose(actual, expected, atol=0.01, rtol=0.01)))
    (out / "onnx_verification_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _capture_reference(policy, loader, sample, episodes, out: Path) -> dict:
    """One official call with every partition boundary hooked (seed 123)."""
    import numpy as np
    import torch

    torch.set_num_threads(8)
    out.mkdir(parents=True)
    step, observation = _observation(policy, loader, sample, episodes)
    if not step.text.strip():
        raise ValueError("the reference sample has no instruction")
    arrays, counts = {}, {}
    _record(arrays, "raw", observation)

    def hook(name):
        def capture(module, positional, kwargs, result):
            number = counts.get(name, 0)
            counts[name] = number + 1
            _record(arrays, f"{name}.{number}.args", positional)
            _record(arrays, f"{name}.{number}.kwargs", kwargs)
            _record(arrays, f"{name}.{number}.output", result)
        return capture

    handles = [m.register_forward_hook(hook(n), with_kwargs=True) for n, m in _stages(policy, everything=True).items()]
    torch.manual_seed(123)
    try:
        with torch.inference_mode():
            actions, _ = policy.get_action(observation)
    finally:
        for handle in handles:
            handle.remove()
    _record(arrays, "final_actions", actions)
    np.savez(out / "reference.npz", **arrays)
    (out / "reference.json").write_text(json.dumps({"sample": sample, "instruction": step.text, "seed": 123,
                                                    "stage_calls": counts}, indent=2) + "\n")
    with np.load(out / "reference.npz", allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _export_partitions(policy, source: Path, data: dict, out: Path, embodiment: int) -> list[dict]:
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    import export_onnx_n1d7 as helpers  # NVIDIA's scripts/deployment, on sys.path via load_policy

    m = _target_modules()
    qwen, head = policy.model.backbone.model.model, policy.model.action_head
    (out / "onnx").mkdir()
    (out / "calibration").mkdir()
    graphs = []
    tensor = lambda key: torch.from_numpy(data[key]).float()

    def hidden(name):
        for key in (f"{name}.0.args.0", f"{name}.0.kwargs.hidden_states"):
            if key in data:
                return tensor(key)
        raise KeyError(name)

    def export(name, factory, inputs):
        module = factory().cpu().float().eval().requires_grad_(False)
        inputs = {k: v.cpu().float() for k, v in inputs.items()}
        path = out / "onnx" / f"{name}.onnx"
        with torch.inference_mode():
            expected = module(*inputs.values()).numpy()
            torch.onnx.export(module, tuple(inputs.values()), str(path), input_names=list(inputs),
                              output_names=["output"], opset_version=17, dynamo=False, do_constant_folding=True)
        onnx.checker.check_model(str(path))
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
        arrays = {k: v.numpy() for k, v in inputs.items()}
        actual = ort.InferenceSession(str(path), sess_options=options,
                                      providers=["CPUExecutionProvider"]).run(None, arrays)[0]
        error = np.abs(actual - expected)
        ok = bool(np.isfinite(actual).all() and np.allclose(actual, expected, atol=1e-3, rtol=1e-3))
        np.savez(out / "calibration" / f"{name}.npz", **{k: v[None] for k, v in arrays.items()})
        graphs.append({"name": name, "file": f"onnx/{name}.onnx", "sha256": _sha(path),
                       "inputs": {k: list(v.shape) for k, v in arrays.items()}, "output": list(actual.shape),
                       "onnx_vs_wrapper": {"ok": ok, "max_abs": float(error.max()), "mean_abs": float(error.mean())}})
        print(f"{name}: ok={ok} max_abs={error.max():.6g}", flush=True)
        if not ok:
            raise ValueError(f"{name}: ONNX differs from its PyTorch wrapper")

    grid = torch.from_numpy(data["vision.0.kwargs.grid_thw"])
    if not torch.all(grid == grid[0]):
        raise ValueError("per-camera graph reuse requires identical camera grids")
    one_grid = grid[:1]
    patches = int(one_grid[0].prod())
    export("vision_patch", lambda: m["VisionPatch"](qwen.visual, one_grid),
           {"patches": tensor("vision.0.args.0")[:patches][None, None]})
    taps = set(qwen.visual.deepstack_visual_indexes)
    start = 0
    while start < len(qwen.visual.blocks):
        stop = start + 1 if start in taps else min(start + 2, len(qwen.visual.blocks))
        export(f"vision_{start:02d}_{stop - 1:02d}",
               lambda a=start, b=stop: m["VisionGroup"](qwen.visual, a, b, one_grid, helpers),
               {"hidden": hidden(f"vision_block_{start}")[:patches][None, None]})
        start = stop
    export("vision_merge", lambda: m["VisionMerge"](qwen.visual.merger),
           {"hidden": hidden("vision_merger")[:patches][None, None]})
    for i, tap in enumerate(qwen.visual.deepstack_visual_indexes):
        export(f"deepstack_merge_{i}", lambda i=i: m["VisionMerge"](qwen.visual.deepstack_merger_list[i]),
               {"hidden": tensor(f"vision_block_{tap}.0.output")[:patches][None, None]})
    for start, stop in language_ranges(len(qwen.language_model.layers)):
        prefix = f"language_layer_{start}.0.kwargs"
        mask = tensor(prefix + ".attention_mask") if prefix + ".attention_mask" in data else \
            _causal_mask(tensor("language.0.kwargs.attention_mask"))
        mask = torch.nan_to_num(mask, neginf=-10000.0).clamp(min=-10000.0)
        export(f"language_{start:02d}_{stop - 1:02d}",
               lambda a=start, b=stop: m["LanguageGroup"](qwen.language_model, a, b),
               {"hidden": hidden(f"language_layer_{start}")[:, None], "attention_mask": mask.permute(0, 2, 3, 1),
                "cos": tensor(prefix + ".position_embeddings.0")[:, None],
                "sin": tensor(prefix + ".position_embeddings.1")[:, None]})
    export("vlln", lambda: m["MatmulNorm"](head.vlln), {"value": hidden("vlln")[:, None]})
    for i, block in enumerate(head.vl_self_attention.transformer_blocks):
        export(f"vl_{i:02d}", lambda b=block: m["VLGroup"]([b]), {"hidden": hidden(f"vl_block_{i}")[:, None]})
    export("state_project", lambda: m["StateProject"](head, embodiment), {"state": hidden("state_encoder")[:, None]})
    for i in range(0, len(head.model.transformer_blocks), 2):
        prefix = f"dit_block_{i}.0.kwargs"
        export(f"dit_{i:02d}_{i + 1:02d}", lambda i=i: m["DiTPair"](head.model, i),
               {"hidden": hidden(f"dit_block_{i}")[:, None], "backbone": tensor(prefix + ".encoder_hidden_states")[:, None],
                "attention_mask": tensor(prefix + ".encoder_attention_mask")[:, None, :, None],
                "temb": tensor(prefix + ".temb")[:, None, None]})
    with torch.inference_mode():
        timestep = torch.from_numpy(data["action_encoder.0.args.1"])
        tau = head.action_encoder.pos_encoding(timestep[:, None].expand(-1, head.action_horizon)).float().cpu()
    export("action_project", lambda: m["ActionProject"](head, embodiment),
           {"actions": hidden("action_encoder")[:, None], "tau": tau[:, None]})
    last = len(head.model.transformer_blocks) - 1
    export("action_tail", lambda: m["ActionTail"](head, embodiment),
           {"hidden": tensor(f"dit_block_{last}.0.output")[:, None],
            "temb": tensor("dit_block_0.0.kwargs.temb")[:, None, None]})
    (out / "partition_manifest.json").write_text(json.dumps(
        {"model_version": "1.7", "backbone_boundary": "native: vlln(last decoder layer output), no language_norm",
         "graphs": graphs}, indent=2) + "\n")
    return graphs


def _causal_mask(padding):
    import torch

    length = padding.shape[-1]  # native SDPA omits explicit masks for an unpadded causal sequence
    mask = torch.triu(torch.full((1, 1, length, length), -10000.0), diagonal=1)
    return mask + (1.0 - padding[:, None, None, :]) * -10000.0


def _capture_calibration(policy, loader, samples, episodes, graphs, out: Path, count: int) -> None:
    """Partition inputs over `count` calibration-split observations, every camera and step."""
    import numpy as np
    import torch

    torch.set_num_threads(4)
    out.mkdir(parents=True)
    expected = {g["name"]: g["inputs"] for g in graphs}
    qwen, head = policy.model.backbone.model.model, policy.model.action_head
    captured: dict = {}

    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().float().cpu().numpy().copy()
        if isinstance(value, (list, tuple)):
            return tuple(cpu(x) for x in value)
        if isinstance(value, dict):
            return {k: cpu(v) for k, v in value.items()}
        return value

    def hook(name):
        def receive(module, positional, kwargs, result):
            captured.setdefault(name, []).append((cpu(positional), cpu(kwargs), cpu(result)))
        return receive

    rows = {name: [] for name in expected}

    def add(name, **feeds):
        if feeds.keys() != expected[name].keys():
            raise ValueError(f"{name}: input names differ")
        clean = {}
        for key, shape in expected[name].items():
            value = np.ascontiguousarray(feeds[key], dtype=np.float32)
            if list(value.shape) != shape or not np.isfinite(value).all():
                raise ValueError(f"{name}:{key} calibration shape/finite mismatch {value.shape} vs {shape}")
            clean[key] = value
        rows[name].append(clean)

    def hidden(name, call=0):
        positional, kwargs, _ = captured[name][call]
        return positional[0] if positional else kwargs["hidden_states"]

    handles = [m.register_forward_hook(hook(n), with_kwargs=True) for n, m in _stages(policy, everything=False).items()]
    used = []
    try:
        for sample in samples:
            if sample["split"] != "calibration" or len(used) >= count:
                continue
            step, observation = _observation(policy, loader, sample, episodes)
            if not step.text.strip():
                continue
            captured.clear()
            seed = 123 + len(used)
            torch.manual_seed(seed)
            with torch.inference_mode():
                policy.get_action(observation)
            counts = [int(np.prod(x)) for x in captured["vision"][0][1]["grid_thw"]]
            patches = counts[0]
            for camera in range(len(counts)):
                cut = slice(camera * patches, (camera + 1) * patches)
                add("vision_patch", patches=hidden("vision")[cut][None, None])
                for name in expected:
                    if re.fullmatch(r"vision_\d+_\d+", name):
                        add(name, hidden=hidden(f"vision_block_{int(name.split('_')[1])}")[cut][None, None])
                add("vision_merge", hidden=hidden("vision_merger")[cut][None, None])
                for i, tap in enumerate(qwen.visual.deepstack_visual_indexes):
                    add(f"deepstack_merge_{i}", hidden=captured[f"vision_block_{tap}"][0][2][cut][None, None])
            for name in expected:
                if re.fullmatch(r"language_\d+_\d+", name):
                    stage = f"language_layer_{int(name.split('_')[1])}"
                    kw = captured[stage][0][1]
                    mask = kw.get("attention_mask")
                    if mask is None:
                        padding = captured["language"][0][1]["attention_mask"]
                        length = padding.shape[-1]
                        mask = np.triu(np.full((1, 1, length, length), -10000., dtype=np.float32), k=1) \
                            + (1 - padding[:, None, None, :]) * -10000.
                    mask = np.maximum(np.nan_to_num(mask, neginf=-10000.), -10000.)
                    cos, sin = kw["position_embeddings"]
                    add(name, hidden=hidden(stage)[:, None], attention_mask=mask.transpose(0, 2, 3, 1),
                        cos=cos[:, None], sin=sin[:, None])
            add("vlln", value=hidden("vlln")[:, None])
            for name in expected:
                if re.fullmatch(r"vl_\d+", name):
                    add(name, hidden=hidden(f"vl_block_{int(name.split('_')[1])}")[:, None])
            add("state_project", state=hidden("state_encoder")[:, None])
            for index in range(len(captured["action_encoder"])):
                action_args = captured["action_encoder"][index][0]
                with torch.inference_mode():
                    tau = head.action_encoder.pos_encoding(torch.from_numpy(action_args[1])[:, None].expand(
                        -1, head.action_horizon)).float().cpu().numpy()
                add("action_project", actions=action_args[0][:, None], tau=tau[:, None])
                for name in expected:
                    if re.fullmatch(r"dit_\d+_\d+", name):
                        stage = f"dit_block_{int(name.split('_')[1])}"
                        kw = captured[stage][index][1]
                        add(name, hidden=hidden(stage, index)[:, None], backbone=kw["encoder_hidden_states"][:, None],
                            attention_mask=kw["encoder_attention_mask"][:, None, :, None], temb=kw["temb"][:, None, None])
                last = len(head.model.transformer_blocks) - 1
                add("action_tail", hidden=captured[f"dit_block_{last}"][index][2][:, None],
                    temb=captured["dit_block_0"][index][1]["temb"][:, None, None])
            used.append({**sample, "seed": seed, "instruction": step.text})
            print(f"calibration {len(used)}/{count}: episode {sample['episode_index']} frame {sample['frame_index']}",
                  flush=True)
    finally:
        for handle in handles:
            handle.remove()
    for name, samples_ in rows.items():
        if not samples_:
            raise ValueError(f"no calibration rows for {name}")
        np.savez(out / f"{name}.npz", **{k: np.stack([s[k] for s in samples_]) for k in samples_[0]})
    (out / "manifest.json").write_text(json.dumps({"observations": used}, indent=2) + "\n")


# ----------------------------------------------------------------- SmoothQuant (onnx)


def smooth_partitions(export: Path, calibration: Path, out: Path, *, alpha: float, pattern: str) -> None:
    """Divide each norm's affine weight (and bias) by s and scale the input channels of its
    weight-only consumers by s, s_c = max|x_c|^alpha / max|W_c|^(1-alpha); FP32-checked.
    ModelSDK quantizes INT8-weight/BF16-activation inputs per token, so one large channel
    otherwise sets a coarse step for every channel of that token."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import helper, numpy_helper

    (out / "onnx").mkdir(parents=True)
    manifest = json.loads((export / "partition_manifest.json").read_text())

    def session(model):
        options = ort.SessionOptions()
        options.log_severity_level, options.intra_op_num_threads = 3, 8
        return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])

    def samples(path):
        with np.load(path) as archive:
            arrays = {k: archive[k] for k in archive.files}
        for index in range(len(next(iter(arrays.values())))):
            yield {k: v[index].astype(np.float32) for k, v in arrays.items()}

    def axis(node, weight):  # input-channel axis of a MatMul/Gemm weight
        if node.op_type == "Gemm":
            return 1 if any(a.name == "transB" and a.i == 1 for a in node.attribute) else 0
        return weight.ndim - 2

    for row in manifest["graphs"]:
        target = out / row["file"]
        model = onnx.load(str(export / row["file"])) if re.fullmatch(pattern, row["name"]) else None
        groups = []
        if model is not None:
            inits = {i.name: i for i in model.graph.initializer}
            consumers: dict = {}
            for node in model.graph.node:
                for name in node.input:
                    consumers.setdefault(name, []).append(node)

            def vector(node, inits=inits):
                found = [i for i in node.input if i in inits and len(inits[i].dims) == 1]
                return found[0] if len(found) == 1 and len(node.input) == 2 else None

            for node in model.graph.node:
                if node.op_type != "Mul" or "norm" not in node.name or vector(node) is None:
                    continue
                bias, output = None, node.output[0]
                users = consumers.get(output, [])
                if len(users) == 1 and users[0].op_type == "Add" and vector(users[0]) is not None:
                    bias, output = users[0], users[0].output[0]
                    users = consumers.get(output, [])
                if users and all(u.op_type in ("MatMul", "Gemm") and u.input[0] == output and u.input[1] in inits
                                 for u in users):
                    groups.append((node, bias, users, output))
        if not groups:
            target.hardlink_to(export / row["file"])
            continue
        probe = onnx.ModelProto()
        probe.CopyFrom(model)
        for *_, output in groups:
            probe.graph.output.append(helper.make_tensor_value_info(output, onnx.TensorProto.FLOAT, None))
        run, absmax = session(probe), None
        for feeds in samples(calibration / f"{row['name']}.npz"):
            current = [np.abs(v.reshape(-1, v.shape[-1])).max(0) for v in run.run(None, feeds)[1:]]
            absmax = current if absmax is None else [np.maximum(a, c) for a, c in zip(absmax, current)]
        result = onnx.ModelProto()
        result.CopyFrom(model)
        inits = {i.name: i for i in result.graph.initializer}
        for (scale_node, bias_node, users, _), activation in zip(groups, absmax):
            weights = [numpy_helper.to_array(inits[u.input[1]]) for u in users]
            weight_absmax = np.max([np.abs(w).max(axis=1 - axis(u, w) if w.ndim == 2 else -1)
                                    for u, w in zip(users, weights)], axis=0)
            s = np.clip(np.maximum(activation, 1e-5) ** alpha / np.maximum(weight_absmax, 1e-5) ** (1 - alpha),
                        1e-4, 1e4).astype(np.float32)
            for node in (scale_node, bias_node):
                if node is not None:
                    name = next(i for i in node.input if i in inits and len(inits[i].dims) == 1)
                    value = numpy_helper.to_array(inits[name])
                    inits[name].CopyFrom(numpy_helper.from_array((value / s).astype(np.float32), name))
            for user, weight in zip(users, weights):
                shape = [1] * weight.ndim
                shape[axis(user, weight)] = -1
                inits[user.input[1]].CopyFrom(
                    numpy_helper.from_array((weight * s.reshape(shape)).astype(np.float32), user.input[1]))
        reference, candidate = session(model), session(result)
        worst = 0.0
        for feeds in samples(calibration / f"{row['name']}.npz"):
            a, b = reference.run(None, feeds)[0], candidate.run(None, feeds)[0]
            worst = max(worst, float(np.linalg.norm(b - a) / np.linalg.norm(a)))
        if worst > 1e-5:
            raise ValueError(f"{row['name']}: smoothed graph not FP32-equivalent (rel L2 {worst:.3g})")
        onnx.save(result, str(target))
        print(f"smooth {row['name']}: {len(groups)} groups, FP32 rel L2 {worst:.3g}", flush=True)


# ------------------------------------------------------------------ runtime (numpy)


def write_runtime(checkpoint: Path, out: Path, policy, graphs: list[dict], data: dict, meta: dict) -> None:
    """Native-interpreter plan, constants, fixture and robot profile (out/runtime/)."""
    import numpy as np
    import torch
    from gr00t.model.modules.embodiment_conditioned_mlp import SinusoidalPositionalEncoding

    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("apply_sincos_state_encoding"):
        raise ValueError("sin/cos state encoding is not supported by the N1.7 runtime plan")
    lane, action_lane = int(config["max_state_dim"]), int(config["max_action_dim"])
    padded = action_lane + (-action_lane % 16)
    keys, widths = meta["state_keys"], meta["group_widths"]
    width, relative = sum(widths), meta["relative_action_dim"]
    steps_count = int(config["num_inference_timesteps"])
    root = out / "runtime"
    (root / "constants").mkdir(parents=True)
    by = {g["name"]: g for g in graphs}
    buffers, steps, sidecars = {}, [], []

    def alloc(name, size):
        if name in buffers and buffers[name] != int(size):
            raise ValueError(name)
        buffers[name] = int(size)
        return name

    def step(op, out_, **kwargs):
        steps.append({"op": op, "out": out_, "args": kwargs})

    def constant(name, value):
        value = np.asarray(value, dtype="<f4").ravel()
        value.tofile(root / "constants" / f"{name}.f32")
        sidecars.append(name)
        alloc(name, len(value))
        step("pack", name, parts=[{"src": name, "sidecar": True, "count": len(value), "dst_offset": 0}])
        return name

    def call(graph, out_, **feeds):
        alloc(out_, np.prod(by[graph]["output"]))
        step("run_elf", out_, graph=graph, **{"in": [feeds[k] for k in by[graph]["inputs"]]})
        return out_

    def part(src, count, dst=0, offset=0):
        return {"src": src, "count": int(count), "dst_offset": int(dst), "src_offset": int(offset)}

    fixtures = {}
    for name in by:
        with np.load(out / "calibration" / f"{name}.npz") as z:
            fixtures[name] = {k: z[k][0] for k in z.files}
    grid = data["vision.0.kwargs.grid_thw"]
    cameras = len(grid)
    patches, channels = int(np.prod(grid[0])), by["vision_patch"]["inputs"]["patches"][-1]
    hidden_width = by["language_00_00"]["inputs"]["hidden"][-1]
    sequence, tokens = data["language.0.kwargs.inputs_embeds"].shape[1], patches // 4
    ids = np.flatnonzero(data["language.0.kwargs.visual_pos_masks"].ravel())
    windows = [ids[i * tokens:(i + 1) * tokens] for i in range(cameras)]
    if any(not np.array_equal(w, np.arange(w[0], w[0] + tokens)) for w in windows):
        raise ValueError("noncontiguous image window")
    taps = list(policy.model.backbone.model.model.visual.deepstack_visual_indexes)
    images, deep = [], [[] for _ in taps]
    for camera in range(cameras):
        value = call("vision_patch", f"camera{camera}_patch", patches=alloc(f"patches{camera}", patches * channels))
        for name in sorted(n for n in by if re.fullmatch(r"vision_\d+_\d+", n)):
            value = call(name, f"camera{camera}_{name}", hidden=value)
            stop = int(name.split("_")[-1])
            if stop in taps:
                i = taps.index(stop)
                deep[i].append(call(f"deepstack_merge_{i}", f"camera{camera}_deep{i}", hidden=value))
        images.append(call("vision_merge", f"camera{camera}_tokens", hidden=value))
    embedding = constant("text_embeddings", data["language.0.kwargs.inputs_embeds"])
    value = alloc("language_input", sequence * hidden_width)
    step("pack", value, parts=[part(embedding, buffers[value])] +
         [part(images[i], tokens * hidden_width, windows[i][0] * hidden_width) for i in range(cameras)])
    for name in sorted(n for n in by if re.fullmatch(r"language_\d+_\d+", n)):
        aux = {k: constant(f"{name}_{k}", v) for k, v in fixtures[name].items() if k != "hidden"}
        value = call(name, name, hidden=value, **aux)
        stop = int(name.split("_")[-1])
        if stop < len(taps):
            scattered = alloc(f"deepstack_{stop}", sequence * hidden_width)
            step("pack", scattered, parts=[part(deep[stop][i], tokens * hidden_width, windows[i][0] * hidden_width)
                                          for i in range(cameras)])
            step("euler", value, src=scattered, scalar=-1.0)
    value = call("vlln", "vl_input", value=value)
    for name in sorted(n for n in by if re.fullmatch(r"vl_\d+", n)):
        value = call(name, name, hidden=value)
    backbone = value
    model_horizon = int(data["action_encoder.0.args.0"].shape[1])
    if model_horizon != int(config["action_horizon"]):
        raise ValueError(f"reference noise horizon {model_horizon} != checkpoint {config['action_horizon']}")
    decoded = int(data[f"final_actions.{meta['action_keys'][0]}"].shape[1])
    stats = json.loads((checkpoint / "statistics.json").read_text())[meta["embodiment_tag"]]
    sm, ss, am, ast = [], [], [], []
    for key, group, rep in zip(keys, widths, meta["action_reps"]):
        v = stats["state"][key]
        low, high = np.asarray(v["q01"]), np.asarray(v["q99"])
        sm.extend((low + high) / 2)
        ss.extend((high - low) / 2)
        v, (lk, hk) = (stats["relative_action"][key], ("min", "max")) if rep == "relative" else \
            (stats["action"][key], ("q01", "q99"))
        if rep == "relative" and np.asarray(v[lk]).shape[0] != decoded:
            raise ValueError("relative statistics horizon differs from the decoded horizon")
        low = np.broadcast_to(np.asarray(v[lk]), (decoded, group))
        high = np.broadcast_to(np.asarray(v[hk]), low.shape)
        am.append((low + high) / 2)
        ast.append((high - low) / 2)
    am, ast = np.concatenate(am, axis=1), np.concatenate(ast, axis=1)
    for name, v in (("state_mean", sm), ("state_std", ss), ("action_mean", am), ("action_std", ast)):
        constant(name, v)
    alloc("state", width)
    alloc("state_normalized", width)
    step("normalize", "state_normalized", src="state", mean="state_mean", std="state_std")
    alloc("state_clipped", width)
    step("clip_unit", "state_clipped", src="state_normalized")
    alloc("state_lane", lane)
    step("pack", "state_lane", parts=[part("state_clipped", width)])
    state = call("state_project", "state_features", state="state_lane")
    dit_width = by["action_tail"]["inputs"]["hidden"][-1]
    alloc("noise", model_horizon * action_lane)
    alloc("actions", model_horizon * action_lane)
    step("pack", "actions", parts=[part("noise", model_horizon * action_lane)])
    encoder = SinusoidalPositionalEncoding(fixtures["action_project"]["tau"].shape[-1])
    for iteration in range(steps_count):
        time = torch.from_numpy(data[f"action_encoder.{iteration}.args.1"])[:, None].expand(-1, model_horizon)
        tau = constant(f"tau_{iteration}", encoder(time).numpy())
        temb = constant(f"temb_{iteration}", data[f"dit_block_0.{iteration}.kwargs.temb"])
        action = call("action_project", f"action_features_{iteration}", actions="actions", tau=tau)
        value = alloc(f"hidden_{iteration}", (model_horizon + 1) * dit_width)
        step("pack", value, parts=[part(state, dit_width), part(action, model_horizon * dit_width, dit_width)])
        for name in sorted(n for n in by if re.fullmatch(r"dit_\d+_\d+", n)):
            mask = constant(f"mask_{iteration}_{name}", fixtures[name]["attention_mask"])
            value = call(name, f"{name}_{iteration}", hidden=value, backbone=backbone, attention_mask=mask, temb=temb)
        value = call("action_tail", f"velocity_padded_{iteration}", hidden=value, temb=temb)
        alloc("velocity", model_horizon * action_lane)
        step("gather_strided", "velocity", src=value, stride=padded, take=action_lane, count=model_horizon)
        step("euler", "actions", src="velocity", scalar=-1.0 / steps_count)
    alloc("normalized_actions", decoded * width)
    step("gather_strided", "normalized_actions", src="actions", stride=action_lane, take=width, count=decoded)
    alloc("clipped_actions", decoded * width)
    step("clip_unit", "clipped_actions", src="normalized_actions")
    alloc("action_chunk", decoded * width)
    step("denormalize", "action_chunk", src="clipped_actions", mean="action_mean", std="action_std")
    if relative:
        alloc("relative_reference", decoded * width)
        step("gather_strided", "relative_reference", src="state", stride=0, take=relative, count=decoded,
             dst_stride=width, clear=True)
        step("euler", "action_chunk", src="relative_reference", scalar=-1.)
    for g in graphs:
        g["input_order"] = list(g["inputs"])
    qwen = processor_config(config["model_name"])
    kwargs = json.loads((checkpoint / "processor_config.json").read_text()).get("processor_kwargs") or {}
    instruction = json.loads((out / "reference/reference.json").read_text())["instruction"]
    context = {
        "image_shape": [int(grid[0][1]) * qwen["patch_size"], int(grid[0][2]) * qwen["patch_size"]],
        "image_preprocess": {
            "format": "groot-qwen3-image-v1", "color_space": "RGB",
            "gr00t_eval": {k: kwargs.get(k) for k in ("use_albumentations", "letter_box_transform",
                                                       "shortest_image_edge", "image_target_size",
                                                       "image_crop_size", "crop_fraction")},
            "qwen": qwen},
        "model_version": "1.7", "relative_action_dim": relative, "embodiment_tag": meta["embodiment_tag"],
        "embodiment_id": meta["embodiment_id"],
        "task_catalog": [{"id": 0, "dataset_id": 0, "instruction": instruction}],
        "n17_graphs": graphs,
        "n17_request_tensors": [{"name": f"patches{i}", "shape": [patches, channels]} for i in range(cameras)]
        + [{"name": "state", "shape": [width]}, {"name": "noise", "shape": [model_horizon, action_lane]}],
        "n17_plan": {"buffers": buffers, "steps": steps, "sidecars": sidecars, "result": "action_chunk"},
    }
    # N1.7 camera slots are named by the model's video keys (front, wrist ...).
    profile = {"state_names": meta["state_names"], "state_dim": width, "action_dim": width,
               "camera_keys": [f"observation.images.{k}" for k in meta["video_keys"]], "fps": meta["fps"],
               "actions_per_chunk": decoded, "calibration_id": meta["calibration_id"], "runtime_context": context}
    (root / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    np.savez(root / "normalization_stats.npz", state_mean=sm, state_std=ss, action_mean=am, action_std=ast)
    fixtures_dir = root / "fixtures"
    (fixtures_dir / "inputs").mkdir(parents=True)
    (fixtures_dir / "expected").mkdir()
    for i in range(cameras):
        data["vision.0.args.0"][i * patches:(i + 1) * patches].astype("<f4").tofile(
            fixtures_dir / "inputs" / f"patches{i}.f32")
    np.concatenate([np.asarray(data[f"raw.state.{k}"]).ravel() for k in keys]).astype("<f4").tofile(
        fixtures_dir / "inputs/state.f32")
    data["action_encoder.0.args.0"].astype("<f4").tofile(fixtures_dir / "inputs/noise.f32")
    np.concatenate([data[f"final_actions.{k}"] for k in meta["action_keys"]], axis=-1).astype("<f4").tofile(
        fixtures_dir / "expected/normalized_actions.f32")
    print(f"runtime: {len(steps)} interpreter operations, {len(graphs)} graph contracts", flush=True)
