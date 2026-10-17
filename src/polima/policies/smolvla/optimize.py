"""Export side of the optimized SmolVLA build (`polima compile --policy smolvla`).

Next to the generic export (onnx/{vision,prefix,suffix,denoise}.onnx, calibration, the fixture,
constants), this writes what the device-program bundle runs instead of the per-graph plan:

  device/prefix.onnx                graphs.PrefixCache emitting the expert's cross-attention K/V
                                    already projected (project_cross)
  device/denoise_step.onnx          graphs.DenoiseStep: suffix + expert + Euler, cache as an input
  device/denoise_step.calib.npz     its inputs over real denoising trajectories
  device/constants/tbias_XX         the step's time-MLP constant (graphs.step_time_bias)
  device/connector.onnx             graphs.VisionConnector: pixel shuffle + projection + image scale as one
                                    strided convolution (NCHW, external HWC)
  device/llima_vision/              the SigLIP tower as an HF PaliGemma folder for llima-compile,
                                    SmoothQuant folded in (smooth_vision), identity projector
  device/samples/sNN/               held-out observations: wire inputs and LeRobot's raw actions
  device/manifest.json              what was written, the SmoothQuant alpha, sample count
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from polima.policies.smolvla import graphs
from polima.policies.smolvla.runtime import DENOISE_STEPS

#: SmoothQuant migration strength for the SigLIP tower (searched on SO-101: 0.65 maximized the
#: simulated INT8 output cosine among 0.3/0.5/0.65/0.8).
VISION_ALPHA = 0.65
PALIGEMMA_BASE = "google/paligemma-3b-pt-224"
PALIGEMMA_FILES = ("config.json", "added_tokens.json", "generation_config.json", "preprocessor_config.json",
                   "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "tokenizer.model")
VISION_PREFIX = "model.vlm_with_expert.vlm.model.vision_model."


def _onnx(module: nn.Module, examples: tuple, path: Path, inputs: list[str], outputs: list[str]) -> Path:
    import onnx

    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(module, examples, str(path), input_names=inputs, output_names=outputs,
                          opset_version=17, do_constant_folding=True)
    model = onnx.shape_inference.infer_shapes(onnx.load(str(path)))
    model.ir_version = min(model.ir_version, 8)
    onnx.save(model, str(path))
    return path


# ------------------------------------------------------------------ vision SmoothQuant


def siglip_tokens(tower: graphs.VisionTower, image: torch.Tensor) -> torch.Tensor:
    """The SigLIP tower without its connector: post_layernorm tokens [1, 1024, 768]."""
    vision = tower.vision
    patches = vision.embeddings.patch_embedding(image).flatten(2).transpose(1, 2)
    hidden = vision.encoder(inputs_embeds=patches + vision.embeddings.position_embedding(tower.position_ids),
                            attention_mask=None).last_hidden_state
    return vision.post_layernorm(hidden)


def smooth_vision(tower: graphs.VisionTower, images: list[torch.Tensor], alpha: float = VISION_ALPHA) -> None:
    """SmoothQuant folded into the SigLIP weights, in place (exact in float):
    layer_norm1 (gamma, beta) / s -> q/k/v columns * s; layer_norm2 / s -> fc1 columns * s;
    v_proj rows / s -> out_proj columns * s; s_j = max|x_j|^alpha / max|W_.j|^(1-alpha)."""
    layers = tower.vision.encoder.layers
    ranges: dict = {}

    def hook(key):
        def record(module, args):
            r = args[0].detach().abs().reshape(-1, args[0].shape[-1]).amax(0)
            ranges[key] = torch.maximum(ranges[key], r) if key in ranges else r
        return record

    handles = []
    for i, layer in enumerate(layers):
        handles += [layer.self_attn.q_proj.register_forward_pre_hook(hook((i, "attn"))),
                    layer.mlp.fc1.register_forward_pre_hook(hook((i, "mlp"))),
                    layer.self_attn.out_proj.register_forward_pre_hook(hook((i, "out")))]
    try:
        with torch.inference_mode():
            for image in images:
                siglip_tokens(tower, image)
    finally:
        for handle in handles:
            handle.remove()

    def factor(key, weights):
        wmax = torch.stack([w.abs().amax(0) for w in weights]).amax(0).clamp_min(1e-5)
        return (ranges[key].clamp_min(1e-5) ** alpha / wmax ** (1 - alpha)).clamp(1e-2, 1e3)

    with torch.no_grad():
        for i, layer in enumerate(layers):
            attn = layer.self_attn
            s = factor((i, "attn"), [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight])
            layer.layer_norm1.weight /= s
            layer.layer_norm1.bias /= s
            for linear in (attn.q_proj, attn.k_proj, attn.v_proj):
                linear.weight *= s[None, :]
            s = factor((i, "mlp"), [layer.mlp.fc1.weight])
            layer.layer_norm2.weight /= s
            layer.layer_norm2.bias /= s
            layer.mlp.fc1.weight *= s[None, :]
            s = factor((i, "out"), [attn.out_proj.weight])
            attn.v_proj.weight /= s[:, None]
            attn.v_proj.bias /= s
            attn.out_proj.weight *= s[None, :]


def paligemma_vision_dir(tower: graphs.VisionTower, out: Path) -> Path:
    """The (smoothed) SigLIP tower as an HF PaliGemma folder that llima-compile turns into one
    VISION graph returning post_layernorm's tokens: identity projector, text width = 768. The
    language model is never compiled, but LLiMa locates it by its embedding table."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    vision = tower.vision
    width = vision.config.hidden_size
    tensors = {f"vision_tower.vision_model.{name}": value.detach().float().contiguous()
               for name, value in vision.state_dict().items()}
    tensors["multi_modal_projector.linear.weight"] = torch.eye(width)
    tensors["multi_modal_projector.linear.bias"] = torch.zeros(width)
    files = {name: Path(hf_hub_download(PALIGEMMA_BASE, name)) for name in PALIGEMMA_FILES}
    config = json.loads(files["config.json"].read_text())
    vocab = config["text_config"].get("vocab_size", 257216)
    tensors["language_model.model.embed_tokens.weight"] = torch.zeros(vocab, width, dtype=torch.bfloat16)
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    for name, source in files.items():
        if name != "config.json":
            (out / name).write_bytes(source.read_bytes())
    vc = vision.config
    side = vc.image_size // vc.patch_size
    config["vision_config"].update(
        hidden_size=width, intermediate_size=vc.intermediate_size, num_hidden_layers=vc.num_hidden_layers,
        num_attention_heads=vc.num_attention_heads, image_size=vc.image_size, patch_size=vc.patch_size,
        num_image_tokens=side * side, projection_dim=width, layer_norm_eps=vc.layer_norm_eps,
        hidden_act=vc.hidden_act)
    config["text_config"]["hidden_size"] = width
    config.update(projection_dim=width, hidden_size=width, mm_tokens_per_image=side * side)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return out


# ------------------------------------------------------------------------- export


def export_device_graphs(checkpoint: Path, dataset: Path, out: Path, *, calibration: int = 8,
                         evaluation: int = 16, alpha: float = VISION_ALPHA) -> dict:
    """Write device/ (see module docstring). `evaluation` held-out observations are drawn
    between the calibration ones, each with its own noise; their expected actions are
    LeRobot's own `sample_actions`, raw (denormalized), as the bundle returns them."""
    from polima.export import samples as sampling
    from polima.export.normalization import from_lerobot_checkpoint

    out.mkdir(parents=True, exist_ok=True)
    policy, image_keys = graphs.load_policy(checkpoint)
    # LeRobot's own forward, in the checkpoint's dtypes, is the reference; take it first.
    keys = ["observation.state", *image_keys, "task"]
    # Evenly spaced over the dataset: even indices calibrate, odd ones (in between) evaluate.
    drawn, _, _ = sampling.load(policy, checkpoint, keys, dataset, count=2 * max(calibration, evaluation))
    calibrate, held = drawn[0::2][:calibration], drawn[1::2][:evaluation]
    stats = {k: np.asarray(v, np.float64) for k, v in from_lerobot_checkpoint(checkpoint, ()).items()}
    action_dim = int(policy.config.action_feature.shape[0])
    samples = out / "samples"
    for index, sample in enumerate(held):
        with torch.inference_mode():
            images, masks = policy.prepare_images(sample)
            state = policy.prepare_state(sample)
            noise = torch.from_numpy(np.random.default_rng(2000 + index).standard_normal((1, 50, 32))
                                     .astype(np.float32))
            actions = policy.model.sample_actions(images, masks, sample["observation.language.tokens"],
                                                  sample["observation.language.attention_mask"], state,
                                                  noise=noise.clone())
        d = samples / f"s{index:02d}"
        (d / "inputs").mkdir(parents=True, exist_ok=True)
        for k in range(2):
            np.ascontiguousarray(images[k][0].float().numpy().transpose(1, 2, 0), "<f4").tofile(
                d / "inputs" / f"image{k}.f32")
        raw_state = state[0, :action_dim].double().numpy() * stats["state_std"] + stats["state_mean"]
        raw_state.astype("<f4").tofile(d / "inputs/state.f32")
        noise[0].numpy().astype("<f4").tofile(d / "inputs/noise.f32")
        raw = actions[0, :, :action_dim].double().numpy() * stats["action_std"] + stats["action_mean"]
        raw.astype("<f4").tofile(d / "expected.f32")

    modules = graphs.build_modules(policy)
    model = policy.model
    tbias = torch.stack([graphs.step_time_bias(model, k) for k in range(DENOISE_STEPS)])
    (out / "constants").mkdir(exist_ok=True)
    for k in range(DENOISE_STEPS):
        tbias[k].numpy().astype("<f4").tofile(out / "constants" / f"tbias_{k:02d}")

    rows: dict[str, list] = {"x": [], "tbias": [], "cache": []}
    images, prefix, step, worst = [], None, None, 0.0
    for sample in calibrate:
        traced = graphs.trace(modules, sample, image_keys)  # specializes prefix/denoise to the task
        images += [image.float() for image in traced["prepared_images"]]
        if step is None:  # the device build's prefix emits the expert's cross-attention K/V
            language = int(sample["observation.language.attention_mask"].sum())
            prefix = graphs.PrefixCache(model, language, project_cross=True).eval()
            step = graphs.DenoiseStep(model, graphs.DenoiseCore(model, language, cross_projected=True)).eval()
            prefix_example = traced["prefix"].unsqueeze(1)
        with torch.inference_mode():
            cache = prefix(traced["prefix"].unsqueeze(1)).reshape(1, 1, graphs.CACHE_ROWS, -1)
        x = traced["noise"].reshape(1, 1, 50, 32)
        with torch.inference_mode():
            for k in range(DENOISE_STEPS):
                rows["x"].append(x.numpy()[None])
                rows["tbias"].append(tbias[k].reshape(1, 1, 1, 1, -1).numpy())
                rows["cache"].append(cache.numpy()[None])
                x = step(x, tbias[k].reshape(1, 1, 1, -1), cache)
        reference = traced["normalized_action"].reshape(-1).numpy()
        worst = max(worst, float(np.abs(x.reshape(50, 32)[:, : traced["action_dim"]].reshape(-1).numpy()
                                        - reference).max()))
    if worst > 1e-4:
        raise ValueError(f"DenoiseStep chain differs from the exported graphs by {worst:.2e}")
    examples = tuple(torch.from_numpy(rows[k][0][0]) for k in ("x", "tbias", "cache"))
    _onnx(step, examples, out / "denoise_step.onnx", ["x", "tbias", "cache"], ["x_next"])
    graphs._export_graph(prefix, prefix_example, out / "prefix.onnx", "prefix_embeddings", "cache")
    np.savez(out / "denoise_step.calib.npz", **{k: np.concatenate(v[: 4 * DENOISE_STEPS]) for k, v in rows.items()})

    connector = graphs.VisionConnector(model).eval()
    tower = modules["vision"]
    with torch.inference_mode():
        tokens = siglip_tokens(tower, images[0])
        grid = connector.grid(tokens)
        gap = float((connector.rows(connector(grid)).reshape(-1)
                     - (tower(images[0]) * graphs.IMAGE_SCALE).reshape(-1)).abs().max())
    if gap > 1e-3:
        raise ValueError(f"SigLIP + connector differs from the vision graph by {gap:.2e}")
    _onnx(connector, (grid.contiguous(),), out / "connector.onnx", ["tokens"], ["image_tokens"])

    smooth_vision(tower, images, alpha)
    with torch.inference_mode():
        drift = float((siglip_tokens(tower, images[0]) - tokens).abs().max())
    if drift > 5e-3:
        raise ValueError(f"SmoothQuant changed the float vision output by {drift:.2e}")
    paligemma_vision_dir(tower, out / "llima_vision")

    manifest = {"denoise_step_chain_max_abs": worst, "connector_max_abs": gap, "smoothquant_alpha": alpha,
                "smoothquant_float_max_abs": drift, "calibration_observations": len(calibrate),
                "evaluation_samples": len(held), "tbias": [f"tbias_{k:02d}" for k in range(DENOISE_STEPS)]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


__all__ = ["VISION_ALPHA", "export_device_graphs", "paligemma_vision_dir", "siglip_tokens", "smooth_vision"]
