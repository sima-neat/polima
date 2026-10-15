"""ACT's torch side: the module decomposition that becomes six ONNX graphs.

This module needs torch and lerobot, so it is imported only by the export stage
running in PoLiMa's self-contained CPU venv. Everything else in
`polima.policies.act` stays numpy-only; `GraphSpec.builder` names the classes
here as strings precisely so the spec can be read in the compiler venv and on
the board, where torch does not exist.

## Why the graph boundaries are where they are

ACT is one network, but the MLA wants fixed-shape rank-4 tensors and no
host-side control flow, so it is cut into six pieces that the board's plan.json
chains together:

    vision_backbone         one camera image  -> 300 tokens      (run twice)
    encoder_layer_00_stem   host-packed 601 tokens -> hidden
    encoder_layer_01/02/03  hidden -> hidden
    decoder_action_tail     hidden -> 100x16 padded actions

Two details are load bearing:

* The **positional embeddings are frozen into the graphs** as buffers. They are
  constant for a fixed camera count and resolution, so computing them on the
  board would be pure cost, and passing them as inputs would add two large
  tensors to every MLA call.
* The **action head is padded from its real width to 16 outputs**. The MLA's last
  dimension is tiled; the profile's real action lanes (6 for the SO-101) are
  sliced back out by the profile-aware runtime, so `decoder_action_tail` keeps
  the same (1, 1, 100, 16) hardware boundary for every robot profile.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from polima.policies.act.runtime import (
    CAMERA_TOKENS,
    CHUNK,
    ENCODER_TOKENS,
    HIDDEN,
    PADDED_ACTION_DIM,
    STEM_TOKENS,
)

HEIGHT, WIDTH = 480, 640
SUPPORTED_DIMS = (6, 7)
#: Encoder tokens *including* the latent token that the stem prepends. The stem
#: takes 601 (state + 2x300) and emits 602.
TOTAL_TOKENS = ENCODER_TOKENS


# ------------------------------------------------------------- checkpoint load


def load_policy(checkpoint: str | Path, lerobot_src: str | Path | None = None):
    """Load an ACT checkpoint and refuse anything the compiled runtime cannot serve.

    The checks are not defensive noise. Each one corresponds to a shape or a
    control-flow assumption baked into the six graphs or into plan.json, and
    without them a mismatched checkpoint exports cleanly, compiles cleanly, and
    produces wrong actions on the arm.
    """
    import sys

    if lerobot_src and str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))
    from lerobot.policies.act import ACTPolicy
    from lerobot.policies.act.configuration_act import ACTConfig

    config = ACTConfig.from_pretrained(str(checkpoint), local_files_only=True)
    config.device = "cpu"
    policy = ACTPolicy.from_pretrained(
        str(checkpoint), config=config, local_files_only=True
    ).cpu().eval()
    config = policy.config
    image_keys = list(config.image_features)

    problems: list[str] = []
    if config.type != "act":
        problems.append(f"type={config.type}, expected act")
    if config.chunk_size != CHUNK or config.n_action_steps != CHUNK:
        problems.append(
            f"chunk_size/n_action_steps are {config.chunk_size}/{config.n_action_steps}, "
            f"but the runtime's wire response is fixed at {CHUNK}"
        )
    state_shape = config.robot_state_feature.shape if config.robot_state_feature else None
    action_shape = config.action_feature.shape
    if state_shape not in {(dim,) for dim in SUPPORTED_DIMS}:
        problems.append(f"state shape must have one of {SUPPORTED_DIMS} dimensions")
    if action_shape not in {(dim,) for dim in SUPPORTED_DIMS}:
        problems.append(f"action shape must have one of {SUPPORTED_DIMS} dimensions")
    if state_shape != action_shape:
        problems.append(f"state/action shapes must match, got {state_shape}/{action_shape}")
    if len(image_keys) != 2:
        problems.append(f"expected exactly 2 cameras, got {image_keys}")
    if any(config.input_features[key].shape != (3, HEIGHT, WIDTH) for key in image_keys):
        problems.append(f"camera shapes must be (3, {HEIGHT}, {WIDTH})")
    if config.dim_model != HIDDEN or config.n_encoder_layers != 4 or config.n_decoder_layers != 1:
        problems.append(
            f"architecture is {config.dim_model}/{config.n_encoder_layers}/"
            f"{config.n_decoder_layers}, but the graph split assumes {HIDDEN}/4/1"
        )
    if config.temporal_ensemble_coeff is not None:
        # Temporal ensembling keeps state across inferences; the board's runtime
        # is stateless by construction, so the two disagree about what an
        # inference means.
        problems.append("temporal ensembling is not supported by the compiled runtime")
    if problems:
        raise ValueError(
            f"unsupported ACT checkpoint {checkpoint}:\n  - " + "\n  - ".join(problems)
        )
    return policy, image_keys


def fixed_position(model: nn.Module, cameras: int = 2) -> torch.Tensor:
    """The encoder positional embedding, evaluated once and frozen.

    Shape (1, 602, 512): one 1-D embedding row per non-image token, then the
    camera embedding repeated per camera. It depends only on the feature-map
    geometry (480/32 x 640/32 = 15 x 20 = 300 tokens), not on the input, which
    is what makes freezing it correct.
    """
    one_d = model.encoder_1d_feature_pos_embed.weight.detach().unsqueeze(0)
    feature = torch.zeros(1, HIDDEN, HEIGHT // 32, WIDTH // 32)
    camera = model.encoder_cam_feat_pos_embed(feature).flatten(2).transpose(1, 2).detach()
    result = torch.cat([one_d, *([camera] * cameras)], dim=1)
    if tuple(result.shape) != (1, TOTAL_TOKENS, HIDDEN):
        raise ValueError(
            f"positional embedding is {tuple(result.shape)}, expected "
            f"(1, {TOTAL_TOKENS}, {HIDDEN}) -- camera count or resolution changed"
        )
    return result


# ------------------------------------------------------------------- modules


class VisionBackbone(nn.Module):
    """One camera image -> 300 projected tokens. Run once per camera."""

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.project = model.encoder_img_feat_input_proj

    def forward(self, image):
        features = self.project(self.backbone(image)["feature_map"])
        return features.flatten(2).transpose(1, 2)


class EncoderStemLayer(nn.Module):
    """Latent + state + both camera token blocks -> first encoder layer output.

    The latent input is a zero vector: ACT's VAE encoder is training-only, and at
    inference the latent is fixed at zero. Materializing it here rather than
    passing it in keeps it out of the wire protocol.
    """

    def __init__(self, model, position):
        super().__init__()
        self.latent = model.encoder_latent_input_proj
        self.state = model.encoder_robot_state_input_proj
        self.layer = model.encoder.layers[0]
        self.latent_dim = model.config.latent_dim
        self.register_buffer("position", position)

    def forward(self, state, camera0, camera1):
        batch = state.shape[0]
        latent = self.latent(
            torch.zeros(batch, self.latent_dim, dtype=state.dtype, device=state.device)
        )
        # Keep this as its own statement. Inlining it into the cat below is
        # numerically identical but emits the state Gemm *after* the latent's
        # Unsqueeze instead of before it, because torch.onnx traces in evaluation
        # order. The graph is equivalent but the ELF is not byte-identical, so a
        # rebuild would no longer reproduce the same ELF.
        state_token = self.state(state)
        hidden = torch.cat(
            [latent.unsqueeze(1), state_token.unsqueeze(1), camera0, camera1], dim=1
        )
        output = self.layer(hidden.transpose(0, 1), pos_embed=self.position.transpose(0, 1))
        return output.transpose(0, 1)


class EncoderLayer(nn.Module):
    def __init__(self, layer, position):
        super().__init__()
        self.layer = layer
        self.register_buffer("position", position)

    def forward(self, hidden):
        return self.layer(
            hidden.transpose(0, 1), pos_embed=self.position.transpose(0, 1)
        ).transpose(0, 1)


class DecoderActionTail(nn.Module):
    def __init__(self, model, position):
        super().__init__()
        self.decoder = model.decoder
        self.action_head = model.action_head
        self.decoder_position = model.decoder_pos_embed
        self.chunk = model.config.chunk_size
        self.hidden = model.config.dim_model
        self.register_buffer("encoder_position", position)

    def forward(self, encoder_hidden):
        memory = encoder_hidden.transpose(0, 1)
        query = torch.zeros(
            self.chunk, encoder_hidden.shape[0], self.hidden,
            dtype=encoder_hidden.dtype, device=encoder_hidden.device,
        )
        decoded = self.decoder(
            query, memory,
            encoder_pos_embed=self.encoder_position.transpose(0, 1),
            decoder_pos_embed=self.decoder_position.weight.unsqueeze(1),
        )
        return self.action_head(decoded.transpose(0, 1))


# ------------------------------------------------- rank-4 exported wrappers
#
# The MLA takes rank-4 NHWC tensors. These wrappers are the only difference
# between the torch modules above (which are readable) and what is exported
# (which is compilable): they reshape a rank-4 input down to the rank-3 the
# network wants and unsqueeze the output back.


class EncoderStemPacked(nn.Module):
    """Takes the host-packed (1, 1, 601, 512) buffer.

    Token 0 carries the profile's state values; tokens 1..300 and
    301..600 are the two cameras. The board packs exactly this layout, so the
    slicing here and `RuntimePlan`'s pack offsets must agree.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.state_dim = module.state.in_features

    def forward(self, stem_input):
        packed = stem_input.reshape(1, STEM_TOKENS, HIDDEN)
        return self.module(
            packed[:, 0, :self.state_dim],
            packed[:, 1:1 + CAMERA_TOKENS],
            packed[:, 1 + CAMERA_TOKENS:1 + 2 * CAMERA_TOKENS],
        ).unsqueeze(1)


class EncoderLayerRank4(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, hidden):
        return self.module(hidden.reshape(1, ENCODER_TOKENS, HIDDEN)).unsqueeze(1)


class DecoderActionRank4(nn.Module):
    """Decoder tail with the profile's action head padded to 16 outputs.

    The padding is a hardware fit, not a model change: rows beyond the real
    action dimension are zeroed weights and zeroed bias, so they contribute
    nothing, and the board slices the real columns back out. Copying the
    trained weights into the top rows leaves the actual outputs bit-identical.
    """

    def __init__(self, module):
        super().__init__()
        self.decoder = module.decoder
        self.decoder_position = module.decoder_position
        self.chunk = module.chunk
        self.hidden = module.hidden
        self.register_buffer("encoder_position", module.encoder_position)
        self.action_dim = module.action_head.out_features
        self.action_head = nn.Linear(module.action_head.in_features, PADDED_ACTION_DIM)
        with torch.no_grad():
            self.action_head.weight.zero_()
            self.action_head.bias.zero_()
            self.action_head.weight[:self.action_dim].copy_(module.action_head.weight)
            self.action_head.bias[:self.action_dim].copy_(module.action_head.bias)

    def forward(self, hidden):
        encoder_hidden = hidden.reshape(1, ENCODER_TOKENS, HIDDEN)
        memory = encoder_hidden.transpose(0, 1)
        query = torch.zeros(
            self.chunk, 1, self.hidden, dtype=hidden.dtype, device=hidden.device
        )
        decoded = self.decoder(
            query, memory,
            encoder_pos_embed=self.encoder_position.transpose(0, 1),
            decoder_pos_embed=self.decoder_position.weight.unsqueeze(1),
        )
        return self.action_head(decoded.transpose(0, 1)).unsqueeze(1)


def build_modules(policy) -> dict[str, nn.Module]:
    """The six graphs, keyed by `GraphSpec.name`."""
    model = policy.model
    position = fixed_position(model)
    return {
        "vision_backbone": VisionBackbone(model),
        "encoder_layer_00_stem": EncoderStemLayer(model, position),
        "encoder_layer_01": EncoderLayer(model.encoder.layers[1], position),
        "encoder_layer_02": EncoderLayer(model.encoder.layers[2], position),
        "encoder_layer_03": EncoderLayer(model.encoder.layers[3], position),
        "decoder_action_tail": DecoderActionTail(model, position),
    }


# --------------------------------------------------------------------- tracing


def pack_stem_input(state: torch.Tensor, camera_tokens: Sequence[torch.Tensor]) -> torch.Tensor:
    """Build the (1, 1, 601, 512) stem buffer exactly as the board does."""
    packed = torch.zeros(1, 1, STEM_TOKENS, HIDDEN, dtype=torch.float32)
    state_dim = state.shape[-1]
    if state_dim not in SUPPORTED_DIMS:
        raise ValueError(f"ACT state has {state_dim} values; supported dimensions are {SUPPORTED_DIMS}")
    packed[:, :, 0, :state_dim] = state
    packed[:, :, 1:1 + CAMERA_TOKENS] = camera_tokens[0].unsqueeze(1)
    packed[:, :, 1 + CAMERA_TOKENS:1 + 2 * CAMERA_TOKENS] = camera_tokens[1].unsqueeze(1)
    return packed


def trace(modules: dict[str, nn.Module], sample: dict, image_keys: Sequence[str]) -> dict:
    """Run the chain in torch, keeping every intermediate.

    The intermediates are the calibration data: each graph is calibrated on the
    activations its predecessor actually produces, not on random noise, which is
    what keeps the quantized chain faithful end to end.
    """
    with torch.inference_mode():
        cameras = [modules["vision_backbone"](sample[key]) for key in image_keys]
        hidden0 = modules["encoder_layer_00_stem"](
            sample["observation.state"], cameras[0], cameras[1]
        )
        hidden1 = modules["encoder_layer_01"](hidden0)
        hidden2 = modules["encoder_layer_02"](hidden1)
        hidden3 = modules["encoder_layer_03"](hidden2)
        action = modules["decoder_action_tail"](hidden3)
    return {
        "cameras": cameras, "hidden0": hidden0, "hidden1": hidden1,
        "hidden2": hidden2, "hidden3": hidden3, "action": action,
    }


# ---------------------------------------------------------------- onnx export

# The pinned Torch 2.3 exporter is already legacy-only and has no ``dynamo``
# keyword. Newer Torch versions default to the dynamo exporter, so explicitly
# disable it when the installed signature supports that request.
_EXPORT_TAKES_DYNAMO = "dynamo" in inspect.signature(torch.onnx.export).parameters


def export_graph(module: nn.Module, inputs: tuple, path: Path,
                 input_names: Sequence[str], output_name: str) -> Path:
    import onnx

    module.cpu().eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {"dynamo": False} if _EXPORT_TAKES_DYNAMO else {}
    with torch.inference_mode():
        torch.onnx.export(
            module, inputs, str(path),
            input_names=list(input_names), output_names=[output_name],
            # Opset 17 and the legacy tracer: AFE's importer is validated
            # against this combination, while dynamo emits a different valid
            # graph that AFE does not support.
            opset_version=17, do_constant_folding=True, **extra,
        )
    onnx.checker.check_model(str(path))
    return path


def export_all(output: Path, modules: dict[str, nn.Module], samples: list[dict],
               traces: list[dict], image_keys: Sequence[str]) -> list[Path]:
    """Write `onnx/<name>.onnx` and `calibration/<name>.npz` for all six graphs."""
    onnx_dir = output / "onnx"
    calibration = output / "calibration"
    calibration.mkdir(parents=True, exist_ok=True)
    first, first_trace = samples[0], traces[0]
    written: list[Path] = []

    written.append(export_graph(
        modules["vision_backbone"], (first[image_keys[0]],),
        onnx_dir / "vision_backbone.onnx", ("image",), "camera_tokens"))
    written.append(export_graph(
        EncoderStemPacked(modules["encoder_layer_00_stem"]),
        (pack_stem_input(first["observation.state"], first_trace["cameras"]),),
        onnx_dir / "encoder_layer_00_stem.onnx", ("stem_input",), "hidden"))
    for index in range(1, 4):
        written.append(export_graph(
            EncoderLayerRank4(modules[f"encoder_layer_{index:02d}"]),
            (first_trace[f"hidden{index - 1}"].unsqueeze(1),),
            onnx_dir / f"encoder_layer_{index:02d}.onnx", ("hidden",), "hidden_out"))
    written.append(export_graph(
        DecoderActionRank4(modules["decoder_action_tail"]),
        (first_trace["hidden3"].unsqueeze(1),),
        onnx_dir / "decoder_action_tail.onnx", ("hidden",), "normalized_actions"))

    # The leading axis of each array is the calibration-sample axis; everything
    # after it is the graph's own fixed shape.
    np.savez(calibration / "vision_backbone.npz",
             image=np.stack([s[key] for s in samples for key in image_keys]))
    np.savez(calibration / "encoder_layer_00_stem.npz",
             stem_input=np.stack([
                 pack_stem_input(s["observation.state"], t["cameras"]).numpy()
                 for s, t in zip(samples, traces, strict=True)]))
    for index in range(1, 4):
        np.savez(calibration / f"encoder_layer_{index:02d}.npz",
                 hidden=np.stack([t[f"hidden{index - 1}"].unsqueeze(1).numpy() for t in traces]))
    np.savez(calibration / "decoder_action_tail.npz",
             hidden=np.stack([t["hidden3"].unsqueeze(1).numpy() for t in traces]))
    return written


# ------------------------------------------------------------------ fixtures


FIXTURE_FILE = "act_fixture.npz"


def write_fixtures(output: Path, sample: dict, trace_value: dict,
                   image_keys: Sequence[str], postprocessor) -> None:
    """The npz the host smoke test uses, and the raw `.f32` the board reads.

    `expected_normalized_actions.f32` is the reference `polima run --fixture`
    compares against. It is written from **PyTorch**, not from onnxruntime --
    a distinction that matters, because the per-graph `<graph>_output.f32` files
    written by the verify step come from onnxruntime instead, and the two differ
    by ~1.4e-06. Mixing them up makes a passing bundle look broken.
    """
    raw_action = postprocessor(trace_value["action"]).detach().cpu().numpy()
    np.savez(
        output / FIXTURE_FILE,
        state=sample["observation.state"].numpy(),
        camera0=sample[image_keys[0]].numpy(),
        camera1=sample[image_keys[1]].numpy(),
        normalized_action=trace_value["action"].numpy(),
        action=raw_action,
    )
    direct = output / "direct_inputs"
    direct.mkdir(parents=True, exist_ok=True)
    for index, key in enumerate(image_keys):
        # NHWC on disk: the board hands this buffer to the MLA unchanged.
        sample[key].numpy().transpose(0, 2, 3, 1).astype("<f4").tofile(
            direct / f"vision_input_{index}.f32")
    sample["observation.state"].numpy().astype("<f4").tofile(direct / "state.f32")
    trace_value["action"].numpy().astype("<f4").tofile(
        direct / "expected_normalized_actions.f32")


# -------------------------------------------------------------------- verify


def verify_chain(onnx_dir: Path, fixture_path: Path, report_path: Path,
                 atol: float = 1e-4, rtol: float = 1e-3,
                 stage_dir: Path | None = None) -> dict:
    """Replay the six graphs under onnxruntime and compare against PyTorch.

    This runs before anything is compiled, so a mismatch here is an export bug
    rather than a quantization one. Separating the two is the entire value: once
    the ONNX chain matches PyTorch, any later drift is attributable to the MLA.

    `stage_dir` also dumps every graph's input and output. Those are what turn
    "the actions are wrong" into "graph 3 is wrong": the board can replay one ELF
    against its recorded input and compare. Every value here is already computed,
    so writing them costs nothing.

    Note these are **onnxruntime** values, unlike `expected_normalized_actions.f32`
    which is PyTorch -- the two differ by ~1.4e-06.
    """
    import onnxruntime as ort

    def session(path: Path):
        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def dump(name: str, array: np.ndarray) -> None:
        if stage_dir is not None:
            stage_dir.mkdir(parents=True, exist_ok=True)
            np.ascontiguousarray(array, dtype="<f4").tofile(stage_dir / f"{name}.f32")

    fixture = np.load(fixture_path)
    vision = session(onnx_dir / "vision_backbone.onnx")
    cameras = [
        vision.run(None, {"image": fixture[f"camera{i}"].astype(np.float32)})[0]
        for i in range(2)
    ]
    for index, tokens in enumerate(cameras):
        dump(f"vision_output_{index}", tokens)

    packed = np.zeros((1, 1, STEM_TOKENS, HIDDEN), np.float32)
    state_dim = fixture["state"].shape[-1]
    action_dim = fixture["normalized_action"].shape[-1]
    if state_dim not in SUPPORTED_DIMS or action_dim not in SUPPORTED_DIMS:
        raise ValueError(
            f"ACT fixture state/action dimensions are {state_dim}/{action_dim}; "
            f"supported dimensions are {SUPPORTED_DIMS}"
        )
    packed[:, :, 0, :state_dim] = fixture["state"]
    packed[:, :, 1:1 + CAMERA_TOKENS] = cameras[0]
    packed[:, :, 1 + CAMERA_TOKENS:1 + 2 * CAMERA_TOKENS] = cameras[1]

    dump("encoder_layer_00_stem_input", packed)
    hidden = session(onnx_dir / "encoder_layer_00_stem.onnx").run(
        None, {"stem_input": packed})[0]
    dump("encoder_layer_00_stem_output", hidden)
    for index in range(1, 4):
        dump(f"encoder_layer_{index:02d}_input", hidden)
        hidden = session(onnx_dir / f"encoder_layer_{index:02d}.onnx").run(
            None, {"hidden": hidden})[0]
        dump(f"encoder_layer_{index:02d}_output", hidden)

    dump("decoder_action_tail_input", hidden)
    padded = session(onnx_dir / "decoder_action_tail.onnx").run(None, {"hidden": hidden})[0]
    dump("decoder_action_tail_output", padded)
    action = padded[:, 0, :, :action_dim]

    reference = fixture["normalized_action"].astype(np.float32)
    difference = np.abs(action - reference)
    report = {
        "ok": bool(np.allclose(action, reference, atol=atol, rtol=rtol)),
        "shape": list(action.shape),
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "atol": atol,
        "rtol": rtol,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
