"""Pi0.5 geometry -> PolicySpec. Stdlib only: the board and the compiler venv import it."""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polima.policies.base import (
    CalibrationSource,
    CompilePlan,
    DatasetContract,
    GraphSpec,
    PolicySpec,
    RobotSpec,
    SmokeSpec,
    TensorSpec,
    WireSpec,
)
from polima.policies.pi05 import runtime as rt

SO101_JOINTS = ("shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                "wrist_flex.pos", "wrist_roll.pos", "gripper.pos")
BASE_CAMERAS = ("observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb",
                "observation.images.right_wrist_0_rgb")


@dataclass(frozen=True)
class Geometry:
    """What varies between Pi0.5 checkpoints and changes the compiled graphs or wire."""

    cameras: tuple[str, ...]           # LeRobot feature keys, in the checkpoint's order
    state_dim: int
    action_dim: int
    camera_shape: tuple[int, int, int] = (224, 224, 3)  # HWC of the training images
    joint_names: tuple[str, ...] = ()
    calibration_id: str = ""
    task: str = ""
    relative_actions: bool = False
    roles: tuple[tuple[str, str], ...] = field(default=())
    #: "llima": the PaliGemma half compiled by LLiMa and run device-resident
    #: (runtime.build_llima_plan); "onnx": PoLiMa's own two-layer ONNX partitions.
    backend: str = "llima"

    @classmethod
    def base(cls) -> Geometry:
        return cls(cameras=BASE_CAMERAS, state_dim=32, action_dim=32)

    @property
    def prefix_tokens(self) -> int:
        if self.backend == "llima":
            return rt.llima_prefix(len(self.cameras))
        return rt.prefix_tokens(len(self.cameras))

    @property
    def language_rows(self) -> int:
        """Prompt rows the client sends: the prefix minus the image tokens."""
        return self.prefix_tokens - rt.IMAGE_TOKENS * len(self.cameras)

    def to_context(self) -> dict[str, Any]:
        return {"cameras": len(self.cameras), "camera_keys": list(self.cameras), "backend": self.backend,
                "prefix_tokens": self.prefix_tokens, "language_tokens": rt.LANGUAGE_TOKENS,
                "language_rows": self.language_rows,
                "image_size": rt.IMAGE_SIZE, "relative_actions": self.relative_actions,
                "task": self.task, "embed_scale": 2048 ** 0.5, "mask": -1.0e4}


def geometry_from_config(checkpoint: str | Path, backend: str = "llima") -> Geometry:
    """Read camera keys and dims from a LeRobot pi05 checkpoint's config.json."""
    config = json.loads((Path(checkpoint) / "config.json").read_text())
    if config.get("type") != "pi05":
        raise ValueError(f"{checkpoint}: config type {config.get('type')!r} is not pi05")
    if int(config.get("chunk_size", rt.CHUNK)) != rt.CHUNK or int(config.get("num_inference_steps", 10)) != 10:
        raise ValueError("PoLiMa's Pi0.5 graphs assume chunk_size 50 and 10 denoising steps")
    if config.get("use_visual_memory") or config.get("use_proprioceptive_memory"):
        raise ValueError("Pi0.5 checkpoints with visual/proprioceptive memory are not supported")
    if list(config.get("image_resolution", [224, 224])) != [224, 224]:
        raise ValueError("PoLiMa's Pi0.5 graphs assume 224x224 images")
    if int(config.get("tokenizer_max_length", 200)) != rt.LANGUAGE_TOKENS:
        raise ValueError("PoLiMa's Pi0.5 graphs assume a 200-token prompt")
    features = config.get("input_features") or {}
    cameras = tuple(k for k, v in features.items() if v.get("type") == "VISUAL")
    state = int((features.get("observation.state") or {}).get("shape", [32])[0])
    action = int(((config.get("output_features") or {}).get("action") or {}).get("shape", [32])[0])
    if not cameras or state > rt.ACTION_LANE or action > rt.ACTION_LANE:
        raise ValueError(f"{checkpoint}: unsupported Pi0.5 geometry ({len(cameras)} cameras, "
                         f"state {state}, action {action}; at most 32 each)")
    shape = features[cameras[0]].get("shape", [3, 224, 224])
    so101 = state == action == 6
    return Geometry(cameras=cameras, state_dim=state, action_dim=action,
                    camera_shape=(int(shape[1]), int(shape[2]), int(shape[0])),
                    joint_names=SO101_JOINTS if so101 else (),
                    calibration_id="so-arm101" if so101 else "",
                    relative_actions=bool(config.get("use_relative_actions", False)),
                    roles=so101_roles(cameras) if so101 and len(cameras) == 2 else (),
                    backend=backend)


def so101_roles(cameras: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """PoLiMa's SO-101 camera roles for a two-camera checkpoint, in checkpoint order:
    the camera whose key names a wrist or gripper is the wrist one, the other overhead."""
    wrist = [any(w in key.rsplit(".", 1)[-1].lower() for w in ("wrist", "gripper", "hand")) for key in cameras]
    if sum(wrist) != 1:
        return ()
    return tuple(("wrist", "Wrist") if is_wrist else ("overhead", "Overhead") for is_wrist in wrist)


def _llima_graphs(g: Geometry, expert_form: str = "nhwc") -> tuple[GraphSpec, ...]:
    """LLiMa's ELFs (compiled by `llima-compile`, not from these specs) and the expert."""
    p, image = g.prefix_tokens, rt.IMAGE_SIZE
    llima = {"layout": "NHWC", "mla_tessellation": False, "external_dram_layout": "HWC",
             "calibration": CalibrationSource("random")}
    hidden = TensorSpec("hidden", (1, 1, p, rt.WIDTH))
    graphs = [GraphSpec(name="vision", builder="llima:vision",
                        inputs=(TensorSpec("image", (1, image, image, 3)),),
                        outputs=(TensorSpec("tokens", (1, 1, rt.IMAGE_TOKENS, rt.WIDTH)),), **llima)]
    for layer in range(rt.PREFIX_LAYERS):
        graphs.append(GraphSpec(
            name=rt.llima_pre(layer), builder="llima:pre",
            inputs=(hidden, TensorSpec("freq_real", (1, 1, p, rt.FREQS)), TensorSpec("freq_imag", (1, 1, p, rt.FREQS))),
            outputs=(TensorSpec("q", (1, 8, p, rt.ROW)), TensorSpec("k", (1, 1, p, rt.ROW)),
                     TensorSpec("v", (1, 1, p, rt.ROW))), **llima))
    graphs.append(GraphSpec(
        name="attn", builder="llima:cache",
        inputs=(TensorSpec("q", (1, 8, p, rt.ROW)), TensorSpec("k", (1, 1, p, rt.ROW)),
                TensorSpec("mask", (1, 1, p, p)), TensorSpec("v", (1, 1, p, rt.ROW))),
        outputs=(TensorSpec("attn", (1, 1, p, rt.WIDTH)),), **llima))
    for layer in range(rt.PREFIX_LAYERS - 1):
        graphs.append(GraphSpec(name=rt.llima_post(layer), builder="llima:post",
                                inputs=(hidden, TensorSpec("attn", (1, 1, p, rt.WIDTH))),
                                outputs=(hidden,), **llima))
    graphs.append(_expert_graph(p, expert_form))
    return tuple(graphs)


def _expert_graph(p: int, form: str = "nhwc") -> GraphSpec:
    if form == "nchw":  # GR00T N1.7's DiT form (blocks.ExpertStepNchw)
        return GraphSpec(
            name="expert", builder="polima.policies.pi05.blocks:ExpertStepNchw",
            inputs=(TensorSpec("x", (1, rt.ACTION_LANE, 1, rt.CHUNK)), TensorSpec("ada", (1, rt.ADA, 1, 1)),
                    TensorSpec("cache", (1, rt.ROW, 2 * rt.EXPERT_LAYERS, p)),
                    TensorSpec("rope", (1, 2 * rt.ROW, 1, rt.CHUNK)), TensorSpec("mask", (1, p + rt.CHUNK, 1, 1))),
            outputs=(TensorSpec("velocity", (1, rt.ACTION_LANE, 1, rt.CHUNK)),),
            layout="NCHW", mla_tessellation=False, external_dram_layout="HWC",
            calibration=CalibrationSource("npz"))
    return GraphSpec(
        name="expert", builder="polima.policies.pi05.graphs:expert_step",
        inputs=(TensorSpec("cache", (1, 1, rt.cache_rows(p), rt.ROW)),
                TensorSpec("x", (1, 1, rt.CHUNK, rt.ACTION_LANE)),
                TensorSpec("cond", (1, 1, 1, rt.COND)),
                TensorSpec("rope", (1, 1, 2 * rt.CHUNK, rt.ROW)),
                TensorSpec("bias", (1, 1, 1, p + rt.CHUNK))),
        outputs=(TensorSpec("velocity", (1, 1, rt.CHUNK, rt.ACTION_LANE)),),
        layout="NHWC", mla_tessellation=False, external_dram_layout="HWC", exit_on_stable_elf=True,
        calibration=CalibrationSource("npz"))


def _graphs(g: Geometry, expert_form: str = "nhwc") -> tuple[GraphSpec, ...]:
    if g.backend == "llima":
        return _llima_graphs(g, expert_form)
    p = g.prefix_tokens
    common = {"layout": "NHWC", "mla_tessellation": False, "external_dram_layout": "HWC",
              "exit_on_stable_elf": True, "calibration": CalibrationSource("npz")}
    graphs = [GraphSpec(
        name="vision", builder="polima.policies.pi05.graphs:VisionGraph",
        inputs=(TensorSpec("patches", (1, 1, rt.IMAGE_TOKENS, rt.PATCH_VALUES)),),
        outputs=(TensorSpec("tokens", (1, 1, rt.IMAGE_TOKENS, rt.WIDTH)),), **common)]
    for index, name in enumerate(rt.block_names()):
        graphs.append(GraphSpec(
            name=name, builder="polima.policies.pi05.graphs:prefix_block",
            inputs=(TensorSpec("hidden", (1, 1, p, rt.WIDTH)), TensorSpec("bias", (1, 1, 1, p))),
            outputs=(TensorSpec("output", (1, 1, rt.block_rows(p), rt.ROW)),), **common))
    graphs.append(_expert_graph(p))
    return tuple(graphs)


def build_spec(g: Geometry, runtime_context: Mapping[str, Any] | None = None) -> PolicySpec:
    cameras = len(g.cameras)
    p = g.prefix_tokens
    joints = g.joint_names or tuple(f"joint_{i:02d}" for i in range(g.state_dim))
    roles = g.roles or tuple((key.rsplit(".", 1)[-1], key.rsplit(".", 1)[-1].replace("_", " ").title())
                             for key in g.cameras)
    if g.backend == "llima":  # LLiMa's vision ELF reads the [-1, 1] image itself, HWC
        request = [TensorSpec(f"image{k}", (rt.IMAGE_SIZE, rt.IMAGE_SIZE, 3)) for k in range(cameras)]
    else:
        request = [TensorSpec(f"image{k}", (rt.IMAGE_TOKENS, rt.PATCH_VALUES)) for k in range(cameras)]
    request += [TensorSpec("language", (g.language_rows, rt.WIDTH)),
                TensorSpec("prefix_bias", (p,)),
                TensorSpec("suffix_rope", (2 * rt.CHUNK, rt.ROW)),
                TensorSpec("noise", (rt.CHUNK, rt.ACTION_LANE))]
    context = {**g.to_context(), **dict(runtime_context or {})}
    if g.task:
        context["task_catalog"] = [{"id": 0, "instruction": g.task}]
    # With SO-101 roles, the contract names cameras by role (the robot client maps
    # roles to devices); the checkpoint's own keys stay in runtime_context.
    camera_keys = tuple(f"observation.images.{role}" for role, _ in g.roles) if g.roles else g.cameras
    return PolicySpec(
        name="pi05",
        display_name="Pi0.5 (PaliGemma + flow-matching action expert)",
        dataset=DatasetContract(
            state_names=joints, state_dim=g.state_dim, action_dim=g.action_dim, camera_keys=camera_keys,
            camera_shape=g.camera_shape, fps=30, single_task=False,
            codebase_version=("v2.0", "v2.1", "v3.0")),
        compile=CompilePlan(
            graphs=_graphs(g, str((runtime_context or {}).get("expert_form", "nhwc"))),
            export_entry="polima.policies.pi05.graphs:export_all",
            fixture_file="pi05_fixture.npz",
            verify_atol=1e-3, verify_rtol=1e-2),
        wire=WireSpec(
            magic=rt.WIRE_MAGIC, version=1, default_port=rt.DEFAULT_PORT,
            request_tensors=tuple(request),
            response_shape=(rt.CHUNK, g.action_dim),
            normalization_side="client"),
        robot=RobotSpec(
            camera_roles=roles, joint_names=joints, actions_per_chunk=rt.CHUNK, default_fps=30,
            calibration_id=g.calibration_id, task_string=g.task or None,
            supports_calibrate=g.calibration_id == "so-arm101"),
        runtime_plan_builder="polima.policies.pi05.runtime:build_plan",
        profile_entry="polima.policies.pi05.profile:specialize_spec",
        runtime_context=context,
        # The PyTorch reference is float32; the bar per precision is set at bundle time.
        smoke=SmokeSpec(cosine_min=0.999, mean_abs_max=0.01),
    )


def specialize_spec(spec: PolicySpec, *, checkpoint: str | Path | None = None,
                    profile: Mapping[str, Any] | None = None, backend: str = "llima", **_: Any) -> PolicySpec:
    """The spec for one checkpoint (from its config.json) or a saved build profile."""
    if profile:
        saved = dict(profile)
        context = dict(saved.get("runtime_context") or {})
        g = Geometry(cameras=tuple(context.get("camera_keys") or saved["camera_keys"]),
                     state_dim=int(saved["state_dim"]),
                     action_dim=int(saved["action_dim"]),
                     camera_shape=tuple(saved.get("camera_shape", (224, 224, 3))),
                     joint_names=tuple(saved.get("state_names") or ()),
                     calibration_id=str(saved.get("calibration_id", "")),
                     task=str(context.get("task", "")),
                     relative_actions=bool(context.get("relative_actions", False)),
                     roles=tuple(tuple(r) for r in saved.get("camera_roles") or ()),
                     backend=str(context.get("backend", "onnx")))
        return build_spec(g, context)
    if checkpoint is None:
        return spec
    return build_spec(geometry_from_config(checkpoint, backend=backend))
