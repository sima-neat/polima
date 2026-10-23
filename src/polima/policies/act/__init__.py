"""ACT (Action Chunking Transformer).

ACT has no language conditioning, which is why the dataset contract is
single-task by default (relaxed per-invocation with --allow-mixed-tasks).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from polima.policies.act.runtime import (
    ACTION_DIM,
    CAMERA_ELEMENTS,
    CAMERA_TOKENS,
    CHUNK,
    ENCODER_TOKENS,
    HIDDEN,
    HIDDEN_ELEMENTS,
    PADDED_ACTION_DIM,
    STEM_ELEMENTS,
    STEM_TOKENS,
)
from polima.policies.base import (
    CalibrationSource,
    CompilePlan,
    DatasetContract,
    GraphSpec,
    PolicySpec,
    RobotSpec,
    TensorSpec,
    WireSpec,
    camera_capture_for_profile,
    camera_hints_for_profile,
)
from polima.policies.registry import register_policy

HEIGHT, WIDTH = 480, 640
STATE_DIM = 6

#: The six SO-101 follower joints, in dataset order.
JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

_ACT_PROFILES = {
    "so-arm101": {
        "state_names": JOINT_NAMES,
        "camera_keys": ("observation.images.overhead", "observation.images.wrist"),
        "fps": 30,
    },
}


def _json_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _checkpoint_profile(checkpoint: str | Path, dataset_root: str | Path | None) -> dict:
    checkpoint = Path(checkpoint)
    config = _json_file(checkpoint / "config.json")
    inputs = config.get("input_features", {})
    outputs = config.get("output_features", {})
    state_dim = tuple(inputs.get("observation.state", {}).get("shape", ()))
    action_dim = tuple(outputs.get("action", {}).get("shape", ()))
    camera_keys = tuple(key for key in inputs if key.startswith("observation.images."))

    if dataset_root is None:
        train = _json_file(checkpoint / "train_config.json")
        dataset_root = train.get("dataset", {}).get("root")
    info = _json_file(Path(dataset_root) / "meta/info.json") if dataset_root else {}
    features = info.get("features", {})
    state_feature = features.get("observation.state", {})
    action_feature = features.get("action", {})

    state_names = tuple(state_feature.get("names") or ())
    action_names = tuple(action_feature.get("names") or ())
    if state_names and action_names and state_names != action_names:
        raise ValueError(
            f"ACT dataset state/action joint names differ: {state_names!r} versus {action_names!r}"
        )

    state_width = state_dim[0] if len(state_dim) == 1 else None
    action_width = action_dim[0] if len(action_dim) == 1 else None
    if state_width != action_width or state_width != 6:
        raise ValueError(
            "ACT supports the SO-101 (6 matching state/action values); "
            f"checkpoint has {state_dim}/{action_dim}"
        )

    calibration_id = "so-arm101"
    expected = _ACT_PROFILES[calibration_id]
    names = state_names or expected["state_names"]
    cameras = camera_keys or tuple(key for key in features if key.startswith("observation.images."))
    fps = info.get("fps", expected["fps"])

    if tuple(names) != expected["state_names"]:
        raise ValueError(
            f"{calibration_id} ACT joint order must be {expected['state_names']}, "
            f"got {tuple(names)}"
        )
    if set(cameras) != set(expected["camera_keys"]):
        raise ValueError(
            f"{calibration_id} ACT cameras must be {expected['camera_keys']}, got {tuple(cameras)}"
        )
    if fps != expected["fps"]:
        raise ValueError(f"{calibration_id} ACT datasets require {expected['fps']} fps, got {fps}")

    return {
        "state_names": list(names),
        "state_dim": state_width,
        "action_dim": action_width,
        "camera_keys": list(cameras),
        "fps": fps,
        "calibration_id": calibration_id,
    }


def specialize_spec(
    spec: PolicySpec,
    *,
    checkpoint: str | Path | None = None,
    dataset_root: str | Path | None = None,
    profile: Mapping[str, Any] | None = None,
) -> PolicySpec:
    """Specialize the shared ACT graph architecture for the SO-101."""

    selected = dict(profile or {})
    if not selected:
        if checkpoint is None:
            return spec
        selected = _checkpoint_profile(checkpoint, dataset_root)

    calibration_id = str(selected.get("calibration_id", ""))
    if calibration_id not in _ACT_PROFILES:
        raise ValueError(
            f"unsupported ACT robot profile {calibration_id!r}; "
            f"expected one of {tuple(_ACT_PROFILES)}"
        )
    expected = _ACT_PROFILES[calibration_id]
    state_names = tuple(selected.get("state_names") or expected["state_names"])
    state_dim = int(selected.get("state_dim", len(state_names)))
    action_dim = int(selected.get("action_dim", state_dim))
    camera_keys = tuple(selected.get("camera_keys") or expected["camera_keys"])
    fps = int(selected.get("fps", expected["fps"]))

    if (
        state_names != expected["state_names"]
        or state_dim != len(expected["state_names"])
        or action_dim != state_dim
        or set(camera_keys) != set(expected["camera_keys"])
        or fps != expected["fps"]
    ):
        raise ValueError(f"invalid {calibration_id} ACT profile metadata: {selected}")

    dataset = replace(
        spec.dataset,
        state_names=state_names,
        state_dim=state_dim,
        action_dim=action_dim,
        camera_keys=camera_keys,
        fps=fps,
    )
    request_tensors = tuple(
        replace(tensor, shape=(state_dim,)) if tensor.name == "state" else tensor
        for tensor in spec.wire.request_tensors
    )
    wire = replace(
        spec.wire,
        request_tensors=request_tensors,
        response_shape=(spec.robot.actions_per_chunk, action_dim),
    )

    roles = tuple(key.rsplit(".", 1)[-1] for key in camera_keys)
    camera_fourcc, camera_capture_fps = camera_capture_for_profile(
        calibration_id, roles, spec.robot.camera_fourcc, spec.robot.camera_capture_fps
    )
    robot = replace(
        spec.robot,
        camera_roles=tuple((role, role.replace("_", " ").title()) for role in roles),
        camera_hints=camera_hints_for_profile(calibration_id, roles, spec.robot.camera_hints),
        camera_fourcc=camera_fourcc,
        camera_capture_fps=camera_capture_fps,
        joint_names=state_names,
        default_fps=fps,
        calibration_id=calibration_id,
    )

    return replace(spec, dataset=dataset, wire=wire, robot=robot)


#: Applied only with --augmentations.
AUGMENTATION_TFS = (
    '{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.9,1.1]}},'
    '"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.9,1.1]}},'
    '"affine":{"weight":0.5,"type":"RandomAffine",'
    '"kwargs":{"degrees":[-1.0,1.0],"translate":[0.02,0.02]}}}'
)

# The encoder hidden state, shared by four graphs. Rank 4 because the MLA path
# needs NHWC; the graph reshapes to (1, 602, 512) internally.
_HIDDEN_IN = TensorSpec("hidden", (1, 1, ENCODER_TOKENS, HIDDEN))
_HIDDEN_OUT = TensorSpec("hidden_out", (1, 1, ENCODER_TOKENS, HIDDEN))


def _encoder_layer(index: int) -> GraphSpec:
    return GraphSpec(
        name=f"encoder_layer_{index:02d}",
        builder="polima.policies.act.graphs:EncoderLayerRank4",
        inputs=(_HIDDEN_IN,),
        outputs=(_HIDDEN_OUT,),
        layout="NHWC",
        precision="bf16",
        calibration=CalibrationSource("npz", samples=8),
    )


ACT_SPEC = PolicySpec(
    name="act",
    display_name="ACT (Action Chunking Transformer)",
    # --------------------------------------------------------------- dataset --
    dataset=DatasetContract(
        state_names=JOINT_NAMES,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        camera_keys=("observation.images.overhead", "observation.images.wrist"),
        camera_shape=(HEIGHT, WIDTH, 3),
        fps=30,
        codebase_version="v3.0",
        single_task=True,
        task_canonicalizer="polima.data.contract:canonical_task",
    ),
    # --------------------------------------------------------------- compile --
    compile=CompilePlan(
        export_entry="polima.policies.act.graphs:export_all",
        verify_entry="polima.policies.act.graphs:verify_chain",
        fixture_entry="polima.policies.act.graphs:write_fixtures",
        normalization_entry="polima.export.normalization:write",
        fixture_file="act_fixture.npz",
        verify_atol=1e-4,
        verify_rtol=1e-3,
        graphs=(
            # ACT is deliberately BF16-only: ModelSDK INT8 saturates this
            # backbone's weights and produces non-finite values on Modalix.
            # A BF16 failure must stop rather than publish an unsafe fallback.
            # The only NCHW graph -- it takes a raw image. Everything downstream
            # is NHWC token data.
            GraphSpec(
                name="vision_backbone",
                builder="polima.policies.act.graphs:VisionBackbone",
                inputs=(TensorSpec("image", (1, 3, HEIGHT, WIDTH)),),
                outputs=(TensorSpec("camera_tokens", (1, CAMERA_TOKENS, HIDDEN)),),
                layout="NCHW",
                precision="bf16",
                calibration=CalibrationSource("npz", samples=8),
            ),
            # Consumes the host-packed (1, 601, 512): token 0 = state,
            # tokens 1..300 = camera 0, tokens 301..600 = camera 1.
            GraphSpec(
                name="encoder_layer_00_stem",
                builder="polima.policies.act.graphs:EncoderStemPacked",
                inputs=(TensorSpec("stem_input", (1, 1, STEM_TOKENS, HIDDEN)),),
                outputs=(_HIDDEN_OUT,),
                layout="NHWC",
                precision="bf16",
                calibration=CalibrationSource("npz", samples=8),
            ),
            _encoder_layer(1),
            _encoder_layer(2),
            _encoder_layer(3),
            # Action head widened 6 -> 16 for MLA channel alignment and
            # zero-filled; the host strides past the padding.
            GraphSpec(
                name="decoder_action_tail",
                builder="polima.policies.act.graphs:DecoderActionRank4",
                inputs=(_HIDDEN_IN,),
                outputs=(TensorSpec("normalized_actions", (1, 1, CHUNK, PADDED_ACTION_DIM)),),
                layout="NHWC",
                precision="bf16",
                calibration=CalibrationSource("npz", samples=8),
            ),
        ),
    ),
    # ------------------------------------------------------------------ wire --
    wire=WireSpec(
        magic=0x4D544341,  # "ACTM" little-endian
        # Version 2: camera frames travel as raw uint8 and the server normalizes
        # them (plan.json wire decode). A version-1 server refuses the request.
        version=2,
        default_port=8092,
        request_header="<IIII",
        response_header="<IIIIfI",
        request_tensors=(
            # HWC uint8 camera frames, 4x smaller than float32. Normalizing two
            # 640x480 frames in Python took ~58 ms per request on the DevKit; the
            # server decodes them with the same float32 arithmetic instead.
            TensorSpec("image0", (HEIGHT, WIDTH, 3), dtype="uint8"),
            TensorSpec("image1", (HEIGHT, WIDTH, 3), dtype="uint8"),
            # Normalized by the client from normalization_stats.npz.
            TensorSpec("state", (STATE_DIM,)),
        ),
        response_shape=(CHUNK, ACTION_DIM),
        normalization_side="client",
        stats_file="normalization_stats.npz",
    ),
    # ----------------------------------------------------------------- robot --
    robot=RobotSpec(
        camera_roles=(("overhead", "Overhead"), ("wrist", "Wrist")),
        # Default cameras: a Logitech C920 overhead and a Sonix wrist camera.
        camera_hints={"overhead": "C920", "wrist": "Sonix"},
        joint_names=JOINT_NAMES,
        actions_per_chunk=CHUNK,
        default_fps=30,
        max_relative_target=12,
        aggregate_fn="weighted_average",
    ),
    runtime_plan_builder="polima.policies.act.runtime:build_plan",
    checkpoint_validator="polima.policies.act.graphs:validate_checkpoint",
    profile_entry="polima.policies.act:specialize_spec",
    # Both cameras, the stem, the encoder and the decoder as one device program:
    # 19.0 -> 15.8 ms on Modalix, outputs byte-identical to the per-graph plan.
    device_chain=("act",),
)

register_policy(ACT_SPEC)

__all__ = ["ACT_SPEC", "AUGMENTATION_TFS", "CAMERA_ELEMENTS", "HIDDEN_ELEMENTS", "JOINT_NAMES", "STEM_ELEMENTS"]
