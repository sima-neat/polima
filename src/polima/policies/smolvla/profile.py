"""Robot-profile selection for the shared SmolVLA graph architecture."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from polima.policies.base import (
    PolicySpec,
    camera_capture_for_profile,
    camera_hints_for_profile,
)


def _json_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _profiles(spec: PolicySpec) -> dict[str, dict[str, Any]]:
    return {
        "so-arm101": {
            "state_names": spec.dataset.state_names,
            "camera_keys": ("observation.images.overhead", "observation.images.wrist"),
            "fps": 30,
        },
    }


def _checkpoint_profile(
    spec: PolicySpec,
    checkpoint: str | Path,
    dataset_root: str | Path | None,
) -> dict:
    """Read the real robot width and order from the training dataset.

    SmolVLA's prepare_state pads the sample itself to the 32-wide model lane, so
    the dataset is authoritative for the host wire width. The checkpoint action
    width still has to agree because it controls output unpadding.
    """

    checkpoint = Path(checkpoint)
    config = _json_file(checkpoint / "config.json")
    outputs = config.get("output_features", {})
    checkpoint_action_shape = tuple(outputs.get("action", {}).get("shape", ()))

    if dataset_root is None:
        train = _json_file(checkpoint / "train_config.json")
        dataset_root = train.get("dataset", {}).get("root")
    if not dataset_root:
        raise ValueError(
            "SmolVLA robot profile requires the training dataset metadata; pass --dataset-root"
        )

    info = _json_file(Path(dataset_root) / "meta/info.json")
    features = info.get("features", {})
    state_feature = features.get("observation.state", {})
    action_feature = features.get("action", {})
    state_shape = tuple(state_feature.get("shape") or ())
    action_shape = tuple(action_feature.get("shape") or ())
    state_names = tuple(state_feature.get("names") or ())
    action_names = tuple(action_feature.get("names") or ())

    if len(state_shape) != 1 or len(action_shape) != 1 or state_shape != action_shape:
        raise ValueError(
            "SmolVLA requires matching one-dimensional state/action features; "
            f"dataset has {state_shape}/{action_shape}"
        )
    if state_names != action_names:
        raise ValueError(
            "SmolVLA dataset state/action joint names differ: "
            f"{state_names!r} versus {action_names!r}"
        )

    width = state_shape[0]
    if width != 6:
        raise ValueError(f"SmolVLA supports the SO-101 (6 joints); found width {width}")
    calibration_id = "so-arm101"
    expected = _profiles(spec)[calibration_id]
    cameras = _model_camera_order(
        checkpoint,
        config,
        tuple(key for key in features if key.startswith("observation.images.")),
    )
    fps = info.get("fps")

    if state_names != expected["state_names"]:
        raise ValueError(
            f"{calibration_id} SmolVLA joint order must be {expected['state_names']}, "
            f"got {state_names}"
        )
    if set(cameras) != set(expected["camera_keys"]):
        raise ValueError(
            f"{calibration_id} SmolVLA cameras must be {expected['camera_keys']}, got {cameras}"
        )
    if fps != expected["fps"]:
        raise ValueError(
            f"{calibration_id} SmolVLA datasets require {expected['fps']} fps, got {fps}"
        )
    if checkpoint_action_shape != (width,):
        raise ValueError(
            "SmolVLA checkpoint action width must match its dataset; "
            f"checkpoint has {checkpoint_action_shape}, dataset has {(width,)}"
        )
    if int(config.get("max_state_dim", 0)) < width or int(config.get("max_action_dim", 0)) < width:
        raise ValueError(f"SmolVLA checkpoint padding lanes cannot hold {width} robot channels")

    profile = {
        "state_names": list(state_names),
        "state_dim": width,
        "action_dim": width,
        "camera_keys": list(cameras),
        "fps": fps,
        "calibration_id": calibration_id,
    }
    task = _single_task(Path(dataset_root))
    if task is not None:
        # The exporter bakes the language embedding of the dataset sample it
        # traces. With exactly one task that sample's task is this string, so
        # the bundle can name the instruction its constants actually encode.
        profile["runtime_context"] = {
            "task_catalog": [{"id": 0, "dataset_id": 0, "instruction": task}],
        }
    return profile


def _model_camera_order(
    checkpoint: Path, config: Mapping[str, Any], dataset_cameras: tuple[str, ...]
) -> tuple[str, ...]:
    """Dataset camera keys in the order the model consumes them.

    SmolVLA's ``prepare_images`` walks ``config.input_features`` in order
    (camera1, camera2, ...), after the checkpoint's rename processor maps
    dataset keys onto those names. The exporter traces cameras in that order,
    so wire slot image0 is the model's first camera. A dataset's own feature
    order is unrelated: grey_eraser_961 lists wrist before overhead while its
    checkpoint maps overhead -> camera1. Using dataset order there silently
    swaps the two images.
    """

    rename: dict[str, str] = {}
    manifest = _json_file(checkpoint / "policy_preprocessor.json")
    for step in manifest.get("steps", []):
        if step.get("registry_name") == "rename_observations_processor":
            rename.update((step.get("config") or {}).get("rename_map") or {})
    inverse = {target: source for source, target in rename.items()}
    ordered = []
    for key in config.get("input_features", {}):
        source = inverse.get(key, key)
        if source in dataset_cameras and source not in ordered:
            ordered.append(source)
    if set(ordered) == set(dataset_cameras):
        return tuple(ordered)
    return dataset_cameras


def _single_task(dataset_root: Path) -> str | None:
    """The dataset's instruction when it has exactly one, else None."""

    tasks: list[str] = []
    parquet = dataset_root / "meta/tasks.parquet"
    jsonl = dataset_root / "meta/tasks.jsonl"
    try:
        if parquet.is_file():
            import pandas as pd

            frame = pd.read_parquet(parquet)
            tasks = [str(value) for value in (
                frame["task"].tolist() if "task" in frame.columns else frame.index.tolist()
            )]
        elif jsonl.is_file():
            tasks = [
                str(json.loads(line)["task"])
                for line in jsonl.read_text().splitlines() if line.strip()
            ]
    except (ImportError, OSError, KeyError, ValueError):
        return None
    return tasks[0] if len(tasks) == 1 else None


def specialize_spec(
    spec: PolicySpec,
    *,
    checkpoint: str | Path | None = None,
    dataset_root: str | Path | None = None,
    profile: Mapping[str, Any] | None = None,
) -> PolicySpec:
    """Specialize SmolVLA's shared 32-wide graph lane for one robot."""

    selected = dict(profile or {})
    if not selected:
        if checkpoint is None:
            return spec
        selected = _checkpoint_profile(spec, checkpoint, dataset_root)

    profiles = _profiles(spec)
    calibration_id = str(selected.get("calibration_id", ""))
    if calibration_id not in profiles:
        raise ValueError(
            f"unsupported SmolVLA robot profile {calibration_id!r}; "
            f"expected one of {tuple(profiles)}"
        )
    expected = profiles[calibration_id]
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
        raise ValueError(f"invalid {calibration_id} SmolVLA profile metadata: {selected}")

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
    runtime_context = dict(selected.get("runtime_context") or {})
    return replace(
        spec, dataset=dataset, wire=wire, robot=robot, runtime_context=runtime_context
    )


__all__ = ["specialize_spec"]
