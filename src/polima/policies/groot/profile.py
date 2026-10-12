"""Checkpoint-driven robot and graph geometry for GR00T N1.6."""

from __future__ import annotations

import importlib.util
import json
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from polima.policies.base import (
    CalibrationSource,
    PolicySpec,
    TensorSpec,
    camera_capture_for_profile,
    camera_hints_for_profile,
)
from polima.policies.groot import runtime as rt

MYCOBOT_JOINT_NAMES = (
    "joint_1.pos",
    "joint_2.pos",
    "joint_3.pos",
    "joint_4.pos",
    "joint_5.pos",
    "joint_6.pos",
    "gripper.pos",
)


def _json_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _processor_config(checkpoint: Path) -> dict:
    for relative in (
        "processor_config.json",
        "experiment_cfg/final_processor_config.json",
    ):
        value = _json_file(checkpoint / relative)
        if value:
            return value
    raise FileNotFoundError(f"GR00T processor configuration is missing under {checkpoint}")


def _statistics(checkpoint: Path) -> dict:
    for relative in ("statistics.json", "experiment_cfg/dataset_statistics.json"):
        value = _json_file(checkpoint / relative)
        if value:
            return value
    raise FileNotFoundError(f"GR00T statistics are missing under {checkpoint}")


def _eagle_preprocessor_config(checkpoint: Path, model_name: str) -> dict:
    """Load the image arithmetic used by GR00T's nested Eagle processor.

    GR00T checkpoints save their own resize/crop configuration but do not copy
    Eagle's ``preprocessor_config.json``. Export still has the Eagle package
    available (it is required to load the checkpoint), so resolve it there and
    make the resulting build contract self-contained. A checkpoint-local copy
    is accepted as well, which keeps tests and relocated installations portable.
    """

    candidates = [checkpoint / "eagle_preprocessor_config.json"]
    model_path = Path(model_name).expanduser()
    if model_path.is_dir():
        candidates.append(model_path / "preprocessor_config.json")

    relative_model = Path(*model_name.split("/"))
    for parent in checkpoint.parents:
        candidates.append(
            parent
            / "Isaac-GR00T"
            / "gr00t"
            / "model"
            / "modules"
            / relative_model
            / "preprocessor_config.json"
        )

    try:
        gr00t_spec = importlib.util.find_spec("gr00t")
    except (ImportError, ModuleNotFoundError, ValueError):
        gr00t_spec = None
    if gr00t_spec and gr00t_spec.submodule_search_locations:
        for root in gr00t_spec.submodule_search_locations:
            candidates.append(
                Path(root) / "model" / "modules" / relative_model / "preprocessor_config.json"
            )

    for path in candidates:
        value = _json_file(path)
        if value and "image_mean" in value and "image_std" in value:
            return value
    raise FileNotFoundError(
        f"cannot locate Eagle preprocessor_config.json for {model_name!r}; "
        "GR00T export cannot create an exact live-camera preprocessing contract"
    )


def _image_preprocess_contract(checkpoint: Path, kwargs: Mapping[str, Any]) -> dict:
    model_name = str(kwargs.get("model_name") or "nvidia/Eagle-Block2A-2B-v2")
    eagle = _eagle_preprocessor_config(checkpoint, model_name)
    if not eagle.get("do_convert_rgb", True):
        raise ValueError("GR00T Eagle processor must convert camera images to RGB")
    if eagle.get("do_resize", False):
        raise ValueError(
            "unsupported GR00T Eagle image processor: resizing must be owned by "
            "process_vision_info before normalization"
        )
    if not eagle.get("do_rescale", False) or not eagle.get("do_normalize", False):
        raise ValueError("GR00T Eagle processor must rescale and normalize images")

    try:
        rescale_factor = float(eagle["rescale_factor"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("GR00T Eagle image rescale factor is missing or invalid") from error
    if rescale_factor <= 0:
        raise ValueError("GR00T Eagle image rescale factor must be positive")
    mean = eagle.get("image_mean")
    std = eagle.get("image_std")
    if not (
        isinstance(mean, list)
        and isinstance(std, list)
        and len(mean) == len(std) == 3
        and all(float(value) > 0 for value in std)
    ):
        raise ValueError("GR00T Eagle image mean/std must each contain three channels")

    return {
        "format": "groot-eagle-image-v1",
        "color_space": "RGB",
        "gr00t_eval": {
            "image_crop_size": kwargs.get("image_crop_size"),
            "image_target_size": kwargs.get("image_target_size"),
            "shortest_image_edge": kwargs.get("shortest_image_edge"),
            "crop_fraction": kwargs.get("crop_fraction"),
            "use_albumentations": bool(kwargs.get("use_albumentations", False)),
            "letter_box_transform": bool(kwargs.get("letter_box_transform", False)),
        },
        # Eagle's process_vision_info performs smart_resize first. The traced
        # image_shape records its resolved result, so the standalone client can
        # resize directly to that exact fixed geometry.
        "eagle": {
            "resize_to_compiled_shape": True,
            "resize_resample": "bicubic",
            "rescale_factor": rescale_factor,
            "image_mean": [float(value) for value in mean],
            "image_std": [float(value) for value in std],
        },
    }


def _training_embodiment(
    checkpoint: Path,
    modalities: Mapping[str, Any],
    embodiment_tag: str | None = None,
) -> str:
    if embodiment_tag:
        selected = embodiment_tag.strip().lower()
        if selected not in modalities:
            raise ValueError(
                f"GR00T checkpoint has no processor configuration for embodiment "
                f"{selected!r}; available: {sorted(modalities)}"
            )
        return selected
    for relative in ("experiment_cfg/config.yaml", "experiment_cfg/conf.yaml"):
        try:
            text = (checkpoint / relative).read_text(encoding="utf-8")
        except OSError:
            continue
        matches = re.findall(r"^\s*embodiment_tag:\s*['\"]?([\w-]+)", text, re.MULTILINE)
        matches = [value.lower() for value in matches if value.lower() in modalities]
        if matches:
            if len(set(matches)) != 1:
                raise ValueError(f"GR00T checkpoint records multiple embodiments: {matches}")
            return matches[0]
    if "new_embodiment" in modalities:
        return "new_embodiment"
    if len(modalities) == 1:
        return next(iter(modalities))
    raise ValueError(
        "cannot determine the GR00T training embodiment from experiment_cfg/config.yaml"
    )


def _group_width(statistics: Mapping[str, Any], section: str, key: str) -> int:
    try:
        values = statistics[section][key]["mean"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"GR00T statistics have no {section}.{key}.mean") from error
    if not isinstance(values, list) or not values:
        raise ValueError(f"GR00T statistics {section}.{key}.mean is empty")
    return len(values[0]) if isinstance(values[0], list) else len(values)


def _state_encoding_contract(kwargs, state_config, widths):
    keys = tuple(state_config.get("modality_keys") or ())
    sincos_keys = set(state_config.get("sin_cos_embedding_keys") or ())
    if not kwargs.get("apply_sincos_state_encoding", False) or not sincos_keys:
        return None
    if sincos_keys != set(keys):
        raise ValueError("mixed affine/sincos GR00T state groups are not supported")
    return {"kind": "grouped_sincos", "group_widths": list(widths)}


def checkpoint_state_encoding(checkpoint: str | Path, embodiment_tag: str | None = None):
    """Read state preprocessing without requiring the external Eagle image cache."""
    checkpoint = Path(checkpoint)
    processor = _processor_config(checkpoint)
    kwargs = processor.get("processor_kwargs") or processor
    modalities = kwargs.get("modality_configs") or {}
    tag = _training_embodiment(checkpoint, modalities, embodiment_tag)
    state_config = modalities[tag]["state"]
    statistics = _statistics(checkpoint)[tag]
    widths = [_group_width(statistics, "state", key) for key in state_config["modality_keys"]]
    return _state_encoding_contract(kwargs, state_config, widths)


def checkpoint_metadata(
    checkpoint: str | Path,
    dataset_root: str | Path | None = None,
    embodiment_tag: str | None = None,
) -> dict[str, Any]:
    """Return the saved embodiment, real widths, horizon and camera contract."""

    checkpoint = Path(checkpoint)
    processor = _processor_config(checkpoint)
    kwargs = processor.get("processor_kwargs") or processor
    modalities = kwargs.get("modality_configs") or {}
    tag = _training_embodiment(checkpoint, modalities, embodiment_tag)
    modality = modalities.get(tag) or {}
    statistics = _statistics(checkpoint)
    embodiment_stats = statistics.get(tag) or {}

    state_keys = tuple((modality.get("state") or {}).get("modality_keys") or ())
    action_cfg = modality.get("action") or {}
    action_keys = tuple(action_cfg.get("modality_keys") or ())
    video_keys = tuple((modality.get("video") or {}).get("modality_keys") or ())
    if not state_keys or state_keys != action_keys:
        raise ValueError(
            f"GR00T requires matching state/action groups; found {state_keys}/{action_keys}"
        )
    state_widths = tuple(_group_width(embodiment_stats, "state", key) for key in state_keys)
    action_widths = tuple(_group_width(embodiment_stats, "action", key) for key in action_keys)
    if state_widths != action_widths:
        raise ValueError(f"GR00T state/action group widths differ: {state_widths}/{action_widths}")
    state_encoding = _state_encoding_contract(kwargs, modality.get("state") or {}, state_widths)
    width = sum(state_widths)
    if width == 7 and state_keys == ("single_arm", "gripper") and state_widths == (6, 1):
        calibration_id = "mycobot280"
        state_names = MYCOBOT_JOINT_NAMES
        expected_fps = 20
        expected_cameras = (
            "observation.images.front",
            "observation.images.wrist",
        )
    elif width == 6 and state_keys == ("single_arm", "gripper") and state_widths == (5, 1):
        calibration_id = "so-arm101"
        state_names = (
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        )
        expected_fps = 30
        expected_cameras = (
            "observation.images.overhead",
            "observation.images.wrist",
        )
    elif (
        tag == "gr1"
        and state_keys == ("left_arm", "right_arm", "left_hand", "right_hand", "waist")
        and state_widths == (7, 7, 6, 6, 3)
    ):
        calibration_id = "gr1"
        state_names = tuple(
            f"{key}.{index}"
            for key, group_width in zip(state_keys, state_widths, strict=True)
            for index in range(group_width)
        )
        expected_fps = 20
        expected_cameras = ("observation.images.ego_view",)
    else:
        widths = dict(zip(state_keys, state_widths, strict=True))
        raise ValueError(
            "PoLiMa GR00T supports SO-101 (5+1), MyCobot (6+1), or Fourier GR1; "
            f"checkpoint {tag!r} has groups {widths}"
        )

    delta_indices = tuple(action_cfg.get("delta_indices") or ())
    if not delta_indices or delta_indices != tuple(range(len(delta_indices))):
        raise ValueError(
            f"GR00T action delta indices must be contiguous from zero: {delta_indices}"
        )
    horizon = len(delta_indices)
    model = _json_file(checkpoint / "config.json")
    model_horizon = int(model.get("action_horizon", 0))
    if not 0 < horizon <= model_horizon:
        raise ValueError(
            f"GR00T processor horizon {horizon} does not fit model horizon {model_horizon}"
        )

    camera_keys = expected_cameras
    source_camera_keys = expected_cameras
    fps = expected_fps
    if dataset_root:
        dataset_root = Path(dataset_root)
        info = _json_file(dataset_root / "meta/info.json")
        if info.get("codebase_version") not in {"v2.0", "v2.1"}:
            raise ValueError(
                f"GR00T calibration dataset must be LeRobot v2.0 or v2.1, got "
                f"{info.get('codebase_version')!r} at {dataset_root}"
            )
        features = info.get("features") or {}
        found = tuple(key for key in features if key.startswith("observation.images."))
        if found:
            source_camera_keys = found
            if calibration_id != "gr1":
                camera_keys = found
        fps = int(info.get("fps", fps))
        modality_meta = _json_file(dataset_root / "meta/modality.json")
        if not modality_meta:
            raise ValueError(
                f"GR00T calibration dataset has no valid {dataset_root / 'meta/modality.json'}"
            )
        for section, keys, widths in (
            ("state", state_keys, state_widths),
            ("action", action_keys, action_widths),
        ):
            for key, group_width in zip(keys, widths, strict=True):
                entry = (modality_meta.get(section) or {}).get(key) or {}
                found_range = (entry.get("start"), entry.get("end"))
                if (
                    not all(isinstance(value, int) for value in found_range)
                    or found_range[1] - found_range[0] != group_width
                ):
                    raise ValueError(
                        f"GR00T {calibration_id} {section}.{key} must select "
                        f"{group_width} values, found {found_range}"
                    )
        video_meta = modality_meta.get("video") or {}
        mapped_cameras = tuple(
            (video_meta.get(key) or {}).get("original_key") for key in video_keys
        )
        if calibration_id == "gr1":
            if len(mapped_cameras) != 1 or mapped_cameras[0] not in source_camera_keys:
                raise ValueError(
                    "GR00T gr1 trained video alias must map to the dataset's one "
                    f"image feature; found {mapped_cameras} in {source_camera_keys}"
                )
        elif mapped_cameras != expected_cameras:
            raise ValueError(
                f"GR00T {calibration_id} video map must be {expected_cameras}, "
                f"found {mapped_cameras}"
            )
        else:
            # The model's image order is its processor's video keys, not the order
            # of the dataset's info.json features (the SO-101 data lists wrist first).
            camera_keys = mapped_cameras
    expected_video_keys = {
        "mycobot280": {"front", "wrist"},
        "so-arm101": {"front", "wrist"},
        "gr1": {"ego_view_bg_crop_pad_res256_freq20"},
    }[calibration_id]
    if set(video_keys) != expected_video_keys:
        raise ValueError(
            f"{calibration_id} GR00T video modalities must be "
            f"{sorted(expected_video_keys)}, got {video_keys}"
        )
    camera_layout_matches = (
        len(source_camera_keys) == 1
        if calibration_id == "gr1"
        else set(camera_keys) == set(expected_cameras)
    )
    if not camera_layout_matches or fps != expected_fps:
        raise ValueError(
            f"{calibration_id} GR00T dataset requires one trained camera slot at "
            f"{expected_fps} fps; found {source_camera_keys} at {fps} fps"
        )

    ids = _json_file(checkpoint / "embodiment_id.json")
    if tag not in ids:
        raise ValueError(f"GR00T checkpoint has no projector id for embodiment {tag!r}")
    action_configs = tuple(action_cfg.get("action_configs") or ())
    if len(action_configs) != len(action_keys):
        raise ValueError(f"GR00T action representation is incomplete for {action_keys}")
    relative_width = 0
    absolute_seen = False
    for key, group_width, config in zip(action_keys, action_widths, action_configs, strict=True):
        representation = str(config.get("rep", "")).lower()
        if representation == "relative":
            if absolute_seen:
                raise ValueError(
                    "GR00T runtime requires relative action groups to form one "
                    f"leading prefix; {key!r} follows an absolute group"
                )
            relative_width += group_width
        elif representation == "absolute":
            absolute_seen = True
        else:
            raise ValueError(
                f"unsupported GR00T action representation {representation!r} for {key!r}"
            )

    return {
        "state_names": list(state_names),
        "state_dim": width,
        "action_dim": width,
        "camera_keys": list(camera_keys),
        "source_camera_keys": list(source_camera_keys),
        "image_preprocess": _image_preprocess_contract(checkpoint, kwargs),
        "fps": fps,
        "calibration_id": calibration_id,
        "actions_per_chunk": horizon,
        "embodiment_tag": tag,
        "embodiment_id": int(ids[tag]),
        "relative_action_dim": relative_width,
        "state_encoding": state_encoding,
        "state_keys": list(state_keys),
        "action_keys": list(action_keys),
        "video_keys": list(video_keys),
    }


def _geometry(spec: PolicySpec, context: Mapping[str, Any]) -> PolicySpec:
    if "sequence" not in context:
        return replace(spec, runtime_context=dict(context))
    sequence = int(context["sequence"])
    starts = tuple(int(value) for value in context.get("image_token_starts", ()))
    patch_grid = tuple(
        int(value) for value in context.get("patch_grid", (rt.PATCH_SIDE, rt.PATCH_SIDE))
    )
    connector_width = int(context.get("connector_width", rt.CONNECTOR_WIDTH))
    if (
        sequence <= 0
        or not starts
        or len(patch_grid) != 2
        or any(value <= 0 or value % rt.UNSHUFFLE for value in patch_grid)
        or connector_width != (patch_grid[0] // rt.UNSHUFFLE) * (patch_grid[1] // rt.UNSHUFFLE)
        or any(start < 0 or start + connector_width > sequence for start in starts)
    ):
        raise ValueError(f"invalid GR00T runtime geometry: {context}")
    vision_width = patch_grid[0] * patch_grid[1]

    graphs = []
    for graph in spec.compile.graphs:
        inputs = []
        for tensor in graph.inputs:
            if graph.name == "eagle_vision_connector" and tensor.name == "hidden":
                tensor = replace(tensor, shape=(1, connector_width, tensor.shape[-1]))
            elif graph.name.startswith("eagle_vision_"):
                tensor = replace(tensor, shape=(1, vision_width, tensor.shape[-1]))
            elif tensor.name == "backbone_features":
                tensor = replace(tensor, shape=(1, sequence, tensor.shape[-1]))
            elif tensor.name == "additive_mask":
                tensor = replace(tensor, shape=(1, sequence))
            elif graph.name.startswith("eagle_qwen_") and tensor.name == "hidden":
                tensor = replace(tensor, shape=(1, 1, sequence, tensor.shape[-1]))
            elif graph.name.startswith("eagle_qwen_") and tensor.name == "attention_mask":
                tensor = replace(tensor, shape=(1, sequence, sequence, 1))
            elif graph.name == "eagle_output_norm" and tensor.name == "hidden":
                tensor = replace(tensor, shape=(1, sequence, tensor.shape[-1]))
            inputs.append(tensor)
        outputs = []
        for tensor in graph.outputs:
            if graph.name == "eagle_vision_connector":
                tensor = replace(tensor, shape=(1, connector_width, tensor.shape[-1]))
            elif graph.name.startswith("eagle_vision_"):
                tensor = replace(tensor, shape=(1, vision_width, tensor.shape[-1]))
            elif graph.name.startswith("eagle_qwen_"):
                tensor = replace(tensor, shape=(1, 1, sequence, tensor.shape[-1]))
            elif graph.name == "eagle_output_norm":
                tensor = replace(tensor, shape=(1, sequence, tensor.shape[-1]))
            outputs.append(tensor)
        calibration = graph.calibration
        if graph.name.startswith("eagle_vision_"):
            calibration = CalibrationSource(
                calibration.kind,
                calibration.producer,
                samples=max(calibration.samples, len(starts)),
            )
        graphs.append(
            replace(
                graph,
                inputs=tuple(inputs),
                outputs=tuple(outputs),
                calibration=calibration,
            )
        )

    patches = tuple(
        TensorSpec(f"patches{index}", (vision_width, rt.PATCH_CHANNELS))
        for index in range(len(starts))
    )
    others = tuple(t for t in spec.wire.request_tensors if not t.name.startswith("patches"))
    wire = replace(spec.wire, request_tensors=(*patches, *others))
    if context.get("task_catalog"):
        wire = replace(wire, version=2)
    return replace(
        spec,
        compile=replace(spec.compile, graphs=tuple(graphs)),
        wire=wire,
        runtime_context=dict(context),
    )


def specialize_spec(
    spec: PolicySpec,
    *,
    checkpoint: str | Path | None = None,
    dataset_root: str | Path | None = None,
    embodiment_tag: str | None = None,
    profile: Mapping[str, Any] | None = None,
) -> PolicySpec:
    """Specialize GR00T's padded architecture for one explicit robot profile."""

    selected = dict(profile or {})
    if selected.get("runtime_context", {}).get("model_version") == "1.7":
        from polima.policies.groot.n17 import specialize_spec as specialize_n17

        return specialize_n17(spec, selected)
    metadata_root = dataset_root
    if dataset_root is not None and Path(dataset_root).is_file():
        metadata_root = None
    saved_context = selected.get("runtime_context") or {}
    missing_image_preprocess = (
        "image_preprocess" not in selected and "image_preprocess" not in saved_context
    )
    if selected and missing_image_preprocess and checkpoint and Path(checkpoint).is_dir():
        # Early portable build contracts saved the traced graph geometry but
        # omitted the deterministic evaluation transform needed by live camera
        # inference. Recover only that metadata from the original checkpoint;
        # the compiled ELFs and their content identity remain untouched.
        selected["image_preprocess"] = checkpoint_metadata(
            checkpoint, metadata_root, embodiment_tag
        )["image_preprocess"]
    elif selected and missing_image_preprocess:
        raise ValueError(
            "GR00T build contract lacks live-camera preprocessing metadata and "
            "its original checkpoint is unavailable; re-export the checkpoint"
        )
    if not selected:
        if checkpoint is None:
            return spec
        selected = checkpoint_metadata(checkpoint, metadata_root, embodiment_tag)

    calibration_id = str(selected.get("calibration_id", ""))
    expected_names = {
        "so-arm101": spec.dataset.state_names,
        "mycobot280": MYCOBOT_JOINT_NAMES,
        "gr1": tuple(selected.get("state_names") or ()),
    }
    if calibration_id not in expected_names:
        raise ValueError(f"unsupported GR00T robot profile {calibration_id!r}")
    state_names = tuple(selected.get("state_names") or expected_names[calibration_id])
    state_dim = int(selected.get("state_dim", len(state_names)))
    action_dim = int(selected.get("action_dim", state_dim))
    camera_keys = tuple(selected.get("camera_keys") or spec.dataset.camera_keys)
    fps = int(selected.get("fps", spec.dataset.fps))
    horizon = int(selected.get("actions_per_chunk", spec.robot.actions_per_chunk))
    if (
        not state_names
        or state_names != expected_names[calibration_id]
        or state_dim != len(state_names)
    ):
        raise ValueError(f"invalid {calibration_id} GR00T state profile: {selected}")
    if action_dim != state_dim or not 0 < horizon <= 50:
        raise ValueError(f"invalid {calibration_id} GR00T action profile: {selected}")

    dataset = replace(
        spec.dataset,
        state_names=state_names,
        state_dim=state_dim,
        action_dim=action_dim,
        camera_keys=camera_keys,
        fps=fps,
        rename_map={key: key.rsplit(".", 1)[-1] for key in camera_keys},
    )
    request = tuple(
        replace(tensor, shape=(state_dim,)) if tensor.name == "state" else tensor
        for tensor in spec.wire.request_tensors
    )
    wire = replace(
        spec.wire,
        request_tensors=request,
        response_shape=(horizon, action_dim),
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
        actions_per_chunk=horizon,
        default_fps=fps,
        calibration_id=calibration_id,
    )
    context = dict(selected.get("runtime_context") or {})
    for name in ("embodiment_tag", "embodiment_id", "relative_action_dim", "image_preprocess", "state_encoding"):
        if name in selected:
            context[name] = selected[name]
    return _geometry(replace(spec, dataset=dataset, wire=wire, robot=robot), context)


def specialize_geometry(spec: PolicySpec, reference: Mapping[str, Any]) -> PolicySpec:
    """Add traced prompt geometry and its exact trained-task catalog."""

    catalog = list(reference.get("task_catalog") or ())
    if not catalog:
        raise ValueError("GR00T trace has no task catalog")
    if [int(task.get("id", -1)) for task in catalog] != list(range(len(catalog))):
        raise ValueError(f"invalid GR00T task catalog ids: {catalog}")

    context = dict(spec.runtime_context)
    context.update(
        {
            "sequence": int(reference["sequence"]),
            "image_shape": [int(value) for value in reference["image_shape"]],
            "patch_grid": [int(value) for value in reference["patch_grid"]],
            "connector_width": int(reference["connector_width"]),
            "image_token_starts": [int(value) for value in reference["image_token_starts"]],
            "task_catalog": catalog,
        }
    )
    specialized = _geometry(spec, context)
    # Version 2 assigns request-header flags to the catalog task id. An older
    # client would always send flags=0, so rejecting it is safer than silently
    # running the first instruction.
    return replace(specialized, wire=replace(specialized.wire, version=2))


__all__ = [
    "MYCOBOT_JOINT_NAMES",
    "checkpoint_metadata",
    "specialize_geometry",
    "specialize_spec",
]
