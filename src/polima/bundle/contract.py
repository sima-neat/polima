"""Restore the exact policy contract carried by a deployable bundle.

Registry policy specs describe the default robot. A compiled bundle may
specialize that architecture for a different robot (for example a 7-channel
arm instead of the 6-channel SO-101), so consumers must not go back to
registry defaults after loading ``bundle.json``.
"""

from __future__ import annotations

from dataclasses import replace

from polima.bundle.layout import Bundle
from polima.policies.base import PolicySpec, TensorSpec
from polima.util.jsonio import read_json


def spec_from_bundle(spec: PolicySpec, bundle: Bundle) -> PolicySpec:
    """Return ``spec`` specialized to the contract in ``bundle.plan_path``.

    New plans contain a complete ``profile`` block. The fallback reconstructs
    the same fields from the ``robot`` and ``wire`` blocks of a plan without
    one.
    """

    plan = read_json(bundle.plan_path)
    profile = plan.get("profile") or _profile_from_plan_blocks(plan)
    if profile and spec.profile_entry:
        spec = spec.specialize_profile(profile=profile)

    wire = plan.get("wire") or {}
    if wire:
        request_tensors = tuple(
            TensorSpec(item["name"], tuple(int(value) for value in item["shape"]),
                       dtype=item.get("dtype", "float32"))
            for item in wire.get("request_tensors", ())
        )
        spec = replace(
            spec,
            wire=replace(
                spec.wire,
                magic=int(wire.get("magic", spec.wire.magic)),
                version=int(wire.get("version", spec.wire.version)),
                default_port=int(wire.get("default_port", spec.wire.default_port)),
                request_tensors=request_tensors or spec.wire.request_tensors,
                response_shape=tuple(
                    int(value) for value in wire.get("response_shape", spec.wire.response_shape)
                ),
            ),
        )

    # The bundle's own pass mark for its fixture check (e.g. mixed GR00T's envelope).
    marks = {k: float(v) for k, v in (bundle.smoke or {}).items() if k in ("cosine_min", "mean_abs_max")}
    if marks:
        spec = replace(spec, smoke=replace(spec.smoke, **marks))

    # Validate after the plan's wire geometry is applied. This catches a
    # malformed or partially-copied bundle before a socket or robot is opened.
    spec.validate()
    return spec


def _profile_from_plan_blocks(plan: dict) -> dict:
    robot = plan.get("robot") or {}
    wire = plan.get("wire") or {}
    calibration_id = robot.get("calibration_id")
    if not calibration_id:
        return {}

    request = {
        item.get("name"): item for item in wire.get("request_tensors", ()) if isinstance(item, dict)
    }
    state_shape = tuple((request.get("state") or {}).get("shape") or ())
    response_shape = tuple(wire.get("response_shape") or ())
    roles = [pair[0] for pair in robot.get("camera_roles", ()) if pair]
    camera_keys = [
        role if str(role).startswith("observation.images.") else f"observation.images.{role}"
        for role in roles
    ]
    state_names = list(robot.get("joint_names") or ())
    state_dim = int(state_shape[-1]) if state_shape else len(state_names)
    action_dim = int(response_shape[-1]) if response_shape else state_dim
    actions_per_chunk = int(
        robot.get("actions_per_chunk") or (response_shape[0] if len(response_shape) > 1 else 0)
    )
    profile = {
        "state_names": state_names,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "camera_keys": camera_keys,
        "fps": int(robot.get("fps", 0)),
        "calibration_id": str(calibration_id),
        "actions_per_chunk": actions_per_chunk,
    }
    runtime_context = plan.get("runtime_context")
    if runtime_context:
        profile["runtime_context"] = runtime_context
    return profile
