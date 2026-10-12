import socket
import threading

from polima.robot.so101 import SO101PolicyClient

"""GR00T transport/packaging through the established SO101 client interface."""

import json
from argparse import Namespace

import numpy as np
import pytest

from polima.bundle.layout import Bundle
from polima.bundle.pack import _plan_dict
from polima.policies.registry import get_policy
from polima.robot.so101 import JOINTS, client_configuration, load_contract
from polima.util.jsonio import write_json


def _bundle_plan(spec):
    """plan.json as a GR00T SO-101 bundle carries it.

    Bundles record the robot profile and the checkpoint's runtime context next
    to the plan; the robot client restores the exact contract from them
    (polima.bundle.contract.spec_from_bundle). Packing those blocks is part of
    the GR00T compile tooling, so the fixture adds them here.
    """
    plan = _plan_dict(spec.build_runtime_plan(), wire=spec)
    plan["robot"]["camera_capture_fps"] = dict(spec.robot.camera_capture_fps)
    context = dict(spec.runtime_context)
    catalog = list(context.get("task_catalog") or ())
    if catalog:
        plan["task_catalog"] = catalog
        plan["tasks"] = [str(task["instruction"]) for task in catalog]
        plan["default_task"] = 0
    plan["profile"] = {
        "state_names": list(spec.dataset.state_names),
        "state_dim": spec.dataset.state_dim,
        "action_dim": spec.dataset.action_dim,
        "camera_keys": list(spec.dataset.camera_keys),
        "fps": spec.dataset.fps,
        "calibration_id": spec.robot.calibration_id,
        "actions_per_chunk": spec.robot.actions_per_chunk,
        "runtime_context": context,
    }
    if context:
        plan["runtime_context"] = context
    return plan

def make_bundle(tmp_path, camera_roles=("overhead", "wrist")):
    spec = get_policy("groot").specialize_profile(
        profile={
            "state_names": list(JOINTS),
            "state_dim": 6,
            "action_dim": 6,
            "camera_keys": [f"observation.images.{role}" for role in camera_roles],
            "fps": 30,
            "calibration_id": "so-arm101",
            "actions_per_chunk": 16,
            "embodiment_tag": "new_embodiment",
            "embodiment_id": 10,
            "relative_action_dim": 5,
            "image_preprocess": {"format": "groot-eagle-image-v1"},
        }
    )
    from polima.policies.groot.profile import specialize_geometry

    spec = specialize_geometry(
        spec,
        {
            "sequence": 257,
            "image_shape": [252, 336],
            "patch_grid": [18, 24],
            "connector_width": 108,
            "image_token_starts": [31, 146],
            "task_catalog": [
                {"id": 0, "dataset_id": 0, "instruction": "first task"},
                {"id": 1, "dataset_id": 1, "instruction": "second task"},
            ],
        },
    )
    bundle = Bundle(
        root=tmp_path,
        policy="groot",
        bundle_id="groot-so101-test",
        tasks=["first task", "second task"],
    )
    write_json(tmp_path / "plan.json", _bundle_plan(spec))
    write_json(tmp_path / "bundle.json", bundle.to_dict())
    write_json(
        tmp_path / "robot_client/robot_rest_position.json", {"positions": dict.fromkeys(JOINTS, 0)}
    )
    write_json(tmp_path / "robot_client/calibration/so-arm101.json", {})
    return bundle, spec

def test_contract_and_task_validation(tmp_path):
    bundle, _spec = make_bundle(tmp_path)
    assert load_contract(tmp_path, "second task")[2] == 1
    assert load_contract(tmp_path, "0")[1].wire.response_shape == (16, 6)
    with pytest.raises(ValueError, match="catalog"):
        load_contract(tmp_path, "2")
    data = json.loads(bundle.plan_path.read_text())
    data["profile"]["calibration_id"] = "mycobot280"
    write_json(bundle.plan_path, data)
    with pytest.raises(ValueError):
        load_contract(tmp_path)

def test_launcher_keeps_existing_lerobot_control_configuration(tmp_path):
    bundle, spec = make_bundle(tmp_path)
    args = Namespace(
        actions_per_chunk=16,
        fps=30,
        robot_port="/dev/arm",
        front_camera="/dev/front",
        wrist_camera="/dev/wrist",
        server_address="127.0.0.1:8093",
        max_relative_target=12,
    )
    env, argv = client_configuration(tmp_path, spec, 1, bundle.tasks[1], args)
    assert env["ACT_SOM_DIRECT"] == "1"
    assert env["POLIMA_TASK_ID"] == "1"
    assert "--task=second task" in argv
    assert "--actions_per_chunk=16" in argv
    assert "--robot.use_degrees=true" in argv
    assert "--aggregate_fn_name=weighted_average" in argv
    cameras = json.loads(next(v.split("=", 1)[1] for v in argv if v.startswith("--robot.cameras=")))
    assert cameras["overhead"]["index_or_path"] == "/dev/front"
    assert cameras["wrist"]["index_or_path"] == "/dev/wrist"
    args.actions_per_chunk = 17
    with pytest.raises(ValueError, match="horizon"):
        client_configuration(tmp_path, spec, 0, bundle.tasks[0], args)

@pytest.mark.parametrize("control_fps", [10, 15, 30])
@pytest.mark.parametrize("capture, expected", [
    ({}, {"overhead": 30, "wrist": 30}),
    ({"front": 30, "wrist": 25}, {"overhead": 30, "wrist": 25}),
    ({"overhead": 20, "front": 30, "wrist": 25}, {"overhead": 20, "wrist": 25}),
])
def test_control_rate_does_not_change_camera_mode(tmp_path, control_fps, capture, expected):
    from dataclasses import replace
    bundle, spec = make_bundle(tmp_path)
    spec = replace(spec, robot=replace(spec.robot, camera_capture_fps=capture))
    args = Namespace(actions_per_chunk=16, fps=control_fps, robot_port="/dev/arm",
                     front_camera="/dev/front", wrist_camera="/dev/wrist",
                     server_address="127.0.0.1:8093", max_relative_target=12)
    _, argv = client_configuration(tmp_path, spec, 0, bundle.tasks[0], args)
    cameras = json.loads(next(a.split("=", 1)[1] for a in argv if a.startswith("--robot.cameras=")))
    assert {role: config["fps"] for role, config in cameras.items()} == expected
    assert f"--fps={control_fps}" in argv
    assert "--robot.max_relative_target=12" in argv

@pytest.mark.parametrize("mode", ["random", "zero"])
def test_noise_mode_controls_only_the_noise_tensor(tmp_path, monkeypatch, mode):
    from polima.robot.policy import CompiledPolicyAdapter

    bundle, spec = make_bundle(tmp_path)
    patch_shape = next(t.shape for t in spec.wire.request_tensors if t.name == "patches0")
    monkeypatch.setattr(
        "polima.robot.policy.prepare_groot_image",
        lambda image, context: np.zeros(patch_shape, dtype=np.float32),
    )
    adapter = CompiledPolicyAdapter(spec, bundle, "127.0.0.1:1", noise=mode, seed=0,
                                    client_factory=lambda *a, **k: None)
    images = [np.zeros((480, 640, 3), dtype=np.uint8)] * 2
    first = adapter._prepare_request(np.zeros(6, dtype=np.float32), images)
    second = adapter._prepare_request(np.zeros(6, dtype=np.float32), images)
    assert first["noise"].shape == (50, 128) and first["noise"].dtype == np.float32
    if mode == "zero":
        assert not first["noise"].any() and not second["noise"].any()
    else:
        assert first["noise"].std() > 0.5 and not np.array_equal(first["noise"], second["noise"])
    with pytest.raises(ValueError, match="noise"):
        CompiledPolicyAdapter(spec, bundle, "127.0.0.1:1", noise="fixed",
                              client_factory=lambda *a, **k: None)

def test_launcher_aggregate_selection(tmp_path, monkeypatch):
    bundle, spec = make_bundle(tmp_path)
    args = Namespace(actions_per_chunk=16, fps=30, robot_port="/dev/arm", front_camera="/dev/front",
                     wrist_camera="/dev/wrist", server_address="127.0.0.1:8093", max_relative_target=12)
    monkeypatch.setenv("POLIMA_AGGREGATE", "latest_only")
    _, argv = client_configuration(tmp_path, spec, 0, bundle.tasks[0], args)
    assert "--aggregate_fn_name=latest_only" in argv
    monkeypatch.setenv("POLIMA_AGGREGATE", "smooth")
    with pytest.raises(ValueError, match="POLIMA_AGGREGATE"):
        client_configuration(tmp_path, spec, 0, bundle.tasks[0], args)

def test_smoothing_selection_and_default_aggregate(tmp_path, monkeypatch):
    bundle, spec = make_bundle(tmp_path)
    args = Namespace(actions_per_chunk=16, fps=15, robot_port="/dev/arm", front_camera="/dev/front",
                     wrist_camera="/dev/wrist", server_address="127.0.0.1:8093", max_relative_target=12)
    monkeypatch.delenv("POLIMA_AGGREGATE", raising=False)
    monkeypatch.delenv("POLIMA_SMOOTH_MOVES", raising=False)
    _, argv = client_configuration(tmp_path, spec, 0, bundle.tasks[0], args)
    assert "--aggregate_fn_name=weighted_average" in argv  # unchanged default when off
    monkeypatch.setenv("POLIMA_SMOOTH_MOVES", "0")
    _, argv = client_configuration(tmp_path, spec, 0, bundle.tasks[0], args)
    assert "--aggregate_fn_name=weighted_average" in argv
    monkeypatch.setenv("POLIMA_SMOOTH_MOVES", "4")
    _, argv = client_configuration(tmp_path, spec, 0, bundle.tasks[0], args)
    assert "--aggregate_fn_name=latest_only" in argv
    assert "--robot.max_relative_target=12" in argv  # safety clamp untouched
    for aggregate in ("average", "weighted_average"):
        monkeypatch.setenv("POLIMA_AGGREGATE", aggregate)
        with pytest.raises(ValueError, match="latest_only"):
            client_configuration(tmp_path, spec, 0, bundle.tasks[0], args)
    monkeypatch.setenv("POLIMA_AGGREGATE", "latest_only")
    client_configuration(tmp_path, spec, 0, bundle.tasks[0], args)
    monkeypatch.setenv("POLIMA_SMOOTH_SHAPE", "cubic")
    with pytest.raises(ValueError, match="POLIMA_SMOOTH_SHAPE"):
        client_configuration(tmp_path, spec, 0, bundle.tasks[0], args)
    monkeypatch.setenv("POLIMA_SMOOTH_SHAPE", "smoothstep")
    client_configuration(tmp_path, spec, 0, bundle.tasks[0], args)
    for value in ("x", "-1", "33", "2.5"):
        monkeypatch.setenv("POLIMA_SMOOTH_MOVES", value)
        with pytest.raises(ValueError, match="POLIMA_SMOOTH_MOVES"):
            client_configuration(tmp_path, spec, 0, bundle.tasks[0], args)

class FakeClock:
    def __init__(self, period=1 / 15):
        self.now, self.period = 0.0, period

    def __call__(self):
        self.now += self.period
        return self.now

def ramp(start, count, step, first=0, gripper=10.0):
    """Plan {timestep: target}; arm joints move `step` per timestep."""
    return {first + i: np.array([start + step * i] * 5 + [gripper]) for i in range(count)}

def test_plan_switch_spreads_jump_over_k_moves():
    from polima.robot.smoothing import Plan, PlanSwitchSmoother

    smoother = PlanSwitchSmoother(4, clock=FakeClock())
    old, new = Plan(ramp(0.0, 16, 1.0)), Plan(ramp(8.0, 16, 1.0, gripper=30.0))
    sent = [smoother(old.steps[t], old, t) for t in range(7)]
    np.testing.assert_array_equal(sent[-1], old.steps[6])  # one plan: untouched
    sent += [smoother(new.steps[t], new, t) for t in range(7, 13)]
    arm = [s[0] for s in sent]
    # The 8-degree switch jump is split into four 2-degree parts on top of the plan's 1/move.
    np.testing.assert_allclose(np.diff(arm[6:12]), [3, 3, 3, 3, 1])
    np.testing.assert_array_equal(sent[11], new.steps[11])  # exactly the new plan after K moves
    assert [s[5] for s in sent[7:]] == [30.0] * 6  # gripper follows the plan unchanged

def test_smoothstep_eases_the_switch_in_and_out():
    from polima.robot.smoothing import Plan, PlanSwitchSmoother

    smoother = PlanSwitchSmoother(5, clock=FakeClock(), shape="smoothstep")
    old, new = Plan(ramp(0.0, 16, 0.0)), Plan(ramp(10.0, 16, 0.0))
    sent = [smoother(old.steps[t], old, t)[0] for t in range(3)]
    sent += [smoother(new.steps[t], new, t)[0] for t in range(3, 10)]
    parts = np.diff(sent[2:])
    expected = np.diff([0.0] + [10 * x * x * (3 - 2 * x) for x in np.arange(1, 6) / 5] + [10, 10])
    np.testing.assert_allclose(parts, expected)
    assert parts[0] < 10 / 5 < parts[2]  # small first part, larger middle, back to plan after 5
    assert sent[-3:] == [10.0] * 3

def test_plan_switch_continuity_and_resets():
    from polima.robot.smoothing import Plan, PlanSwitchSmoother

    clock = FakeClock()
    smoother = PlanSwitchSmoother(4, clock=clock)
    a, b, c = Plan(ramp(0.0, 16, 1.0)), Plan(ramp(8.0, 16, 1.0)), Plan(ramp(-8.0, 16, 1.0))
    out = [smoother(a.steps[t], a, t)[0] for t in range(5)]
    out += [smoother(b.steps[t], b, t)[0] for t in range(5, 7)]  # switch mid-decay ...
    out += [smoother(c.steps[t], c, t)[0] for t in range(7, 13)]  # ... continues from the sent target
    assert np.abs(np.diff(out)).max() <= 5.0 + 1e-9
    assert out[-1] == c.steps[12][0]
    # A repeated timestep (LeRobot merge race): no own movement, first quarter of the jump.
    d = Plan(ramp(20.0, 16, 1.0))
    assert smoother(d.steps[12], d, 12)[0] == out[-1] + (d.steps[12][0] - out[-1]) / 4
    # Untagged actions pass through; a long pause starts fresh.
    assert smoother(np.full(6, 50.0), None, None)[0] == 50.0
    smoother(a.steps[0], a, 0)
    clock.now += 2.0
    np.testing.assert_array_equal(smoother(b.steps[1], b, 1), b.steps[1])


@pytest.mark.parametrize("camera_roles", [
    ("overhead", "wrist"), ("wrist", "overhead"),
    ("front", "wrist"), ("wrist", "front"),
])
def test_transport_uses_real_wire_and_decodes_six_joint_chunk(tmp_path, monkeypatch, camera_roles):
    import struct

    _bundle, spec = make_bundle(tmp_path, camera_roles)
    monkeypatch.setenv("POLIMA_SO101_BUNDLE", str(tmp_path))
    monkeypatch.setenv("POLIMA_TASK_ID", "1")
    patch_shape = next(t.shape for t in spec.wire.request_tensors if t.name == "patches0")
    monkeypatch.setattr(
        "polima.robot.policy.prepare_groot_image",
        lambda image, context: np.full(patch_shape, image[0, 0, 0], dtype=np.float32),
    )
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(5)
    captured = {}
    expected = np.arange(96, dtype="<f4").reshape(16, 6)

    def receive_exact(stream, count):
        result = bytearray()
        while len(result) < count:
            part = stream.recv(count - len(result))
            if not part:
                raise RuntimeError("incomplete request")
            result.extend(part)
        return bytes(result)

    def serve():
        try:
            stream, _ = listener.accept()
            with stream:
                stream.settimeout(5)
                magic, version, request_id, task = struct.unpack("<IIII", receive_exact(stream, 16))
                captured.update(magic=magic, version=version, task=task)
                for tensor in spec.wire.request_tensors:
                    captured[tensor.name] = np.frombuffer(
                        receive_exact(stream, tensor.elements * 4), dtype="<f4"
                    ).reshape(tensor.shape)
                response = (
                    struct.pack("<IIIIfI", magic, version, request_id, 0, 12.5, expected.size)
                    + expected.tobytes()
                )
                # Deliberately fragment the response to exercise exact receive logic.
                stream.sendall(response[:9])
                stream.sendall(response[9:])
        except (OSError, RuntimeError, ValueError) as error:
            captured["error"] = error
        finally:
            listener.close()

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    client = SO101PolicyClient(f"127.0.0.1:{listener.getsockname()[1]}", timeout=5)
    state = np.arange(6, dtype=np.float32)
    actions, latency = client.predict(
        np.zeros((480, 640, 3), dtype=np.uint8), np.ones((480, 640, 3), dtype=np.uint8), state
    )
    worker.join(6)
    assert not worker.is_alive()
    assert "error" not in captured, captured
    assert captured["magic"] == spec.wire.magic and captured["task"] == 1
    np.testing.assert_array_equal(captured["state"], state)
    for index, role in enumerate(camera_roles):
        assert np.all(captured[f"patches{index}"] == (1 if role == "wrist" else 0))
    assert captured["noise"].shape == (50, 128)
    np.testing.assert_array_equal(actions, expected)  # no second normalization
    assert latency == 12.5
    with pytest.raises(ValueError, match="six finite"):
        client.predict(None, None, np.zeros(7))
