"""GR00T transport/packaging through the established SO101 client interface."""

import json
import socket
import threading
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from polima.bundle.import_legacy import collect_robot_files
from polima.bundle.layout import Bundle
from polima.bundle.pack import _plan_dict
from polima.policies.registry import get_policy
from polima.robot.so101 import JOINTS, SO101PolicyClient, client_configuration, load_contract
from polima.util.jsonio import write_json


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
    write_json(tmp_path / "plan.json", _plan_dict(spec.build_runtime_plan(), wire=spec))
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
    from types import SimpleNamespace
    events = []
    monkeypatch.setattr("polima.robot.so101._TRIAL_METRICS", SimpleNamespace(
        record=lambda kind, **values: events.append((kind, values))))
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
    assert [kind for kind, _ in events] == ["request_started", "request_sent", "reply_received", "prediction"]
    assert len({values["request_id"] for _, values in events}) == 1
    assert events[2][1]["server_compute_ms"] == 12.5
    with pytest.raises(ValueError, match="six finite"):
        client.predict(None, None, np.zeros(7))


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


def test_packaging_reuses_control_source_and_requires_own_rest_pose(tmp_path, monkeypatch):
    bundle, spec = make_bundle(tmp_path / "export")
    stacks = tmp_path / "stacks"
    act = stacks / "ACT"
    for file in [
        "run_robot_client_with_live_view.py",
        "preview_robot_cameras.py",
        ".camera_focus_config.json",
        "calibration/robots/so_follower/so-arm101.json",
    ]:
        dest = act / file
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("existing source")
    monkeypatch.setenv("POLIMA_REPO_ROOT", str(stacks))
    assets = collect_robot_files(spec, bundle.root)
    assert (
        assets["run_robot_client_with_live_view.py"] == act / "run_robot_client_with_live_view.py"
    )
    assert (
        assets["robot_rest_position.json"] == bundle.root / "robot_client/robot_rest_position.json"
    )
    assert "SO101PolicyClient" in assets["scripts/act_som_client.py"].read_text()
    assert all(p.is_file() for p in assets.values())
    with pytest.raises(FileNotFoundError, match="dataset-derived"):
        collect_robot_files(spec)


def test_rest_pose_uses_only_episode_starts(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from polima.export.robot import write_so101_rest_pose

    write_json(
        tmp_path / "meta/info.json", {"features": {"observation.state": {"names": list(JOINTS)}}}
    )
    (tmp_path / "data").mkdir()
    pd.DataFrame(
        {"frame_index": [0, 1, 0], "observation.state": [[1.0] * 6, [999.0] * 6, [3.0] * 6]}
    ).to_parquet(tmp_path / "data/episodes.parquet")
    output = tmp_path / "rest.json"
    write_so101_rest_pose(tmp_path, output, JOINTS)
    assert list(json.loads(output.read_text())["positions"].values()) == [2.0] * 6


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
    monkeypatch.setenv("POLIMA_AGGREGATE", "crossfade")
    _, argv = client_configuration(tmp_path, spec, 0, bundle.tasks[0], args)
    assert "--aggregate_fn_name=crossfade" in argv
    monkeypatch.setenv("POLIMA_AGGREGATE", "smooth")
    with pytest.raises(ValueError, match="POLIMA_AGGREGATE"):
        client_configuration(tmp_path, spec, 0, bundle.tasks[0], args)


def lerobot_merge(source=None):
    """LeRobot's own _aggregate_action_queues, executed without importing its gRPC client."""
    import ast
    import importlib.util
    from collections.abc import Callable
    from queue import Queue

    import torch

    if source is None:
        spec = importlib.util.find_spec("lerobot")  # the async package itself requires grpcio
        if spec is None:
            pytest.skip("lerobot is not installed")
        source = Path(spec.origin).parent / "async_inference" / "robot_client.py"
    tree = ast.parse(Path(source).read_text())
    (node,) = [n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_aggregate_action_queues"]

    class TimedAction:
        def __init__(self, timestamp, timestep, action):
            self.timestamp, self.timestep, self.action = timestamp, timestep, action

        def get_timestamp(self):
            return self.timestamp

        def get_timestep(self):
            return self.timestep

        def get_action(self):
            return self.action

    namespace = {"Queue": Queue, "TimedAction": TimedAction, "Callable": Callable, "torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), source, "exec"), namespace)  # noqa: S102 - the shipped function itself
    return namespace["_aggregate_action_queues"], TimedAction


def check_crossfade(source=None):
    from queue import Queue
    from types import SimpleNamespace

    import torch

    from polima.robot import crossfade

    merge, TimedAction = lerobot_merge(source)

    class Client:  # only what LeRobot's merge touches
        _aggregate_action_queues = merge

    configs = SimpleNamespace(AGGREGATE_FUNCTIONS={})
    aggregate = crossfade.install(configs, Client)
    assert configs.AGGREGATE_FUNCTIONS["crossfade"] is aggregate
    client = Client()
    client.action_queue_lock = threading.Lock()
    client.latest_action_lock = threading.Lock()
    client.latest_action = 44
    client.action_queue = Queue()
    for step in range(45, 50):  # rest of the previous plan: value 0
        client.action_queue.put(TimedAction(timestamp=0.0, timestep=step, action=torch.zeros(6)))
    incoming = [TimedAction(timestamp=1.0, timestep=step, action=torch.ones(6)) for step in range(24, 74)]
    client._aggregate_action_queues(incoming, aggregate)
    merged = {a.get_timestep(): float(a.get_action()[0]) for a in client.action_queue.queue}
    assert sorted(merged) == list(range(45, 74))  # executed timesteps dropped, as before
    assert [round(merged[s], 6) for s in range(45, 50)] == [round(i / 6, 6) for i in range(1, 6)]
    assert all(merged[s] == 1.0 for s in range(50, 74))  # new-only timesteps unchanged
    with pytest.raises(RuntimeError, match="merge hook"):
        crossfade.Crossfade()(torch.zeros(1), torch.ones(1))


def test_crossfade_through_lerobot_merge():
    check_crossfade()


# --- POLIMA_SMOOTH_MOVES: plan-switch smoothing ------------------------------------------

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
    for aggregate in ("crossfade", "weighted_average"):
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


def lerobot_sources():
    """Installed LeRobot, plus POLIMA_LEROBOT_SOURCE (e.g. the devkit's 0.4.4 robot_client.py)."""
    import importlib.util
    import os

    sources = []
    spec = importlib.util.find_spec("lerobot")
    if spec is not None:
        sources.append(Path(spec.origin).parent / "async_inference" / "robot_client.py")
    if os.environ.get("POLIMA_LEROBOT_SOURCE"):
        sources.append(Path(os.environ["POLIMA_LEROBOT_SOURCE"]))
    return sources


def functions_from(source, names, namespace):
    import ast

    tree = ast.parse(Path(source).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if not nodes:  # LeRobot's are methods of RobotClient
        nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    assert sorted(n.name for n in nodes) == sorted(names)
    import __future__

    # The packaged script uses postponed annotations (RobotClient etc. are not needed).
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec",
                   flags=__future__.annotations.compiler_flag, dont_inherit=True)
    exec(code, namespace)  # noqa: S102 - the packaged script's own functions
    return [namespace[name] for name in names]


@pytest.mark.parametrize("shape, expected", [("linear", [6, 9, 12, 15, 18, 19, 20]),
                                             ("smoothstep", [6, 8.25, 12, 15.75, 18, 19, 20])])
@pytest.mark.parametrize("source", lerobot_sources() or [None], ids=lambda p: str(p))
def test_smoothing_through_lerobot_and_bundle_script(tmp_path, monkeypatch, source, shape, expected):
    """LeRobot's queue/control methods + the packaged script's receiver, rest snap and
    completion wrappers, installed in launch order, with metrics and action trace."""
    import time
    from collections.abc import Callable
    from queue import Empty, Queue
    from types import SimpleNamespace

    import torch

    from polima.robot import smoothing
    from polima.robot.action_trace import ActionTrace
    from polima.robot.metrics import TrialMetrics, instrument_loop
    from polima.util.paths import repo_root

    if source is None or not Path(source).is_file():
        pytest.skip("LeRobot robot_client.py is not available")
    script = repo_root() / "ACT/run_robot_client_with_live_view.py"
    if not script.is_file():
        pytest.skip("packaged SO101 robot-client script is not available (set POLIMA_REPO_ROOT)")

    class TimedAction:
        def __init__(self, timestamp, timestep, action):
            self.timestamp, self.timestep, self.action = timestamp, timestep, action

        def get_timestamp(self):
            return self.timestamp

        def get_timestep(self):
            return self.timestep

        def get_action(self):
            return self.action

    lerobot = {"Queue": Queue, "TimedAction": TimedAction, "Callable": Callable, "torch": torch,
               "time": time, "Any": object}
    merge, control, convert = functions_from(
        source, ["_aggregate_action_queues", "control_loop_action", "_action_tensor_to_action_dict"], lerobot)

    class Robot:
        action_features = dict.fromkeys(JOINTS, float)

        def __init__(self):
            self.sent = []
            self.bus = SimpleNamespace(sync_read=lambda register, *a, **k: dict.fromkeys(JOINTS, 0.0))

        def send_action(self, action):
            self.sent.append(dict(action))
            return dict(action)

    class Client:
        def __init__(self):
            self.robot = Robot()

        @property
        def running(self):
            return not self.shutdown_event.is_set()

        def actions_available(self):
            with self.action_queue_lock:
                return not self.action_queue.empty()

        def control_loop_observation(self, task, verbose=False):
            return None

    Client._aggregate_action_queues = merge
    Client.control_loop_action = control
    Client._action_tensor_to_action_dict = convert

    # Launch order: polima.robot.so101.main installs, then the packaged script captures.
    smoothing.install(Client, 4, shape=shape)
    bundle = {"np": np, "torch": torch, "time": time, "Empty": Empty, "TimedAction": TimedAction,
              "_return_to_reset_position": lambda *a, **k: True,
              "_original_aggregate_action_queues": Client._aggregate_action_queues,
              "_original_control_loop_action": Client.control_loop_action,
              "_original_action_tensor_to_dict": Client._action_tensor_to_action_dict,
              "_JOINT_NAMES": JOINTS}
    receive, rest_snap, completion, latency_merge, start = functions_from(script, [
        "_som_receive_actions", "_action_tensor_to_dict_with_rest_snap",
        "_control_loop_action_with_completion", "_aggregate_action_queues_with_latency",
        "_som_start"], bundle)
    Client.start = start  # ActionTrace locates the script's return-to-rest through it
    Client._aggregate_action_queues = latency_merge
    Client.control_loop_action = completion
    Client._action_tensor_to_action_dict = rest_snap
    Client.receive_actions = receive
    # Then run_instrumented's metrics and POLIMA_ACTION_TRACE wrappers.
    metrics = TrialMetrics(tmp_path / "metrics", {})
    restore_metrics = instrument_loop(Client, metrics)
    restore_trace = ActionTrace(metrics).install(Client)

    plans = []

    class SOM:
        request_id = 0

        def predict(self, overhead, wrist, state):
            SOM.request_id += 1
            client.shutdown_event.set()  # one receive iteration per plan
            return plans.pop(0), 1.0

    client = Client()
    client.config = SimpleNamespace(actions_per_chunk=16, aggregate_fn=None)
    client.action_queue, client.action_queue_lock = Queue(), __import__("threading").Lock()
    client.latest_action, client.latest_action_lock = -1, __import__("threading").Lock()
    client.action_chunk_size, client.action_queue_size = -1, []
    client.start_barrier = SimpleNamespace(wait=lambda: None)
    client.shutdown_event, client.must_go = __import__("threading").Event(), __import__("threading").Event()
    client.logger = SimpleNamespace(info=lambda *a, **k: None, exception=lambda *a, **k: None,
                                    error=lambda *a, **k: None)
    client.som_client, client.som_observations = SOM(), Queue()
    client.reset_position = dict.fromkeys(JOINTS, -40.0)
    client.rest_snap_tolerance, client.rest_snap_arm_distance = 5.0, 15.0
    client._rest_snap_armed = client._rest_snap_latched = False
    client._rest_snap_latched_s = None
    client.auto_repeat_enabled, client.auto_repeat_pause_s = True, 2.0
    client.auto_complete_enabled = False

    def arrive(plan, observed_step):
        plans.append(np.stack([plan[t] for t in sorted(plan)]).astype(np.float32))
        client.shutdown_event.clear()
        observation = dict.fromkeys(JOINTS, 0.0)
        observation.update(overhead=None, wrist=None)
        client.som_observations.put(SimpleNamespace(get_observation=lambda: observation,
                                                    get_timestep=lambda: observed_step,
                                                    get_timestamp=lambda: time.time()))
        client.receive_actions()

    def run(count):
        return [client.control_loop_action()["elbow_flex.pos"] for _ in range(count)]

    old, new = ramp(0.0, 16, 1.0), ramp(8.0, 16, 1.0, gripper=30.0)
    arrive(old, 0)
    sent = run(7)
    arrive(new, 0)  # predicted from timestep 0, arrives after timestep 6 was executed
    queued = {a.get_timestep(): float(a.get_action()[2]) for a in client.action_queue.queue}
    sent += run(6)
    assert sent[:7] == [float(i) for i in range(7)]
    np.testing.assert_allclose(sent[6:], expected)
    assert queued == {t: new[t][2] for t in range(7, 16)}  # the queue itself is not modified
    assert client.latest_action == 12  # LeRobot's timestep bookkeeping is unchanged
    assert [a["gripper.pos"] for a in client.robot.sent[7:]] == [30.0] * 6

    trace = [e for e in metrics.events if e["event"] == "action_command"]
    assert trace[8]["blended_targets"]["elbow_flex.pos"] == new[8][2]  # policy target recorded
    assert trace[8]["requested_targets"]["elbow_flex.pos"] == 12.0  # smoothed target sent

    # Rest-snap hold: the script sends rest; when the hold ends it clears the queue for a
    # new cycle, and smoothing starts fresh instead of pulling toward the old cycle.
    client._rest_snap_armed = client._rest_snap_latched = True
    client._rest_snap_latched_s = time.monotonic() - 10
    assert run(1) == [-40.0]
    far = ramp(-40.0, 16, 0.5, first=13)
    arrive(far, 13)
    assert run(2) == [-39.5, -39.0]  # the new plan exactly, not pulled toward ~20

    restore_trace()
    restore_metrics()
    metrics.close()


@pytest.mark.parametrize("smooth, aggregate, shape", [
    ("4", None, None), ("5", None, "smoothstep"), ("0", "crossfade", None), (None, None, None)])
def test_launcher_installs_options_before_script_and_records_them(tmp_path, monkeypatch, smooth,
                                                                    aggregate, shape):
    import sys
    import types

    from polima.robot import so101

    make_bundle(tmp_path / "bundle")

    class RobotClient:
        def _aggregate_action_queues(self, incoming, aggregate_fn=None):
            pass

        def _action_tensor_to_action_dict(self, tensor):
            pass

    originals = dict(vars(RobotClient))
    module = types.ModuleType("lerobot.async_inference.robot_client")
    module.RobotClient, module.async_client = RobotClient, lambda: None
    package = types.ModuleType("lerobot.async_inference")  # the real one needs grpcio
    package.robot_client = module
    package.configs = types.ModuleType("lerobot.async_inference.configs")
    package.configs.AGGREGATE_FUNCTIONS = {}
    monkeypatch.setitem(sys.modules, "lerobot.async_inference.configs", package.configs)
    lerobot = pytest.importorskip("lerobot")
    monkeypatch.setattr(lerobot, "async_inference", package, raising=False)
    monkeypatch.setitem(sys.modules, "lerobot.async_inference", package)
    monkeypatch.setitem(sys.modules, "lerobot.async_inference.robot_client", module)
    seen = {}

    def run_path(path, run_name):
        # What the packaged script captures at import.
        seen["merge"] = RobotClient._aggregate_action_queues
        seen["convert"] = RobotClient._action_tensor_to_action_dict
        seen["argv"] = list(sys.argv)

    monkeypatch.setattr(so101.runpy, "run_path", run_path)
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    for name, value in (("POLIMA_SMOOTH_MOVES", smooth), ("POLIMA_AGGREGATE", aggregate),
                        ("POLIMA_SMOOTH_SHAPE", shape), ("POLIMA_SUBSTEPS", None),
                        ("POLIMA_RESET_EASE", None)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setenv("POLIMA_METRICS_DIR", str(tmp_path / "metrics"))
    monkeypatch.setenv("POLIMA_TASK_ID", "0")
    for name in ("POLIMA_SO101_BUNDLE", "ACT_SOM_DIRECT", "ACT_SOM_STATS", "ACT_REST_POSITION",
                 "ACT_CAMERA_FOCUS_CONFIG", "ACT_PERSPECTIVE_CAMERA", "ACT_WRIST_CAMERA"):
        monkeypatch.setenv(name, "")  # main() sets these; restored after the test
    so101.main(["--bundle", str(tmp_path / "bundle"), "--robot-port", "/dev/arm",
                "--front-camera", "/dev/front", "--wrist-camera", "/dev/wrist",
                "--server-address", "127.0.0.1:8093", "--fps", "15"])
    moves = int(smooth or 0)
    expected = aggregate or ("latest_only" if moves else "weighted_average")
    assert (seen["merge"] is not originals["_aggregate_action_queues"]) == (bool(moves) or bool(aggregate))
    assert (seen["convert"] is not originals["_action_tensor_to_action_dict"]) == bool(moves)
    assert f"--aggregate_fn_name={expected}" in seen["argv"]
    config = json.loads((tmp_path / "metrics/config.json").read_text())
    assert config["smooth_moves"] == moves and config["aggregate_fn_name"] == expected
    assert config["smooth_shape"] == (shape or "linear")
    assert config["substeps"] == 1 and config["reset_ease"] == "linear"  # defaults: off
    assert config["arm_p"] is None
    if moves:  # the installed conversion uses the requested shape
        smoother = next(c.cell_contents for c in seen["convert"].__closure__
                        if type(c.cell_contents).__name__ == "PlanSwitchSmoother")
        assert smoother.moves == moves
        assert smoother.remaining_fraction(0.25) == (0.75 if shape is None else 1 - 0.15625)


def test_launcher_installs_substeps_and_eased_reset_in_order(tmp_path, monkeypatch):
    """Sub-steps hook in before the script captures its methods; the eased reset replaces
    the script's function at async_client, before the action trace wraps it."""
    import sys
    import types

    from polima.robot import so101
    from polima.util.paths import repo_root

    script = repo_root() / "ACT/run_robot_client_with_live_view.py"
    if not script.is_file():
        pytest.skip("packaged SO101 robot-client script is not available (set POLIMA_REPO_ROOT)")
    make_bundle(tmp_path / "bundle")

    class RobotClient:
        def __init__(self, config=None):
            pass

        def _aggregate_action_queues(self, incoming, aggregate_fn=None):
            pass

        def _action_tensor_to_action_dict(self, tensor):
            pass

        def control_loop_action(self, verbose=False):
            pass

        def control_loop_observation(self, task, verbose=False):
            pass

        def actions_available(self):
            return False

    originals = dict(vars(RobotClient))
    seen = {}
    module = types.ModuleType("lerobot.async_inference.robot_client")
    module.RobotClient = RobotClient
    module.async_client = lambda: seen.update(reset=seen["namespace"]["_return_to_reset_position"])
    package = types.ModuleType("lerobot.async_inference")  # the real one needs grpcio
    package.robot_client = module
    lerobot = pytest.importorskip("lerobot")
    monkeypatch.setattr(lerobot, "async_inference", package, raising=False)
    monkeypatch.setitem(sys.modules, "lerobot.async_inference", package)
    monkeypatch.setitem(sys.modules, "lerobot.async_inference.robot_client", module)

    def run_path(path, run_name):
        # The script's import-time captures, its own functions, then its __main__ call.
        seen.update(action=RobotClient.control_loop_action, init=RobotClient.__init__,
                    convert=RobotClient._action_tensor_to_action_dict)
        namespace = seen["namespace"] = {"time": None, "socket": None}
        functions_from(script, ["_return_to_reset_position", "_som_start"], namespace)
        seen["linear"] = namespace["_return_to_reset_position"]
        RobotClient.start = namespace["_som_start"]
        sys.modules["lerobot.async_inference.robot_client"].async_client()

    monkeypatch.setattr(so101.runpy, "run_path", run_path)
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    for name, value in (("POLIMA_SMOOTH_MOVES", "5"), ("POLIMA_SMOOTH_SHAPE", "smoothstep"),
                        ("POLIMA_SUBSTEPS", "2"), ("POLIMA_RESET_EASE", "smoothstep"),
                        ("POLIMA_ACTION_TRACE", "1"), ("POLIMA_TASK_ID", "0"),
                        ("POLIMA_METRICS_DIR", str(tmp_path / "metrics")), ("POLIMA_ARM_P", "24")):
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("POLIMA_AGGREGATE", raising=False)
    from lerobot.robots.so_follower import SOFollower
    lerobot_configure = SOFollower.configure
    monkeypatch.setattr(SOFollower, "configure", lerobot_configure)  # undone after the test
    for name in ("POLIMA_SO101_BUNDLE", "ACT_SOM_DIRECT", "ACT_SOM_STATS", "ACT_REST_POSITION",
                 "ACT_CAMERA_FOCUS_CONFIG", "ACT_PERSPECTIVE_CAMERA", "ACT_WRIST_CAMERA"):
        monkeypatch.setenv(name, "")
    so101.main(["--bundle", str(tmp_path / "bundle"), "--robot-port", "/dev/arm",
                "--front-camera", "/dev/front", "--wrist-camera", "/dev/wrist",
                "--server-address", "127.0.0.1:8093", "--fps", "15"])
    assert seen["action"] is not originals["control_loop_action"]
    assert seen["init"] is not originals["__init__"]
    assert seen["convert"] is not originals["_action_tensor_to_action_dict"]
    # At async_client the trace's return-to-rest hook wraps the eased script function.
    traced = seen["reset"]
    inner = next(c.cell_contents for c in traced.__closure__ if callable(c.cell_contents)
                 and getattr(c.cell_contents, "__name__", "") == "_return_to_reset_position")
    assert getattr(inner, "__polima_reset_ease__", None) == "smoothstep"
    assert inner.__globals__ is seen["namespace"] and inner is not seen["linear"]
    assert module.async_client.__name__ == "<lambda>"  # restored after the run
    config = json.loads((tmp_path / "metrics/config.json").read_text())
    assert (config["smooth_moves"], config["smooth_shape"], config["substeps"], config["reset_ease"],
            config["aggregate_fn_name"]) == (5, "smoothstep", 2, "smoothstep", "latest_only")
    assert config["arm_p"] == 24 and SOFollower.configure is not lerobot_configure


# --- POLIMA_SUBSTEPS: interpolated sends within a tick -----------------------------------

class VirtualTime:
    """Deterministic stand-in for the time module; callbacks fire as time passes."""

    def __init__(self):
        self.now = 1000.0
        self.events = []  # (due, callback)

    def advance(self, seconds):
        self.now += seconds
        for event in [e for e in self.events if e[0] <= self.now]:
            self.events.remove(event)
            event[1]()

    def sleep(self, seconds):
        self.advance(max(0.0, seconds))

    def perf_counter(self):
        return self.now

    monotonic = time = perf_counter


def lerobot_robot_functions(source, namespace):
    """SOFollower.send_action and its clamp, from the same LeRobot tree as `source`."""
    import logging
    from pprint import pformat

    package = Path(source).parents[1]
    namespace.update(logging=logging, pformat=pformat, check_if_not_connected=lambda function: function)
    functions_from(package / "robots/utils.py", ["ensure_safe_goal_position"], namespace)
    return functions_from(package / "robots/so_follower/so_follower.py", ["send_action"], namespace)[0]


def substep_client(tmp_path, monkeypatch, source, smooth, count=2, fps=15):
    """Client class with LeRobot's loop and robot send, the packaged script's wrappers,
    metrics and action trace, installed in launch order; time is virtual."""
    import threading
    from collections.abc import Callable
    from queue import Empty, Queue
    from types import SimpleNamespace

    import torch

    from polima.robot import smoothing, substeps
    from polima.robot.action_trace import ActionTrace
    from polima.robot.metrics import TrialMetrics, instrument_loop
    from polima.util.paths import repo_root

    if source is None or not Path(source).is_file():
        pytest.skip("LeRobot robot_client.py is not available")
    script = repo_root() / "ACT/run_robot_client_with_live_view.py"
    if not script.is_file():
        pytest.skip("packaged SO101 robot-client script is not available (set POLIMA_REPO_ROOT)")
    clock = VirtualTime()
    monkeypatch.setattr(substeps, "time", clock)

    class TimedAction:
        def __init__(self, timestamp, timestep, action):
            self.timestamp, self.timestep, self.action = timestamp, timestep, action

        def get_timestamp(self):
            return self.timestamp

        def get_timestep(self):
            return self.timestep

        def get_action(self):
            return self.action

    lerobot = {"Queue": Queue, "TimedAction": TimedAction, "Callable": Callable, "torch": torch,
               "time": clock, "Any": object}
    (merge, control, convert, loop, ready, available) = functions_from(source, [
        "_aggregate_action_queues", "control_loop_action", "_action_tensor_to_action_dict",
        "control_loop", "_ready_to_send_observation", "actions_available"], lerobot)
    robot_send = lerobot_robot_functions(source, {})
    motors = [j.removesuffix(".pos") for j in JOINTS]

    class Bus:
        def __init__(self):
            self.present = dict.fromkeys(motors, 0.0)
            self.writes = []  # (time, goal)

        def sync_read(self, register, *args, **kwargs):
            clock.advance(0.0015)
            return dict(self.present)

        def sync_write(self, register, goal):
            clock.advance(0.0015)
            self.writes.append((clock.now, dict(goal)))
            self.present = dict(goal)  # an arm that follows instantly

    class Robot:
        action_features = dict.fromkeys(JOINTS, float)
        send_action = robot_send

        def __init__(self):
            self.bus = Bus()
            self.config = SimpleNamespace(max_relative_target=12.0, num_read_retries=1)

    class Client:
        def __init__(self):
            self.robot = Robot()

        @property
        def running(self):
            return not self.shutdown_event.is_set()

        def control_loop_observation(self, task, verbose=False):
            self.observed += 1
            if self.observed >= self.stop_after:
                self.shutdown_event.set()
            return {}

    Client._aggregate_action_queues = merge
    Client.control_loop_action = control
    Client._action_tensor_to_action_dict = convert
    Client.control_loop = loop
    Client._ready_to_send_observation = ready
    Client.actions_available = available
    if smooth:
        smoothing.install(Client, 5, shape="smoothstep", clock=clock.monotonic)
    state = substeps.install(Client, count)
    bundle = {"np": np, "torch": torch, "time": clock, "Empty": Empty, "TimedAction": TimedAction,
              "_return_to_reset_position": lambda *a, **k: True,
              "_original_aggregate_action_queues": Client._aggregate_action_queues,
              "_original_control_loop_action": Client.control_loop_action,
              "_original_action_tensor_to_dict": Client._action_tensor_to_action_dict,
              "_JOINT_NAMES": JOINTS}
    rest_snap, completion, latency_merge, start = functions_from(script, [
        "_action_tensor_to_dict_with_rest_snap", "_control_loop_action_with_completion",
        "_aggregate_action_queues_with_latency", "_som_start"], bundle)
    Client._aggregate_action_queues = latency_merge
    Client.control_loop_action = completion
    Client._action_tensor_to_action_dict = rest_snap
    Client.start = start
    metrics = TrialMetrics(tmp_path / "metrics", {})
    state.record = metrics.record
    restore_metrics = instrument_loop(Client, metrics)
    restore_trace = ActionTrace(metrics).install(Client)
    client = Client()
    client.config = SimpleNamespace(environment_dt=1 / fps, fps=fps)
    client.action_queue, client.action_queue_lock = Queue(), threading.Lock()
    client.latest_action, client.latest_action_lock = -1, threading.Lock()
    client.action_chunk_size, client.action_queue_size = 16, []
    client._chunk_size_threshold = 1.0
    client.start_barrier = SimpleNamespace(wait=lambda: None)
    client.shutdown_event, client.must_go = threading.Event(), threading.Event()
    client.logger = SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None)
    client.som_client = SimpleNamespace(request_id=0)
    client.reset_position = dict.fromkeys(JOINTS, -40.0)
    client.rest_snap_tolerance, client.rest_snap_arm_distance = 5.0, 15.0
    client._rest_snap_armed = client._rest_snap_latched = False
    client._rest_snap_latched_s = None
    client.auto_repeat_enabled, client.auto_repeat_pause_s = True, 2.0
    client.auto_complete_enabled = False
    client.observed, client.stop_after = 0, 0

    def arrive(plan, first):
        client.som_client.request_id += 1
        client._aggregate_action_queues([
            TimedAction(clock.time(), first + i, torch.tensor(plan[first + i], dtype=torch.float32))
            for i in range(len(plan))])

    def close():
        restore_trace()
        restore_metrics()
        metrics.close()

    return SimpleNamespace(client=client, clock=clock, arrive=arrive, metrics=metrics, close=close,
                           bus=client.robot.bus)


@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize("source", lerobot_sources() or [None], ids=lambda p: str(p))
def test_substeps_keep_the_tick_and_its_bookkeeping(tmp_path, monkeypatch, source, smooth):
    rig = substep_client(tmp_path, monkeypatch, source, smooth)
    client, clock, bus = rig.client, rig.clock, rig.bus
    rig.arrive(ramp(0.0, 16, 1.0), 0)
    # A new plan lands while a tick waits between its sends (t0 + 5 ticks + 10 ms).
    t0 = clock.now
    clock.events.append((t0 + 5 / 15 + 0.010, lambda: rig.arrive(ramp(0.0, 16, 1.0, first=0), 0)))
    client.stop_after = 10
    client.control_loop(task="pick")

    ticks = [e["elapsed_s"] for e in rig.metrics.events if e["event"] == "action_started"]
    commands = [e for e in rig.metrics.events if e["event"] == "action_command"]
    assert len(ticks) == len(commands) == 10  # one trace record per tick
    starts = [t for t, _ in bus.writes]
    # LeRobot's period is kept: the tick's work (2 sends, 33 ms apart) ends inside it.
    writes_per_tick = np.diff(np.searchsorted(starts, np.r_[t0 + np.arange(11) / 15]))
    assert writes_per_tick.tolist() == [1] + [2] * 9  # no interpolation before a first send
    finals = [w for k, w in enumerate(bus.writes) if k == 0 or k % 2 == 0]
    final_times = np.array([t for t, _ in finals])
    np.testing.assert_allclose(np.diff(final_times[1:]), 1 / 15, atol=1e-9)
    intermediates = [w for k, w in enumerate(bus.writes) if k % 2 == 1]
    np.testing.assert_allclose(final_times[1:] - np.array([t for t, _ in intermediates]),
                               1 / 30, atol=1e-9)
    # Intermediate = halfway between the previous goal and the tick's goal.
    for (_, mid), (_, before), (_, after) in zip(intermediates, finals, finals[1:]):
        for motor in mid:
            assert mid[motor] == pytest.approx((before[motor] + after[motor]) / 2)
    # Bookkeeping: each timestep once, in order, despite the merge during a wait.
    assert [c["timestep"] for c in commands] == list(range(10))
    assert client.latest_action == 9
    for command, (_, goal) in zip(commands, finals):
        assert command["final_targets"] == {k + ".pos": v for k, v in goal.items()}
    assert sum(e["event"] == "substep_send" for e in rig.metrics.events) == 9
    rig.close()


@pytest.mark.parametrize("source", lerobot_sources() or [None], ids=lambda p: str(p))
def test_substeps_clamp_every_send_and_skip_stale_or_reset(tmp_path, monkeypatch, source):
    rig = substep_client(tmp_path, monkeypatch, source, smooth=False)
    client, clock, bus = rig.client, rig.clock, rig.bus
    rig.arrive(ramp(0.0, 4, 0.0), 0)
    performed = [client.control_loop_action()]
    rig.arrive({1 + i: np.full(6, 40.0) for i in range(8)}, 1)  # 40-degree jump at timestep 1
    performed.append(client.control_loop_action())
    goals = [g for _, g in bus.writes]
    assert len(goals) == 3
    assert goals[1]["elbow_flex"] == 12.0  # intermediate (20) capped at present + 12
    assert goals[2]["elbow_flex"] == 24.0  # final (40) capped at the new present + 12
    assert performed[1]["elbow_flex.pos"] == 24.0  # the script sees the final send's goal
    # A send outside a policy tick (return to rest) goes straight through and ends
    # interpolation; so does a gap longer than STALE_S.
    client.robot.send_action(dict.fromkeys(JOINTS, 24.0))
    client.control_loop_action()
    assert len(bus.writes) == 5
    clock.advance(0.3)
    client.control_loop_action()
    assert len(bus.writes) == 6
    clock.advance(1 / 15)
    client.control_loop_action()
    assert len(bus.writes) == 8
    rig.close()


def test_substeps_setting_validation():
    from polima.robot.substeps import substeps_from_env

    assert substeps_from_env({}) == 1
    assert substeps_from_env({"POLIMA_SUBSTEPS": "2"}) == 2
    for value in ("0", "5", "x", "1.5"):
        with pytest.raises(ValueError, match="POLIMA_SUBSTEPS"):
            substeps_from_env({"POLIMA_SUBSTEPS": value})


# --- POLIMA_RESET_EASE: eased return to rest ----------------------------------------------

def script_reset(clock):
    """The packaged script's own _return_to_reset_position and _som_start."""
    from polima.util.paths import repo_root

    script = repo_root() / "ACT/run_robot_client_with_live_view.py"
    if not script.is_file():
        pytest.skip("packaged SO101 robot-client script is not available (set POLIMA_REPO_ROOT)")
    namespace = {"time": clock, "socket": None}
    functions_from(script, ["_return_to_reset_position", "_som_start"], namespace)
    return namespace


def run_reset(function, start, rest):
    from types import SimpleNamespace

    sent = []

    class Robot:
        def get_observation(self):
            return dict(sent[-1] if sent else start)

        def send_action(self, action):
            sent.append(dict(action))
            return action

    client = SimpleNamespace(reset_completed=False, reset_position=rest, robot=Robot(),
                             shutdown_event=SimpleNamespace(set=lambda: None),
                             logger=SimpleNamespace(info=lambda *a, **k: None,
                                                    warning=lambda *a, **k: None))
    assert function(client, stop_inference=False, force=True)
    return np.array([[s[k] for k in JOINTS] for s in sent])


@pytest.mark.parametrize("ease", ["smoothstep", "smootherstep"])
def test_eased_reset_changes_only_the_segment_ratios(ease):
    from polima.robot import reset_ease

    namespace = script_reset(VirtualTime())
    linear = namespace["_return_to_reset_position"]
    start = dict(zip(JOINTS, [-5.8, 6.9, -14.4, 74.9, -68.7, 29.3]))
    rest = dict(zip(JOINTS, [-6.5, -103.3, 96.5, 76.2, -92.1, 0.6]))
    before, after = run_reset(linear, start, rest), run_reset(reset_ease.eased(linear, ease), start, rest)
    assert before.shape == after.shape == (75 + 250 + 1, 6)  # same rate, durations, final write
    np.testing.assert_allclose(after[[74, 324, 325]], before[[74, 324, 325]])  # same corner/end
    curve = reset_ease.EASES[ease]
    lift = (after[:75, 1] - start["shoulder_lift.pos"]) / (before[74, 1] - start["shoulder_lift.pos"])
    np.testing.assert_allclose(lift, [curve(s / 75) for s in range(1, 76)])
    origin = np.array([[start[k] for k in JOINTS]])
    # Per-joint speed (deg/s) of each 50 Hz command, from standstill at the start.
    speed = np.abs(np.diff(np.r_[origin, after[:325]], axis=0)) * 50
    linear_speed = np.abs(np.diff(np.r_[origin, before[:325]], axis=0)) * 50
    jump = np.abs(np.diff(np.r_[np.zeros((1, 6)), speed], axis=0)).max()
    linear_jump = np.abs(np.diff(np.r_[np.zeros((1, 6)), linear_speed], axis=0)).max()
    assert linear_jump > 20 and jump < 0.1 * linear_jump  # no abrupt start, stop or corner
    assert speed[0].max() < 1 and speed[74].max() < 1


def test_eased_reset_installs_at_async_client_and_refuses_unknown_source():
    from types import SimpleNamespace

    from polima.robot import reset_ease

    namespace = script_reset(VirtualTime())
    original = namespace["_return_to_reset_position"]
    seen = {}
    module = SimpleNamespace(RobotClient=SimpleNamespace(start=namespace["_som_start"]),
                             async_client=lambda: seen.update(reset=namespace["_return_to_reset_position"]))
    restore = reset_ease.install(module, "smoothstep")
    module.async_client()
    assert seen["reset"].__polima_reset_ease__ == "smoothstep"
    assert seen["reset"].__globals__ is namespace  # the script's own globals
    restore()
    namespace["_return_to_reset_position"] = original

    def _return_to_reset_position(self):  # no ratio assignments: not the expected script
        return True

    with pytest.raises(RuntimeError, match="not eased"):
        reset_ease.eased(_return_to_reset_position, "smoothstep")
    for value in ("ease", "cubic"):
        with pytest.raises(ValueError, match="POLIMA_RESET_EASE"):
            reset_ease.ease_from_env({"POLIMA_RESET_EASE": value})
    assert reset_ease.ease_from_env({}) == "linear"


def test_prebuilt_bundle_hydrates_without_the_act_stack(tmp_path, monkeypatch):
    """A customer deploying a prebuilt GR00T bundle has no ACT stack checkout:
    the bundle's own copy of the control loop is kept, byte for byte."""
    from polima.bundle.import_legacy import hydrate_robot_client

    bundle, spec = make_bundle(tmp_path / "bundle")
    client = bundle.root / "robot_client"
    packed = {
        "run_robot_client_with_live_view.py": "loop",
        "preview_robot_cameras.py": "preview",
        "camera_focus_config.json": "{}",
        "calibration/so-arm101.json": "{}",
    }
    for relative, text in packed.items():
        (client / relative).parent.mkdir(parents=True, exist_ok=True)
        (client / relative).write_text(text)
    monkeypatch.setenv("POLIMA_REPO_ROOT", str(tmp_path / "no-stacks"))
    assets = collect_robot_files(spec, bundle.root)
    for relative in packed:
        if relative.startswith("calibration/"):
            assert relative not in assets  # each arm brings its own calibration
        else:
            assert assets[relative] == client / relative
    hydrate_robot_client(bundle, spec)
    for relative, text in packed.items():
        assert (client / relative).read_text() == text
    assert (client / "start.sh").is_file()

    (client / "preview_robot_cameras.py").unlink()
    with pytest.raises(FileNotFoundError, match="preview_robot_cameras.py"):
        collect_robot_files(spec, bundle.root)


def test_a_run_uses_the_bundles_calibration_else_the_devkits_else_refuses(tmp_path):
    from polima.robot.so101 import arm_calibration_dir

    robot_dir, devkit = tmp_path / "bundle/robot_client", tmp_path / "devkit"
    with pytest.raises(FileNotFoundError, match="polima robot calibrate"):
        arm_calibration_dir(robot_dir, devkit)
    write_json(devkit / "so-arm101.json", {})
    assert arm_calibration_dir(robot_dir, devkit) == devkit
    write_json(robot_dir / "calibration/so-arm101.json", {})
    assert arm_calibration_dir(robot_dir, devkit) == robot_dir / "calibration"
