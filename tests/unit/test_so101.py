from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest

from polima.policies.act import ACT_SPEC
from polima.robot import preview
from polima.robot import so101_loop as so101

REST = {name: 0.0 for name in ACT_SPEC.robot.joint_names}


class FakeArm:
    """Moves instantly to each commanded pose unless `stuck`."""

    def __init__(self, pose: dict[str, float], stuck: bool = False) -> None:
        self.pose, self.stuck, self.sent = dict(pose), stuck, []

    def get_observation(self) -> dict[str, float]:
        return dict(self.pose)

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.sent.append(dict(action))
        if not self.stuck:
            self.pose.update(action)
        return action


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(so101.time, "sleep", lambda _s: None)


def test_rest_snap_arms_only_after_the_arm_leaves_rest():
    snap = so101.RestSnap(REST, repeat=False, pause_s=2)
    near = {**REST, "shoulder_pan.pos": 3.0}
    assert snap.apply(dict(near), 0)[0] == near  # still near start: not a "return"
    snap.apply({**REST, "shoulder_pan.pos": 40.0}, 1)
    assert snap.armed
    held, restart = snap.apply(dict(near), 2)
    assert held == REST and not restart
    assert snap.apply({**REST, "shoulder_pan.pos": 40.0}, 99)[0] == REST  # held without repeat


def test_rest_snap_with_repeat_releases_after_the_pause():
    snap = so101.RestSnap(REST, repeat=True, pause_s=2)
    snap.apply({**REST, "elbow_flex.pos": 30.0}, 0)
    snap.apply(dict(REST), 1)
    assert snap.apply(dict(REST), 2.5) == (REST, False)
    assert snap.apply(dict(REST), 3.0)[1]
    assert not snap.armed and snap.latched_at is None


def test_grasp_cycle_needs_open_close_open():
    cycle = so101.GraspCycle()
    frames = [5.0] * 10 + [25.0] * 8 + [5.0] * 5 + [25.0] * 7
    assert not any(cycle.update(g, t * 0.1) for t, g in enumerate(frames))
    assert cycle.update(25.0, 4.0)  # eighth open frame confirms the release
    assert not cycle.update(25.0, 4.1)


def test_grasp_cycle_ignores_a_release_before_the_minimum_runtime():
    cycle = so101.GraspCycle()
    for gripper in [25.0] * 8 + [5.0] * 5 + [25.0] * 20:
        assert not cycle.update(gripper, 0.5)


def test_return_to_rest_lifts_before_turning():
    arm = FakeArm({**REST, "shoulder_pan.pos": 60.0, "shoulder_lift.pos": 40.0})
    so101.return_to_rest(arm, REST, fps=10)
    lift = arm.sent[14]  # end of the 1.5 s lift at 10 Hz
    assert lift["shoulder_pan.pos"] == 60.0 and lift["shoulder_lift.pos"] == 20.0
    assert arm.pose == REST


def test_return_to_rest_reports_an_arm_that_does_not_arrive():
    arm = FakeArm({**REST, "elbow_flex.pos": 30.0}, stuck=True)
    with pytest.raises(TimeoutError):
        so101.return_to_rest(arm, REST, fps=10, settle_s=0)


class FakeRobotClient:
    """The parts of LeRobot's RobotClient the SO-101 client builds on."""

    def __init__(self, config) -> None:
        self.config, self.robot = config, FakeArm(REST)
        self.shutdown_event, self.must_go = threading.Event(), threading.Event()
        self.action_queue, self.action_queue_lock = Queue(), threading.Lock()
        self.logger, self.stopped = so101.log, False

    def stop(self) -> None:
        self.stopped = True

    def control_loop_action(self, verbose: bool = False) -> dict[str, float]:
        return {**REST, "gripper.pos": self.gripper}

    def _action_tensor_to_action_dict(self, _tensor) -> dict[str, float]:
        return dict(REST)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(so101.SO101Client, "rest", REST)
    monkeypatch.setattr(so101.SO101Client, "repeat_pause_s", 0.0)
    cls = type("Client", (so101.SO101Client, FakeRobotClient), {})
    return cls(SimpleNamespace())


def test_release_with_repeat_returns_to_rest_and_starts_over(client, monkeypatch):
    monkeypatch.setattr(client.cycle, "update", lambda *_: True)
    client.robot.pose["elbow_flex.pos"] = 30.0
    client.action_queue.put("planned past the release")
    client.gripper = 25.0
    client.control_loop_action()
    assert client.robot.pose == REST and client.action_queue.empty()
    assert not client.shutdown_event.is_set() and not client.at_rest


def test_release_without_repeat_stops(client, monkeypatch):
    monkeypatch.setattr(type(client), "auto_repeat", False)
    monkeypatch.setattr(client.cycle, "update", lambda *_: True)
    client.gripper = 25.0
    client.control_loop_action()
    assert client.shutdown_event.is_set()


def test_a_retry_after_a_missed_grasp_starts_a_fresh_cycle(client):
    """The gripper closed on nothing and the arm went home: reopening is a retry."""
    for gripper in [25.0] * 8 + [5.0] * 5:
        client.cycle.update(gripper, 10.0)
    client.snap.apply({**REST, "elbow_flex.pos": 30.0}, 0)  # left rest
    client.snap.apply(dict(REST), 1)  # came back: hold, then restart
    client._action_tensor_to_action_dict(None)
    assert not any(client.cycle.update(25.0, 20.0) for _ in range(8))


def test_stop_returns_to_rest_before_disconnecting(client):
    client.robot.pose["wrist_flex.pos"] = 45.0
    client.stop()
    assert client.robot.pose == REST and client.stopped


def test_lerobot_arguments_match_the_proven_launcher(tmp_path):
    args = SimpleNamespace(overhead_camera="/dev/v4l/a", wrist_camera="/dev/v4l/b", fps=30,
                           server_address="127.0.0.1:8092", robot_port="/dev/ttyACM0",
                           max_relative_target=12, task="", actions_per_chunk=100)
    argv = so101.lerobot_arguments(args, ACT_SPEC, tmp_path)
    assert f"--robot.calibration_dir={tmp_path / 'robot_client/calibration'}" in argv
    assert "--robot.max_relative_target=12" in argv and "--aggregate_fn_name=weighted_average" in argv
    assert "--chunk_size_threshold=0.5" in argv
    cameras = next(item for item in argv if item.startswith("--robot.cameras="))
    assert cameras.count("fourcc: MJPG") == 2 and "'/dev/v4l/a'" in cameras


def test_an_uncalibrated_arm_is_refused_before_anything_moves(tmp_path, capsys):
    client = tmp_path / "robot_client"
    client.mkdir()
    (client / "robot_rest_position.json").write_text(json.dumps({"positions": REST}))
    (tmp_path / "bundle.json").write_text(json.dumps(
        {"bundle_id": "act-x", "policy": "act", "graphs": []}))
    (tmp_path / "plan.json").write_text("{}")
    devices = [tmp_path / name for name in ("arm", "overhead", "wrist")]
    for device in devices:
        device.touch()

    code = so101.main(["--bundle", str(tmp_path), "--robot-port", str(devices[0]),
                       "--overhead-camera", str(devices[1]), "--wrist-camera", str(devices[2])])
    assert code == 1 and "not calibrated" in capsys.readouterr().err


class FakeFeed:
    def frames(self):
        yield b"jpeg-1"


def test_preview_streams_each_camera_behind_its_token():
    server = preview.serve({"overhead": FakeFeed(), "wrist": FakeFeed()}, "127.0.0.1", 0, "tok")
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        # Studio reads <url>/stream/<role>; older bundles called the overhead "perspective".
        for path in ("/tok/stream/overhead", "/tok/stream/perspective", "/tok/stream/wrist"):
            with urllib.request.urlopen(base + path, timeout=5) as response:
                assert b"jpeg-1" in response.read()
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/wrong/stream/wrist", timeout=5)
    finally:
        server.shutdown()


def test_bundle_preview_launcher_accepts_the_board_options():
    """native robot.cpp passes these; the old ACT preview rejected two of them."""
    with pytest.raises(SystemExit) as stop:
        preview.main(["--help"])
    assert stop.value.code == 0
    launcher = Path(so101.__file__).with_name("assets") / "so101/preview_robot_cameras.py"
    assert "polima.robot.preview" in launcher.read_text()


@pytest.fixture
def recorded(client, tmp_path, monkeypatch):
    from polima.robot import evidence

    recorder = evidence.RunRecorder.create(tmp_path, bundle_id="act-x")
    monkeypatch.setattr(so101.SO101Client, "recorder", recorder)
    monkeypatch.setattr(so101.SO101Client, "failure", None)
    monkeypatch.setattr(so101.SO101Client, "stop_requested", False)
    return recorder


def test_the_loop_records_actions_and_the_pick_cycle(client, recorded, monkeypatch):
    import itertools

    from polima.robot import evidence

    frames = itertools.count(0, 0.2)  # one action every 200 ms
    monkeypatch.setattr(so101.time, "monotonic", lambda: next(frames))
    for gripper in [25.0] * 8 + [5.0] * 5 + [25.0] * 8:
        client.gripper = gripper
        client.control_loop_action()
    so101.finish_record(recorded, "completed", None)

    kinds = [e["event"] for e in evidence.read_events(recorded.folder)]
    assert kinds.count("action_sent") == 21
    assert ["gripper_opened", "grasp_detected", "release_detected", "cycle_completed", "cycle_started"] == [
        k for k in kinds if k not in ("run_started", "action_sent", "run_completed")]
    assert evidence.summarize(recorded.folder)["cycles"] == 1


@pytest.mark.parametrize(("flag", "status"), [("stop_requested", "interrupted"), ("failure", "failed")])
def test_how_a_run_ends_decides_its_status(recorded, monkeypatch, flag, status):
    from polima.robot import evidence

    monkeypatch.setattr(so101.SO101Client, flag, True if flag == "stop_requested" else "policy request failed")
    so101.finish_record(recorded, "completed", None)
    assert evidence.summarize(recorded.folder)["status"] == status


def test_a_run_record_says_what_ran_and_with_which_setup(tmp_path, monkeypatch):
    from polima.robot import evidence

    monkeypatch.setenv("POLIMA_ROOT", str(tmp_path / "board"))
    bundle = tmp_path / "act-x"
    (bundle / "robot_client/calibration").mkdir(parents=True)
    calibration = bundle / "robot_client/calibration/so-arm101.json"
    calibration.write_text("{}")
    (bundle / "robot_client/robot_rest_position.json").write_text(json.dumps(
        {"source_dataset": "pick_place", "method": "median episode start", "positions": REST}))
    args = SimpleNamespace(robot_port="/dev/arm", overhead_camera="/dev/top", wrist_camera="/dev/wrist",
                           fps=30, actions_per_chunk=100, max_relative_target=12, server_address="127.0.0.1:8092")
    policy = SimpleNamespace(task_id=0, task="Pick the block")

    recorder = so101.start_record(bundle, ACT_SPEC, policy, args, calibration)
    recorder.finish("completed")

    summary = evidence.list_runs(tmp_path / "board")[0]
    assert summary["bundle_id"] == "act-x" and summary["policy"] == ACT_SPEC.display_name
    assert summary["robot"]["type"] == "SO-101 follower" and summary["cameras"]["wrist"] == "/dev/wrist"
    assert summary["calibration"]["sha256"] and summary["rest_pose"]["source_dataset"] == "pick_place"
    assert summary["settings"]["chunk_threshold"] == 0.5 and summary["task"]["instruction"] == "Pick the block"


# -- one launcher for every SO-101 bundle ---------------------------------------


@pytest.mark.parametrize(("policy", "loop"), [("act", True), ("groot", False), ("smolvla", False)])
def test_so101_launcher_hands_only_act_bundles_to_the_polima_loop(tmp_path, monkeypatch,
                                                                   policy, loop):
    """start.sh always runs polima.robot.so101; ACT goes to so101_loop, GR00T/SmolVLA stay."""
    from polima.robot import so101 as launcher

    (tmp_path / "bundle.json").write_text(json.dumps({"policy": policy}))
    calls = []
    monkeypatch.setattr(so101, "main", lambda argv=None: calls.append(argv) or 7)
    argv = ["--bundle", str(tmp_path), "--robot-port", "/dev/null"]
    if loop:
        assert launcher.main(argv) == 7 and calls == [argv]
    else:
        with pytest.raises(SystemExit):  # its own parser: required cameras are missing
            launcher.main(argv)
        assert calls == []


def test_every_so101_launcher_runs_the_polima_module():
    start = Path(so101.__file__).with_name("assets") / "so101/start.sh"
    assert "-m polima.robot.so101 " in start.read_text()
