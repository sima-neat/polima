"""SO-101 client launch and recorded control behavior."""
import ast
import logging
import threading
import urllib.error
import urllib.request
from itertools import pairwise
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest

from polima.robot import preview, so101


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

def test_every_so101_launcher_runs_the_polima_module():
    start = Path(so101.__file__).with_name("assets") / "so101/start.sh"
    assert "-m polima.robot.so101 " in start.read_text()


def control_functions():
    """Load control functions with a virtual arm, without initializing hardware."""
    source = Path(so101.__file__).with_name("assets") / "so101/run_robot_client_with_live_view.py"
    names = {"_begin_task_cycle", "_action_tensor_to_dict_with_rest_snap", "_control_loop_action_with_completion",
             "_return_to_reset_position"}
    nodes = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    clock = SimpleNamespace(now=0.0, monotonic=lambda: clock.now,
                            sleep=lambda seconds: setattr(clock, "now", clock.now + seconds))
    namespace = {"RobotClient": object, "time": clock,
                 "_original_action_tensor_to_dict": lambda _, value: dict(value),
                 "_original_control_loop_action": lambda self, _: dict(self.action),
                 "record": lambda *args, **kwargs: None}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102 - isolated control replay
    return namespace, clock


def virtual_client():
    return SimpleNamespace(
        logger=logging.getLogger("test"), reset_position=dict.fromkeys(so101.JOINTS, 0.),
        rest_snap_tolerance=5., rest_snap_arm_distance=15., _rest_snap_armed=False,
        _rest_snap_latched=False, _rest_snap_latched_s=None, auto_repeat_enabled=True,
        auto_repeat_pause_s=2., action_queue=Queue(), action_queue_lock=threading.Lock(),
        must_go=threading.Event(), shutdown_event=threading.Event(), reset_completed=False,
        auto_complete_enabled=True, _auto_complete_triggered=False,
        _auto_complete_started_s=None, _auto_complete_initial_open_seen=False,
        _auto_complete_grasp_seen=False, _auto_complete_open_count=0, _auto_complete_close_count=0,
        auto_complete_open_threshold=20., auto_complete_close_threshold=10.,
        auto_complete_open_frames=8, auto_complete_close_frames=5, auto_complete_min_runtime_s=2.)


def test_recorded_actions_hold_exact_rest_and_clear_stale_plan():
    functions, clock = control_functions()
    client = virtual_client()
    apply = functions["_action_tensor_to_dict_with_rest_snap"]
    for position, expected in [(0., 0.), (8., 8.), (16., 16.), (6., 6.), (4., 0.), (30., 0.)]:
        action = dict.fromkeys(so101.JOINTS, position)
        assert set(apply(client, action).values()) == {expected}
    assert client._rest_snap_latched
    client.action_queue.put("stale")
    clock.now = 2.
    assert set(apply(client, dict.fromkeys(so101.JOINTS, 30.)).values()) == {0.}
    assert client.action_queue.empty() and client.must_go.is_set()
    assert not client._rest_snap_armed and not client._rest_snap_latched


@pytest.mark.parametrize("enabled", [False, True])
def test_pick_detector_replays_open_close_release_with_original_pause(enabled):
    functions, clock = control_functions()
    client = virtual_client()
    client.auto_complete_enabled = enabled
    returns = []
    functions["_return_to_reset_position"] = lambda *a, **kw: returns.append(kw) or True
    for position in [0.] * 10 + [30.] * 8 + [0.] * 5 + [30.] * 8:
        clock.now += .1
        client.action = {"gripper.pos": position}
        result = functions["_control_loop_action_with_completion"](client)
        assert result == client.action
    assert len(returns) == int(enabled)
    if enabled:
        assert returns == [{"stop_inference": False, "force": True}]
        assert clock.now == pytest.approx(5.1)
        assert not client._auto_complete_triggered and not client._auto_complete_grasp_seen


def test_fast_default_and_slow_preserve_motion_recipes():
    from polima.policies.registry import get_policy
    spec = get_policy("groot")
    assert so101.policy_recipe(spec, {}) == {"fps": 30, "POLIMA_SMOOTH_MOVES": "10",
        "POLIMA_SUBSTEPS": "1", "POLIMA_CHUNK_THRESHOLD": "0.6", "POLIMA_GROOT_NOISE": "zero", "POLIMA_AUTO_COMPLETE": "0"}
    assert so101.policy_recipe(spec, {"POLIMA_FAST": "0"}) == {"fps": 15, "POLIMA_SMOOTH_MOVES": "5",
        "POLIMA_SUBSTEPS": "2", "POLIMA_CHUNK_THRESHOLD": "0.5", "POLIMA_GROOT_NOISE": "zero", "POLIMA_AUTO_COMPLETE": "0"}
    assert so101.policy_recipe(get_policy("act"), {})["POLIMA_CHUNK_THRESHOLD"] == "0.5"


def test_calibration_is_customer_owned(tmp_path):
    client, calibration = tmp_path / "client", tmp_path / "calibration"
    with pytest.raises(FileNotFoundError, match="polima robot calibrate"):
        so101.arm_calibration_dir(client, calibration)
    calibration.mkdir()
    (calibration / "so-arm101.json").write_text("{}")
    assert so101.arm_calibration_dir(client, calibration) == calibration


@pytest.mark.parametrize("ease", ["linear", "smoothstep"])
def test_return_to_rest_preserves_rates_limits_and_final_pose(ease):
    from polima.robot import reset_ease

    functions, clock = control_functions()
    client = virtual_client()
    pose = dict.fromkeys(so101.JOINTS, 30.)
    sent = []

    def send(action):
        sent.append(dict(action))
        pose.update(action)

    client.robot = SimpleNamespace(get_observation=lambda: dict(pose), send_action=send)
    reset = functions["_return_to_reset_position"]
    if ease != "linear":
        reset = reset_ease.eased(reset, ease)
    assert reset(client)
    assert len(sent) == 326 and clock.now == pytest.approx(6.5)
    assert all(action["shoulder_pan.pos"] == pytest.approx(30.) for action in sent[:75])
    assert sent[74]["shoulder_lift.pos"] == 10.
    assert sent[-1] == client.reset_position
    for previous, action in pairwise(sent):
        assert max(abs(action[key] - previous[key]) for key in action) < 12.


def test_telemetry_storage_failure_does_not_change_robot_flow(monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("storage unavailable")

    monkeypatch.setattr(so101, "_RECORDER", SimpleNamespace(event=unavailable))
    monkeypatch.setattr(so101, "_RUN_STATUS", "completed")
    so101.record("inference_completed", model_ms=20.)
    assert so101._RUN_STATUS == "completed"
    so101.record("error", message="inference stopped")
    so101.record("operator_stop")
    assert so101._RUN_STATUS == "failed"


@pytest.mark.parametrize("roles", [("overhead", "wrist"), ("wrist", "overhead")])
def test_act_request_preserves_checkpoint_camera_order(tmp_path, monkeypatch, roles):
    import json

    import numpy as np

    from polima.wire.client import PolimaSOMClient

    (tmp_path / "bundle.json").write_text(json.dumps(
        {"bundle_id": "act-test", "policy": "act", "graphs": []}))
    (tmp_path / "plan.json").write_text(json.dumps({"profile": {
        "calibration_id": "so-arm101",
        "camera_keys": [f"observation.images.{role}" for role in roles],
    }}))
    np.savez(tmp_path / "normalization_stats.npz",
             state_mean=np.zeros(6), state_std=np.ones(6),
             action_mean=np.zeros(6), action_std=np.ones(6),
             camera0_mean=np.zeros(3), camera0_std=np.ones(3),
             camera1_mean=np.full(3, .1), camera1_std=np.full(3, 2.))
    monkeypatch.setenv("POLIMA_SO101_BUNDLE", str(tmp_path))
    monkeypatch.setenv("POLIMA_TASK_ID", "0")
    captured = {}

    def reply(client, tensors, **kwargs):
        captured.update(tensors)
        return np.zeros(client.spec.wire.response_shape, dtype=np.float32), 1.

    monkeypatch.setattr(PolimaSOMClient, "predict_raw", reply)
    client = so101.SO101PolicyClient("127.0.0.1:8093")
    images = {"overhead": np.full((480, 640, 3), 64, dtype=np.uint8),
              "wrist": np.full((480, 640, 3), 191, dtype=np.uint8)}
    client.predict(images["overhead"], images["wrist"], np.zeros(6))
    # Raw frames in checkpoint camera order; the server applies camera<i>'s stats.
    assert captured["image0"].dtype == np.uint8
    np.testing.assert_array_equal(captured["image0"], images[roles[0]])
    np.testing.assert_array_equal(captured["image1"], images[roles[1]])


def test_policy_rest_starts_a_fresh_grasp_cycle():
    functions, clock = control_functions()
    client = virtual_client()
    client._auto_complete_initial_open_seen = True
    client._auto_complete_grasp_seen = True
    client._auto_complete_started_s = 0.
    client._auto_complete_close_count = 5
    client._auto_complete_open_count = 7
    client._rest_snap_armed = True
    apply = functions["_action_tensor_to_dict_with_rest_snap"]
    apply(client, dict.fromkeys(so101.JOINTS, 0.))
    clock.now = 2.
    apply(client, dict.fromkeys(so101.JOINTS, 30.))
    returns = []
    functions["_return_to_reset_position"] = lambda *a, **kw: returns.append(kw) or True
    for _ in range(16):
        clock.now += .1
        client.action = {"gripper.pos": 30.}
        functions["_control_loop_action_with_completion"](client)
    assert returns == [], "Opening for the next pickup must not complete the previous cycle"
    assert not client._auto_complete_grasp_seen
    for position in [0.] * 5 + [30.] * 8:
        clock.now += .1
        client.action = {"gripper.pos": position}
        functions["_control_loop_action_with_completion"](client)
    assert len(returns) == 1


@pytest.mark.parametrize("policy,expected", [("groot", "0"), ("act", "1")])
def test_completion_recipe_default_and_explicit_override(policy, expected):
    spec = SimpleNamespace(name=policy)
    env = {}
    so101.apply_recipe(spec, env)
    assert env.get("POLIMA_AUTO_COMPLETE", "1") == expected
    env = {"POLIMA_AUTO_COMPLETE": "1"}
    so101.apply_recipe(spec, env)
    assert env["POLIMA_AUTO_COMPLETE"] == "1"


@pytest.mark.parametrize("policy,override,expected", [
    ("groot", None, "0"), ("groot", "1", "1"), ("groot", "0", "0"),
    ("act", None, "1"), ("act", "0", "0"),
])
def test_launch_completion_setting(tmp_path, monkeypatch, policy, override, expected):
    import os
    spec = so101.get_policy(policy)
    bundle = SimpleNamespace(bundle_id="test")
    for name in ("POLIMA_AUTO_COMPLETE", "ACT_AUTO_COMPLETE", "POLIMA_NO_AUTO_COMPLETE"):
        monkeypatch.delenv(name, raising=False)
    if override is not None:
        monkeypatch.setenv("POLIMA_AUTO_COMPLETE", override)
    monkeypatch.setattr(so101, "load_contract", lambda *args: (bundle, spec, 0))
    monkeypatch.setattr(so101, "instruction", lambda *args: "place")
    monkeypatch.setattr(so101, "client_configuration", lambda *args: ({}, ["client.py"]))
    monkeypatch.setattr(so101.smoothing, "moves_from_env", lambda: 0)
    monkeypatch.setattr(so101.substeps, "substeps_from_env", lambda: 1)
    monkeypatch.setattr(so101.reset_ease, "ease_from_env", lambda: "linear")
    monkeypatch.setattr(so101, "start_record", lambda *args: None)
    captured = []
    monkeypatch.setattr(so101.runpy, "run_path", lambda *args, **kwargs:
                        captured.append(os.environ.get("ACT_AUTO_COMPLETE", "1")))
    monkeypatch.setattr(so101.sys, "argv", ["polima"])
    monkeypatch.setattr(so101.sys, "path", list(so101.sys.path))
    monkeypatch.setattr(so101.os, "environ", os.environ.copy())
    # apply_recipe's default refers to the process environment.
    original_recipe = so101.apply_recipe
    monkeypatch.setattr(so101, "apply_recipe", lambda spec: original_recipe(spec, so101.os.environ))
    so101.main(["--bundle", str(tmp_path), "--robot-port", "arm", "--front-camera", "front",
                "--wrist-camera", "wrist", "--server-address", "localhost:8093"])
    assert captured == [expected]
