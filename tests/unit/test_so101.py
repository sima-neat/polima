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
             "_return_to_reset_position", "_recover_from_camera_fault",
             "_guarded_control_loop_observation", "_stop_after_reset", "_run_with_camera_retries", "_som_receive_actions", "_request_graceful_shutdown"}
    nodes = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    clock = SimpleNamespace(now=0.0, monotonic=lambda: clock.now,
                            sleep=lambda seconds: setattr(clock, "now", clock.now + seconds))
    from polima.robot import camera_guard
    namespace = {"RobotClient": object, "time": clock, "camera_guard": camera_guard,
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


@pytest.mark.parametrize("ease", ["linear", "smoothstep"])
def test_lost_camera_releases_rests_closes_and_stops(monkeypatch, ease):
    from polima.robot import camera_guard

    monkeypatch.setattr(camera_guard.time, "sleep", lambda _s: None)
    functions, _ = control_functions()
    if ease != "linear":
        from polima.robot import reset_ease
        functions["_return_to_reset_position"] = reset_ease.eased(
            functions["_return_to_reset_position"], ease)
    client = virtual_client()
    pose = dict.fromkeys(so101.JOINTS, 30.)
    pose["gripper.pos"] = 5.
    sent = []

    def send(action):
        sent.append(dict(action))
        pose.update(action)

    class Lost:
        is_connected = True

        def read_latest(self):
            raise TimeoutError("latest frame is too old")

    client.robot = SimpleNamespace(
        get_observation=lambda: pytest.fail("recovery must not read cameras"),
        bus=SimpleNamespace(sync_read=lambda _: {k.removesuffix(".pos"): v for k, v in pose.items()}),
        send_action=send, cameras={"wrist": Lost()})
    client.camera_watch = camera_guard.CameraWatch(limit=2)
    client.running = True
    client.action_queue.put("stale")
    functions["_original_control_loop_observation"] = lambda *a: pytest.fail("observed a dead camera")
    observe = functions["_guarded_control_loop_observation"]
    assert observe(client, "task") is None and not client.shutdown_event.is_set()
    assert observe(client, "task") is None
    assert client.shutdown_event.is_set() and client.action_queue.empty()
    opened = next(i for i, action in enumerate(sent) if action["gripper.pos"] == 40.)
    returned = next(i for i in range(opened, len(sent))
                    if all(abs(sent[i][k] - v) < 1e-6 for k, v in client.reset_position.items()
                           if k != "gripper.pos"))
    assert all(a["gripper.pos"] == pytest.approx(40.) for a in sent[opened:returned + 1])
    assert sent[-1]["gripper.pos"] == client.reset_position["gripper.pos"]


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
    images = {"overhead": np.full((480, 640, 3), .25, dtype=np.float32),
              "wrist": np.full((480, 640, 3), .75, dtype=np.float32)}
    client.predict(images["overhead"], images["wrist"], np.zeros(6))
    np.testing.assert_array_equal(captured["image0"], images[roles[0]])
    np.testing.assert_allclose(captured["image1"], (images[roles[1]] - .1) / 2.)


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


def test_failed_camera_recovery_does_not_attempt_another_motion(monkeypatch):
    from polima.robot import camera_guard
    functions, _ = control_functions()
    client = virtual_client()
    client.robot = object()
    monkeypatch.setattr(camera_guard, "release_and_rest", lambda *a: False)
    functions["_recover_from_camera_fault"](client, camera_guard.CameraFault("wrist", "lost"))
    functions["_return_to_reset_position"] = lambda *a: pytest.fail("second recovery motion")
    stops = []
    functions["_original_stop"] = lambda c: stops.append(c)
    functions["_stop_after_reset"](client)
    assert stops == [client]


def test_camera_fault_marks_run_failed(monkeypatch):
    monkeypatch.setattr(so101, "_RUN_STATUS", "completed")
    monkeypatch.setattr(so101, "_RECORDER", None)
    so101.record("camera_fault", camera="wrist", message="read failed")
    assert so101._RUN_STATUS == "failed"


@pytest.mark.parametrize("camera_lost", [True, False])
def test_action_failure_checks_for_camera_loss_without_hiding_motor_error(camera_lost):
    functions, _ = control_functions()
    client = virtual_client()
    client.robot = SimpleNamespace(cameras={})
    error = ConnectionError("motor response missing")
    def fail(*args):
        raise error
    functions["_original_control_loop_action"] = fail
    recovered = []
    functions["_recover_from_camera_fault"] = lambda c, fault: recovered.append(fault.role)
    class Camera:
        is_connected = False
    if camera_lost:
        client.robot.cameras = {"wrist": Camera()}
    with pytest.raises(ConnectionError) as caught:
        functions["_control_loop_action_with_completion"](client)
    assert caught.value is error
    assert recovered == (["wrist"] if camera_lost else [])


@pytest.mark.parametrize("outcome, expected_runs", [
    ("reconnected", 2), ("missing", 1), ("rest_failed", 1),
    ("operator_stop", 1), ("normal", 1), ("repeated_fault", 4),
])
def test_camera_retry_lifecycle(monkeypatch, outcome, expected_runs):
    from polima.robot import camera_guard
    functions, _ = control_functions()
    stopped = threading.Event()
    runs, events, waited = [], [], []
    functions.update(_operator_stopped=stopped, _active_client=None,
                     _frames={"wrist": b"old"}, _frame_condition=threading.Condition())
    functions["record"] = lambda kind, **kw: events.append(kind)

    def run():
        assert functions["_active_client"] is None
        client = virtual_client()
        client.robot = SimpleNamespace(cameras={"wrist": object()})
        client._camera_recovery_attempted = outcome != "normal"
        client.reset_completed = outcome != "rest_failed"
        if runs and outcome != "repeated_fault":
            client._camera_recovery_attempted = False
            assert not functions["_frames"]
        functions["_active_client"] = client
        runs.append(client)

    def wait(cameras, stop, log):
        waited.append(cameras)
        if outcome == "operator_stop":
            stop.set()
        return outcome != "missing"

    functions["_original_stop"] = lambda client: None
    functions["async_client"] = run
    monkeypatch.setattr(camera_guard, "wait_for_cameras", wait)
    functions["_run_with_camera_retries"]()
    assert len(runs) == expected_runs
    assert len(events) >= len(runs) - 1
    if outcome in ("normal", "rest_failed"):
        assert not waited


def test_retry_does_not_swallow_cleanup_or_motor_exception():
    functions, _ = control_functions()
    functions.update(_operator_stopped=threading.Event(), _active_client=None)
    def fail():
        raise RuntimeError("motor or cleanup failed")
    functions["async_client"] = fail
    with pytest.raises(RuntimeError, match="motor or cleanup failed"):
        functions["_run_with_camera_retries"]()


def test_inflight_inference_is_discarded_after_shutdown():
    import numpy as np
    functions, _ = control_functions()
    functions.update(_JOINT_NAMES=so101.JOINTS, np=np)
    class Client:
        shutdown_event = threading.Event()
        @property
        def running(self):
            return not self.shutdown_event.is_set()
    client = Client()
    client.start_barrier = SimpleNamespace(wait=lambda: None)
    client.logger = logging.getLogger("test")
    client.robot = SimpleNamespace(action_features=dict.fromkeys(so101.JOINTS))
    client.som_observations = Queue()
    client.som_observations.put(SimpleNamespace(get_observation=lambda: {
        **dict.fromkeys(so101.JOINTS, 0.), "overhead": None, "wrist": None}))
    def predict(*args):
        client.shutdown_event.set()
        return np.zeros((32, 6)), 233.
    client.som_client = SimpleNamespace(predict=predict)
    client._aggregate_action_queues = lambda *args: pytest.fail("published actions after shutdown")
    functions["_som_receive_actions"](client)


def test_graceful_stop_cancels_retries():
    import signal
    functions, _ = control_functions()
    stop = threading.Event()
    client = virtual_client()
    functions.update(_operator_stopped=stop, _active_client=client, signal=signal)
    functions["_request_graceful_shutdown"](signal.SIGTERM, None)
    assert stop.is_set() and client.shutdown_event.is_set()


def test_shutdown_closes_remaining_resources_after_motor_disconnect_failure():
    functions, _ = control_functions()
    client = virtual_client()
    client._camera_recovery_attempted = True
    calls = []
    def fail_stop(client):
        raise OSError("USB device disappeared")
    def bad_camera():
        calls.append("camera1")
        raise OSError("camera gone")
    client.robot = SimpleNamespace(
        bus=SimpleNamespace(port_handler=SimpleNamespace(closePort=lambda: calls.append("serial"))),
        cameras={"one": SimpleNamespace(is_connected=True, disconnect=bad_camera),
                 "two": SimpleNamespace(is_connected=True, disconnect=lambda: calls.append("camera2"))})
    client.channel = SimpleNamespace(close=lambda: calls.append("channel"))
    functions["_original_stop"] = fail_stop
    functions["_stop_after_reset"](client)
    assert calls == ["serial", "camera1", "camera2", "channel"]
    assert client._cleanup_failed
    assert client._cleanup_done
    functions["_stop_after_reset"](client)
    assert len(calls) == 4


def test_startup_interrupt_still_closes_cameras_and_cancels_retry():
    functions, _ = control_functions()
    stopped = threading.Event()
    client = virtual_client()
    calls = []
    functions.update(_operator_stopped=stopped, _active_client=None)
    functions["_return_to_reset_position"] = lambda c: calls.append("rest")
    functions["_original_stop"] = lambda c: calls.append("disconnect")
    def start():
        functions["_active_client"] = client
        stopped.set()
        raise KeyboardInterrupt
    functions["async_client"] = start
    with pytest.raises(KeyboardInterrupt):
        functions["_run_with_camera_retries"]()
    assert calls == ["rest", "disconnect"]
    assert client._cleanup_done and client.shutdown_event.is_set()


def test_failed_cleanup_never_restarts_camera_recovery():
    functions, _ = control_functions()
    client = virtual_client()
    client._camera_recovery_attempted = True
    client._cleanup_done = True
    client._cleanup_failed = True
    client.reset_completed = True
    functions.update(_operator_stopped=threading.Event(), _active_client=None)
    def run():
        functions["_active_client"] = client
    functions["async_client"] = run
    functions["_run_with_camera_retries"]()


def test_normal_run_does_not_create_detailed_recording(monkeypatch):
    monkeypatch.delenv('POLIMA_RECORD_RUN', raising=False)
    assert so101.start_record(None, None, None, None, None) is None


def test_queue_diagnostics_stay_bounded():
    client = SimpleNamespace(action_queue_size=[])
    so101.bound_queue_history(client)
    for n in range(100000):
        client.action_queue_size.append(n)
    assert len(client.action_queue_size) == 900
    assert client.action_queue_size[-1] == 99999


def test_normal_logging_keeps_errors_and_rotates(tmp_path, monkeypatch):
    from logging.handlers import RotatingFileHandler
    monkeypatch.setenv('POLIMA_ROOT', str(tmp_path))
    logger = logging.Logger('bounded-test', logging.DEBUG)  # noqa: LOG001 - isolated handlers for this test
    old = logging.FileHandler(tmp_path / 'old.log')
    logger.addHandler(old)
    so101.configure_run_logging(logger)
    handler = next(h for h in logger.handlers if isinstance(h, RotatingFileHandler))
    assert handler.maxBytes == 2 * 1024 * 1024 and handler.backupCount == 2
    assert old.stream is None
    logger.info('routine action')
    logger.error('camera fault')
    handler.flush()
    text = Path(handler.baseFilename).read_text()
    assert 'routine action' not in text and 'camera fault' in text
    handler.maxBytes = 100
    for _ in range(100):
        logger.warning('repeated fault with enough text to rotate')
    handler.close()
    assert len(list((tmp_path / 'var/log').glob('robot-errors.log*'))) <= 3
