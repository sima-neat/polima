import logging
from types import SimpleNamespace

import numpy as np
import pytest

from polima.robot import camera_guard
from polima.robot.camera_guard import CameraFault, CameraWatch, probe, release_and_rest

FRAME = np.zeros((4, 4, 3), dtype=np.uint8)


class Camera:
    def __init__(self, path="/dev/null", frame=FRAME, error=None, connected=True):
        self.config = SimpleNamespace(index_or_path=path)
        self.frame, self.error, self.is_connected = frame, error, connected

    def read_latest(self):
        if self.error:
            raise self.error
        return self.frame


class Bus:
    def __init__(self, robot):
        self.robot = robot

    def sync_read(self, _register):
        return {key.removesuffix(".pos"): value for key, value in self.robot.position.items()}


class Robot:
    def __init__(self):
        self.position = {"shoulder_pan.pos": 10.0, "gripper.pos": 5.0}
        self.sent = []
        self.bus = Bus(self)

    def get_observation(self):
        raise AssertionError("the reset must not read the cameras")

    def send_action(self, action):
        self.sent.append(dict(action))
        self.position = dict(action)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(camera_guard.time, "sleep", lambda _s: None)


def test_healthy_cameras_pass():
    probe({"overhead": Camera(), "wrist": Camera()})


@pytest.mark.parametrize("camera, reason", [
    (Camera(path="/dev/polima-missing-camera"), "not enumerated"),
    (Camera(connected=False), "not connected"),
    (Camera(error=TimeoutError("latest frame is too old")), "cannot grab a frame"),
    (Camera(frame=None), "empty frame"),
    (Camera(frame=np.zeros((0,), dtype=np.uint8)), "empty frame"),
])
def test_each_camera_failure_names_the_camera(camera, reason):
    with pytest.raises(CameraFault, match=reason) as fault:
        probe({"overhead": Camera(), "wrist": camera})
    assert fault.value.role == "wrist"


def test_missing_devices_ignores_indices():
    assert camera_guard.missing_devices(
        {"overhead": "/dev/polima-missing-camera", "wrist": 0, "front": "/dev/null"}) == ["overhead"]


def test_watch_tolerates_a_dropped_frame_and_trips_on_repeats():
    watch = CameraWatch(limit=3)
    bad = {"wrist": Camera(error=RuntimeError("read thread is not running"))}
    assert watch.check(bad) is None
    assert watch.check({"wrist": Camera()}) is None and watch.count == 0
    assert watch.check(bad) is None and watch.check(bad) is None
    assert isinstance(watch.check(bad), CameraFault)


def test_watch_limit_from_environment(monkeypatch):
    monkeypatch.setenv(camera_guard.ENV_FAILURES, "0")
    with pytest.raises(ValueError, match="at least 1"):
        CameraWatch()


def test_recovery_opens_then_rests_then_closes():
    robot, order = Robot(), []
    rest = {"shoulder_pan.pos": 0.0, "gripper.pos": 2.0}

    def move_to_rest(opened):
        assert opened == 40.0
        order.append(("rest", robot.position["gripper.pos"]))
        robot.position = {**robot.position, "shoulder_pan.pos": 0.0}
        return True

    assert release_and_rest(robot, rest, move_to_rest, logging.getLogger(), open_position=40.0)
    gripper = [action["gripper.pos"] for action in robot.sent]
    opened = gripper.index(40.0)
    assert order == [("rest", 40.0)]
    assert all(action["shoulder_pan.pos"] == 10.0 for action in robot.sent[:opened + 1])
    assert gripper[-1] == 2.0 and robot.sent[-1]["shoulder_pan.pos"] == 0.0


def test_recovery_stops_if_rest_is_not_reached():
    robot = Robot()
    assert not release_and_rest(robot, {"gripper.pos": 2.0}, lambda _: False,
                                logging.getLogger(), open_position=40.0)
    assert robot.sent[-1]["gripper.pos"] == 40.0


def test_recovery_never_raises_when_the_arm_fails():
    robot = Robot()
    robot.bus.sync_read = lambda _r: (_ for _ in ()).throw(OSError("serial bus gone"))
    assert not release_and_rest(robot, {"gripper.pos": 2.0}, lambda _: True, logging.getLogger(), open_position=40.0)


@pytest.mark.parametrize("pose, value", [({}, "40"), ({"gripper.pos": 2.}, "invalid")])
def test_invalid_recovery_settings_do_not_move(pose, value, monkeypatch):
    monkeypatch.setenv(camera_guard.ENV_GRIPPER_OPEN, value)
    robot = Robot()
    assert not release_and_rest(robot, pose, lambda _: pytest.fail("must not return"), logging.getLogger())
    assert robot.sent == []


@pytest.mark.parametrize("healthy", [True, False])
def test_reconnect_checks_frames_and_closes_cameras(healthy):
    from threading import Event
    class ReconnectCamera(Camera):
        def connect(self):
            self.is_connected = True
        def disconnect(self):
            self.is_connected = False
    camera = ReconnectCamera(connected=False, error=None if healthy else TimeoutError("no frames"))
    assert camera_guard.wait_for_cameras(
        {"wrist": camera}, Event(), logging.getLogger(), timeout_s=.03, interval_s=.01,
        stable_s=0) is healthy
    assert not camera.is_connected


def test_reconnect_cancelled_never_opens_camera():
    from threading import Event
    stopped = Event()
    stopped.set()
    assert not camera_guard.wait_for_cameras({"wrist": object()}, stopped, logging.getLogger())


def test_reconnect_cleans_partial_open_and_aborts_if_cleanup_fails():
    from threading import Event
    closed = []
    class Partial(Camera):
        def connect(self):
            self.is_connected = True
            raise RuntimeError("settings failed after opening")
        def disconnect(self):
            closed.append(True)
            raise RuntimeError("cannot close")
    with pytest.raises(RuntimeError, match="cleanup failed"):
        camera_guard.wait_for_cameras({"wrist": Partial(connected=False)}, Event(), logging.getLogger())
    assert closed == [True]


def test_reconnect_cancellation_closes_open_cameras():
    from threading import Event
    stop = Event()
    class Cancel(Camera):
        def connect(self):
            self.is_connected = True
            stop.set()
        def disconnect(self):
            self.is_connected = False
    camera = Cancel(connected=False)
    assert not camera_guard.wait_for_cameras({"wrist": camera}, stop, logging.getLogger())
    assert not camera.is_connected


def test_reconnect_retries_after_transient_open_failure():
    from threading import Event
    class Transient(Camera):
        attempts = 0
        def connect(self):
            self.attempts += 1
            if self.attempts == 1:
                raise ConnectionError("USB not ready")
            self.is_connected = True
        def disconnect(self):
            self.is_connected = False
    camera = Transient(connected=False)
    assert camera_guard.wait_for_cameras({"wrist": camera}, Event(), logging.getLogger(),
                                         timeout_s=1, interval_s=.001, stable_s=.01)
    assert camera.attempts == 2 and not camera.is_connected



@pytest.mark.parametrize("lag_joint", ["gripper.pos", "shoulder_pan.pos"])
def test_recovery_waits_for_mechanical_settling(monkeypatch, lag_joint):
    robot = Robot()
    rest = {"shoulder_pan.pos": 0., "gripper.pos": 0.}
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(camera_guard.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(camera_guard.time, "sleep", lambda s: setattr(clock, "now", clock.now+s))
    closed_at = None
    def send(action):
        nonlocal closed_at
        robot.position = dict(action)
        robot.sent.append(dict(action))
        if action["gripper.pos"] <= .001:
            if closed_at is None:
                closed_at = clock.now
            if clock.now - closed_at < .3:
                robot.position[lag_joint] += 3.
    robot.send_action = send
    def rest_arm(opened):
        robot.position.update(rest)
        robot.position["gripper.pos"] = opened
        return True
    assert release_and_rest(robot, rest, rest_arm, logging.getLogger())
    assert clock.now - closed_at >= .3
    assert robot.position == rest


def test_recovery_settle_failure_reports_joint_error(monkeypatch, caplog):
    robot = Robot()
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(camera_guard.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(camera_guard.time, "sleep", lambda s: setattr(clock, "now", clock.now+s))
    def stuck(action):
        robot.position.update({**action, "gripper.pos": 20.})
    robot.send_action = stuck
    assert not release_and_rest(robot, {"shoulder_pan.pos": 10., "gripper.pos": 0.},
                                lambda opened: True, logging.getLogger())
    assert "gripper.pos" in caplog.text and "20.00" in caplog.text
    assert clock.now <= 7.1
