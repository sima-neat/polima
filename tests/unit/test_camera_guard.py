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

    def move_to_rest():
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
    assert not release_and_rest(robot, {"gripper.pos": 2.0}, lambda: False,
                                logging.getLogger(), open_position=40.0)
    assert robot.sent[-1]["gripper.pos"] == 40.0


def test_recovery_never_raises_when_the_arm_fails():
    robot = Robot()
    robot.bus.sync_read = lambda _r: (_ for _ in ()).throw(OSError("serial bus gone"))
    assert not release_and_rest(robot, {}, lambda: True, logging.getLogger(), open_position=40.0)
