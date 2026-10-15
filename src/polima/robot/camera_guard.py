"""Detect a lost robot camera and bring the arm to rest instead of freezing mid-task."""
from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path

ENV_FAILURES = "POLIMA_CAMERA_FAILURES"
ENV_GRIPPER_OPEN = "POLIMA_RECOVERY_GRIPPER_OPEN"
GRIPPER = "gripper.pos"


class CameraFault(RuntimeError):
    def __init__(self, role: str, reason: str):
        super().__init__(f"{role} camera: {reason}")
        self.role = role
        self.reason = reason


def _device(camera) -> str | None:
    config = getattr(camera, "config", None)
    source = getattr(config, "index_or_path", None)
    return str(source) if isinstance(source, (str, Path)) else None


def missing_devices(paths: Mapping[str, object]) -> list[str]:
    """Roles whose configured /dev path is not enumerated."""
    return [role for role, path in paths.items()
            if isinstance(path, (str, Path)) and str(path).startswith("/dev/")
            and not os.path.exists(path)]


def probe(cameras: Mapping[str, object]) -> None:
    """Raise CameraFault for the first camera that is gone or cannot give a frame."""
    for role, camera in cameras.items():
        path = _device(camera)
        if path and path.startswith("/dev/") and not os.path.exists(path):
            raise CameraFault(role, f"{path} is not enumerated")
        if not getattr(camera, "is_connected", True):
            raise CameraFault(role, "not connected")
        try:
            frame = camera.read_latest()
        except Exception as error:  # any read failure is a camera fault
            raise CameraFault(role, f"cannot grab a frame ({error})") from error
        if frame is None or getattr(frame, "size", 0) == 0:
            raise CameraFault(role, "returned an empty frame")


class CameraWatch:
    """Count consecutive faults; trip after `limit` in a row so one dropped frame is tolerated."""

    def __init__(self, limit: int | None = None):
        self.limit = int(os.environ.get(ENV_FAILURES, "3")) if limit is None else limit
        if self.limit < 1:
            raise ValueError(f"{ENV_FAILURES} must be at least 1")
        self.count = 0

    def check(self, cameras: Mapping[str, object]) -> CameraFault | None:
        try:
            probe(cameras)
        except CameraFault as fault:
            self.count += 1
            return fault if self.count >= self.limit else None
        self.count = 0
        return None


def joint_positions(robot) -> dict[str, float]:
    """Read joints from the motor bus only, so a dead camera cannot block the reset."""
    bus = getattr(robot, "bus", None)
    if bus is not None:
        return {f"{motor}.pos": float(value)
                for motor, value in bus.sync_read("Present_Position").items()}
    return {key: float(value) for key, value in robot.get_observation().items()
            if key.endswith(".pos")}


def move_gripper(robot, target: float, duration_s: float = 1.0, fps: int = 50) -> None:
    """Ramp only the gripper to `target`, holding the arm where it is."""
    start = joint_positions(robot)
    steps = max(int(duration_s * fps), 1)
    for step in range(1, steps + 1):
        action = dict(start)
        action[GRIPPER] = start[GRIPPER] + (target - start[GRIPPER]) * step / steps
        robot.send_action(action)
        time.sleep(1 / fps)


def release_and_rest(robot, rest: Mapping[str, float], move_to_rest: Callable[[], bool],
                     log, open_position: float | None = None) -> bool:
    """Open the gripper, return to rest, then close it. Never raises; True when at rest."""
    if open_position is None:
        open_position = float(os.environ.get(ENV_GRIPPER_OPEN, "40"))
    try:
        log.warning("Camera recovery 1/3: opening the gripper to release any object")
        move_gripper(robot, open_position)
        log.warning("Camera recovery 2/3: returning to the resting position")
        rested = move_to_rest()
        if not rested:
            log.error("Camera recovery: the arm did not reach rest")
            return False
        log.warning("Camera recovery 3/3: closing the gripper")
        if GRIPPER in rest:
            move_gripper(robot, float(rest[GRIPPER]))
        return True
    except Exception as error:  # noqa: BLE001 - the arm may have failed too; never hang here
        log.error("Camera recovery failed: %s", error)
        return False
