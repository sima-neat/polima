"""Detect a lost robot camera and bring the arm to rest instead of freezing mid-task."""
from __future__ import annotations

import math
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


def release_and_rest(robot, rest: Mapping[str, float], move_to_rest: Callable[[float], bool],
                     log, open_position: float | None = None) -> bool:
    """Open the gripper, return to rest, then close it. Never raises; True when at rest."""
    try:
        if not rest or GRIPPER not in rest:
            raise ValueError("Camera recovery requires a rest pose including the gripper")
        if open_position is None:
            open_position = float(os.environ.get(ENV_GRIPPER_OPEN, "40"))
        log.warning("Camera recovery 1/3: opening the gripper to release any object")
        move_gripper(robot, open_position)
        log.warning("Camera recovery 2/3: returning to the resting position")
        rested = move_to_rest(open_position)
        if not rested:
            log.error("Camera recovery: the arm did not reach rest")
            return False
        log.warning("Camera recovery 3/3: closing the gripper")
        if GRIPPER in rest:
            move_gripper(robot, float(rest[GRIPPER]))
        deadline = time.monotonic() + 5.0
        while True:
            actual = joint_positions(robot)
            errors = {key: abs(actual[key] - target) for key, target in rest.items()}
            if not all(math.isfinite(error) for error in errors.values()):
                raise RuntimeError(f"Invalid joint readings during recovery: {actual}")
            if max(errors.values()) <= 2.0:
                return True
            if time.monotonic() >= deadline:
                detail = "; ".join(
                    f"{key}: actual={actual[key]:.2f}, target={rest[key]:.2f}, error={error:.2f}"
                    for key, error in errors.items() if error > 2.0)
                raise TimeoutError(f"Rest did not settle within 5s after closing gripper; {detail}")
            robot.send_action(dict(rest))
            time.sleep(.02)
    except Exception as error:  # noqa: BLE001 - the arm may have failed too; never hang here
        log.error("Camera recovery failed: %s", error)
        return False


def wait_for_cameras(cameras: Mapping[str, object], stopped, log, *,
                     timeout_s: float = 60.0, interval_s: float = 2.0,
                     stable_s: float = 1.0) -> bool:
    """Probe disconnected cameras without touching motors; release all handles before return."""
    deadline = time.monotonic() + timeout_s
    while cameras and not stopped.is_set() and time.monotonic() < deadline:
        attempted = []
        ready = False
        try:
            missing = missing_devices({role: _device(camera) for role, camera in cameras.items()})
            if missing:
                raise CameraFault(missing[0], "waiting for USB device")
            for camera in cameras.values():
                if stopped.is_set():
                    return False
                attempted.append(camera)
                camera.connect()
            stable_until = time.monotonic() + stable_s
            while not stopped.is_set() and time.monotonic() < deadline:
                probe(cameras)
                if time.monotonic() >= stable_until:
                    ready = True
                    break
                stopped.wait(min(.1, max(0, deadline - time.monotonic())))
        except Exception as error:  # noqa: BLE001 - camera drivers expose several exception types
            log.warning("Waiting for cameras before retry: %s", error)
        finally:
            cleanup_errors = []
            for camera in reversed(attempted):
                if (getattr(camera, "is_connected", False)
                        or getattr(camera, "videocapture", None) is not None
                        or getattr(camera, "thread", None) is not None):
                    try:
                        camera.disconnect()
                    except Exception as error:  # noqa: BLE001 - finish closing other cameras, then abort
                        cleanup_errors.append(str(error))
            if cleanup_errors:
                raise RuntimeError("Camera retry cleanup failed: " + "; ".join(cleanup_errors))
        if ready and not stopped.is_set():
            return True
        stopped.wait(min(interval_s, max(0, deadline - time.monotonic())))
    return False
