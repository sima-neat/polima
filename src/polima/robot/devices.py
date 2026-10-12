"""Finding the arm and the cameras, and saying so when they are not there.

## Why by-id paths, never /dev/video0

`/dev/videoN` is assigned in enumeration order, so unplugging a camera or
rebooting can swap the two. Both cameras then still open, the policy still runs,
and the arm reaches for the wrong place -- there is no error anywhere. The
`/dev/v4l/by-id/` names are stable per physical device, which is why this
refuses to guess a bare index.

Depends on stdlib only: it runs on the DevKit, in the `lerobot` venv, and on a
host with no hardware at all.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

V4L_BY_ID = Path("/dev/v4l/by-id")
SERIAL_GLOBS = ("ttyACM*", "ttyUSB*")

#: A camera's by-id name ends with the capture interface. Index 0 is the video
#: node; higher indices are metadata streams that open but produce nothing.
VIDEO_NODE = re.compile(r"-video-index0$")


@dataclass(frozen=True)
class Camera:
    path: str          # /dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920-video-index0
    name: str          # usb-046d_HD_Pro_Webcam_C920-video-index0
    node: str = ""     # the /dev/videoN it resolves to, when resolvable

    @property
    def vendor_model(self) -> str:
        """The human-recognizable middle of the by-id name."""
        stem = self.name.removeprefix("usb-")
        stem = VIDEO_NODE.sub("", stem)
        return stem or self.name


@dataclass(frozen=True)
class SerialPort:
    path: str
    name: str


def list_cameras(root: Path | None = None) -> list[Camera]:
    """Every capture node under /dev/v4l/by-id, sorted by name."""
    directory = Path(root) if root is not None else V4L_BY_ID
    if not directory.is_dir():
        return []
    cameras = []
    for entry in sorted(directory.iterdir()):
        if not VIDEO_NODE.search(entry.name):
            continue
        node = ""
        try:
            node = str(entry.resolve())
        except OSError:
            pass
        cameras.append(Camera(path=str(entry), name=entry.name, node=node))
    return cameras


def list_serial_ports(root: Path | None = None) -> list[SerialPort]:
    """Candidate arm ports. The SO-101 follower enumerates as ttyACM*."""
    directory = Path(root) if root is not None else Path("/dev")
    if not directory.is_dir():
        return []
    found: list[SerialPort] = []
    for pattern in SERIAL_GLOBS:
        for entry in sorted(directory.glob(pattern)):
            found.append(SerialPort(path=str(entry), name=entry.name))
    return found


def match_cameras(
    roles: Sequence[tuple[str, str]],
    cameras: Sequence[Camera],
    hints: Mapping[str, str] | None = None,
    overrides: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Assign discovered cameras to policy camera roles.

    Returns (role -> device path, problems). A role is filled by, in order:

      1. an explicit override (`--overhead-camera /dev/v4l/by-id/...`)
      2. a hint matching the by-id name, case-insensitively (`C920`, `Sonix`)
      3. nothing -- and it is reported, not guessed

    The refusal to fall back on ordering is the point. Two cameras assigned the
    wrong way round is the failure with no symptom: everything runs, and the arm
    reaches for the wrong place.
    """
    hints = hints or {}
    overrides = overrides or {}
    assigned: dict[str, str] = {}
    problems: list[str] = []
    taken: set[str] = set()

    for role, _label in roles:
        override = overrides.get(role)
        if override:
            assigned[role] = override
            taken.add(override)
            continue

        hint = hints.get(role, "")
        if not hint:
            problems.append(f"{role}: no camera hint and no --{role}-camera given")
            continue
        matches = [
            camera for camera in cameras
            if hint.lower() in camera.name.lower() and camera.path not in taken
        ]
        if not matches:
            problems.append(f"{role}: no camera matching {hint!r}")
            continue
        if len(matches) > 1:
            problems.append(
                f"{role}: {len(matches)} cameras match {hint!r} "
                f"({', '.join(c.vendor_model for c in matches)}); "
                f"pass --{role}-camera to choose"
            )
            continue
        assigned[role] = matches[0].path
        taken.add(matches[0].path)

    return assigned, problems


def pick_arm_port(ports: Sequence[SerialPort], override: str | None = None) -> tuple[str, str]:
    """(path, problem). Unambiguous only when exactly one port exists."""
    if override:
        return override, ""
    if not ports:
        return "", "no serial port found (expected /dev/ttyACM*); is the arm powered and plugged in?"
    if len(ports) > 1:
        return "", (
            f"{len(ports)} serial ports present ({', '.join(p.name for p in ports)}); "
            "pass --robot-port to choose"
        )
    return ports[0].path, ""
