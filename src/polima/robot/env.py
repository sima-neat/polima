"""Finding the interpreter that drives the arm, on the SoM.

The robot client runs on the board, next to the policy server -- the host only
compiles and deploys. That is not incidental: the client sends two 640x480
frames per control step, and at 30 Hz over the wire that is ~110 MB/s of raw
observation. Keeping client and server on the same machine makes that a loopback
copy instead of a network.

So the search starts at the board's venv:

    /media/nvme/lerobot   py3.11, lerobot 0.4.4, opencv, no flask

`$POLIMA_ROBOT_PYTHON` overrides it, which is how you drive an arm attached to a
host for bring-up or teleop. Nothing else is searched automatically: silently
finding a host conda env would let `polima robot` appear to work in the one place
the architecture says it should not run.

The report always names the lerobot version, because the board's 0.4.4 and a
host's 0.6.1 are far enough apart that the client's patch layer cannot assume
either.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

#: The board's robot venv -- where the client is meant to run.
BOARD_VENV = Path("/media/nvme/lerobot")

PROBE = (
    "import json, sys, importlib\n"
    "found = {}\n"
    "for name in ('lerobot', 'cv2', 'flask', 'torch'):\n"
    "    try:\n"
    "        module = importlib.import_module(name)\n"
    "        found[name] = getattr(module, '__version__', 'present')\n"
    "    except Exception:\n"
    "        found[name] = ''\n"
    "print(json.dumps({'python': sys.version.split()[0], 'modules': found}))\n"
)


@dataclass(frozen=True)
class RobotEnv:
    python: str
    origin: str                 # how it was found, for the report
    version: str = ""           # python version
    lerobot: str = ""
    cv2: str = ""
    flask: str = ""
    torch: str = ""

    @property
    def usable(self) -> bool:
        """lerobot and opencv are required; flask only backs the browser view."""
        return bool(self.lerobot and self.cv2)

    @property
    def missing(self) -> list[str]:
        return [
            name for name, value in
            (("lerobot", self.lerobot), ("cv2", self.cv2)) if not value
        ]

    def to_dict(self) -> dict:
        return {
            "python": self.python, "origin": self.origin, "version": self.version,
            "lerobot": self.lerobot, "cv2": self.cv2, "flask": self.flask,
            "torch": self.torch, "usable": self.usable,
        }


def probe(python: str | Path, origin: str, timeout: float = 60.0) -> RobotEnv | None:
    """Import-probe an interpreter. Returns None if it cannot even run."""
    try:
        result = subprocess.run([str(python), "-c", PROBE], capture_output=True, check=False,
                                text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        # The probe prints one JSON line; a deprecation warning on stderr is
        # normal (flask 3.1 warns about __version__) and must not fail the read.
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return None
    modules = payload.get("modules", {})
    return RobotEnv(
        python=str(python), origin=origin, version=payload.get("python", ""),
        lerobot=modules.get("lerobot", ""), cv2=modules.get("cv2", ""),
        flask=modules.get("flask", ""), torch=modules.get("torch", ""),
    )


def candidates() -> list[tuple[str, str]]:
    """(python path, origin) in search order, without probing them."""
    found: list[tuple[str, str]] = []
    override = os.environ.get("POLIMA_ROBOT_PYTHON")
    if override:
        found.append((override, "$POLIMA_ROBOT_PYTHON"))

    board = BOARD_VENV / "bin" / "python"
    if board.exists():
        found.append((str(board), f"board venv {BOARD_VENV}"))

    # The running interpreter counts only if it already has the stack -- which
    # on the board's own polima install it will not, by design. Host conda envs
    # are deliberately NOT searched: see the module docstring.
    import sys

    found.append((sys.executable, "the current interpreter"))
    return found


def discover() -> RobotEnv | None:
    """First candidate that can actually drive a robot, else the best near-miss.

    Returning a near-miss rather than None matters for the report: "found
    lerobot but no opencv" is a fixable sentence, and "no environment" is not.
    """
    best: RobotEnv | None = None
    for python, origin in candidates():
        env = probe(python, origin)
        if env is None:
            continue
        if env.usable:
            return env
        if best is None or (env.lerobot and not best.lerobot):
            best = env
    return best


