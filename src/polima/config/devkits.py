"""Saved Modalix DevKits: a name, an ssh address and an optional key, kept in
~/.config/polima/devkits.json ($POLIMA_DEVKITS overrides the file).

`polima workbench` adds, checks and removes them. Every command that takes a
board (`--board`, `POLIMA_BOARD`) also accepts a saved name and then uses the
saved address and key. A DevKit on the network and one cabled straight to this
computer are the same thing here: an address ssh reaches without a password.
(A Workbench started on a DevKit already shows that DevKit through its Studio.)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

from polima.config.base import DEFAULT_BOARD_ROOT

HOST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*@[A-Za-z0-9][A-Za-z0-9._-]*$")  # never an ssh option
NAME = re.compile(r"^[A-Za-z0-9._-]{1,40}$")
PROBE = (f"echo ARCH=$(uname -m); "
         f"echo FREE=$(df -BG --output=avail /media/nvme 2>/dev/null | tail -1 | tr -d ' G'); "
         f"test -x {DEFAULT_BOARD_ROOT}/bin/polima_server && echo POLIMA=yes; "
         f"pgrep -x polima_server >/dev/null && echo SERVER=yes; true")


def path() -> Path:
    config = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return Path(os.environ.get("POLIMA_DEVKITS") or config / "polima/devkits.json")


def load() -> list[dict]:
    try:
        items = json.loads(path().read_text())
    except (OSError, ValueError):
        return []
    return [d for d in items if isinstance(d, dict) and NAME.match(str(d.get("name", "")))
            and HOST.match(str(d.get("host", "")))] if isinstance(items, list) else []


@contextmanager
def _editing():
    """The saved list, locked from read to write so two simultaneous edits both land."""
    target = path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target.with_name(target.name + ".lock"), "w") as lock:
        _lock(lock)
        items = load()
        yield items
        with tempfile.NamedTemporaryFile("w", dir=target.parent, delete=False) as temporary:
            temporary.write(json.dumps(items, indent=2) + "\n")
        os.replace(temporary.name, target)


def _lock(stream) -> None:
    """Exclusive until the file closes: fcntl on macOS/Linux, msvcrt on Windows."""
    try:
        import fcntl
    except ImportError:
        import msvcrt

        stream.write(" ")
        stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
    else:
        fcntl.flock(stream, fcntl.LOCK_EX)


def add(host: str, key: str = "", name: str = "") -> dict:
    """Save (or replace, by name) one DevKit; nothing is contacted here."""
    host, key = host.strip(), key.strip()
    if not HOST.match(host):
        raise ValueError("the address must be user@host, for example sima@192.168.1.20")
    name = name.strip() or host.split("@")[-1]
    if not NAME.match(name):
        raise ValueError("a name may use letters, digits, '.', '_' and '-' only")
    if key and not Path(key).expanduser().is_file():
        raise ValueError(f"there is no ssh key file at {key}")
    entry = {"name": name, "host": host, **({"key": key} if key else {})}
    with _editing() as items:
        items[:] = [d for d in items if d["name"] != name] + [entry]
    return entry


def remove(name: str) -> None:
    with _editing() as items:
        kept = [d for d in items if d["name"] != name]
        if len(kept) == len(items):
            raise LookupError(f"no saved DevKit named {name!r}")
        items[:] = kept


def find(name_or_host: str) -> dict | None:
    return next((d for d in load() if name_or_host in (d["name"], d["host"])), None)


def ssh_options(entry: dict, base: tuple[str, ...]) -> tuple[str, ...]:
    key = entry.get("key")
    return (*base, "-i", str(Path(key).expanduser()), "-o", "IdentitiesOnly=yes") if key else tuple(base)


def probe(entry: dict, base: tuple[str, ...], timeout: int = 4) -> dict:
    """What ssh finds on the DevKit: a fixed, read-only command, never anything the caller sends."""
    argv = ["ssh", *ssh_options(entry, base), "-o", f"ConnectTimeout={timeout}", entry["host"], PROBE]
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout + 6, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"reachable": False, "error": str(error)}
    if result.returncode:
        return {"reachable": False, "error": (result.stderr.strip().splitlines() or ["ssh failed"])[-1]}
    facts = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    free = facts.get("FREE", "")
    return {"reachable": True, "modalix": facts.get("ARCH") == "aarch64",
            "free_gb": int(free) if free.isdigit() else None,
            "polima": "POLIMA" in facts, "server": "SERVER" in facts}


def apply(board) -> None:
    """A board given by saved name (or saved address) takes the saved address and key."""
    entry = find(board.host) if board.host else None
    if entry:
        board.ssh_options = ssh_options(entry, tuple(board.ssh_options))
        board.host = entry["host"]
