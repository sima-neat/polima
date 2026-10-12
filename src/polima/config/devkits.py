"""Saved Modalix DevKits: a name, an ssh address and an optional key, kept in
~/.config/polima/devkits.json ($POLIMA_DEVKITS overrides the file).

`polima workbench` adds, checks and removes them. Every command that takes a
board (`--board`, `POLIMA_BOARD`) also accepts a saved name and then uses the
saved address and key. A DevKit on the network and one cabled straight to this
computer are the same thing here: an address ssh reaches without a password.
`install_key` sets that up once from the DevKit's password, which is never stored.
(A Workbench started on a DevKit already shows that DevKit through its Studio.)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from polima.config.base import DEFAULT_BOARD_ROOT

HOST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*@[A-Za-z0-9][A-Za-z0-9._-]*$")  # never an ssh option
NAME = re.compile(r"^[A-Za-z0-9._-]{1,40}$")
PROBE = (f"echo ARCH=$(uname -m); "
         f"echo FREE=$(df -BG --output=avail /media/nvme 2>/dev/null | tail -1 | tr -d ' G'); "
         f"test -x {DEFAULT_BOARD_ROOT}/bin/polima_server && echo POLIMA=yes; "
         f"pgrep -x polima_server >/dev/null && echo SERVER=yes; "
         "echo STUDIO=$(systemctl show -p LoadState --value polima-studio.service 2>/dev/null); true")


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


def validate(host: str, key: str = "", name: str = "") -> dict:
    """Validate connection fields without saving or contacting a device."""
    host, key = host.strip(), key.strip()
    if not HOST.match(host):
        raise ValueError("the address must be user@host, for example sima@192.168.1.20")
    name = name.strip() or host.split("@")[-1]
    if not NAME.match(name):
        raise ValueError("a name may use letters, digits, '.', '_' and '-' only")
    if key and not Path(key).expanduser().is_file():
        raise ValueError(f"there is no ssh key file at {key}")
    return {"name": name, "host": host, **({"key": key} if key else {})}


def add(host: str, key: str = "", name: str = "") -> dict:
    """Save (or replace, by name) one DevKit; nothing is contacted here."""
    entry = validate(host, key, name)
    name = entry["name"]
    with _editing() as items:
        items[:] = [d for d in items if d["name"] != name] + [entry]
    return entry


def install_key(host: str, password: str, key: str = "", timeout: int = 15) -> str:
    """Log in once with the DevKit's password and authorise this computer's ssh key there.

    Uses `key`, else a dedicated ~/.ssh/id_ed25519_polima key, creating it if needed. The
    password is passed to ssh for this one login and never stored. Returns the key path.
    """
    if not HOST.match(host.strip()):
        raise ValueError("the address must be user@host, for example sima@192.168.1.20")
    private = Path(key).expanduser() if key else Path.home() / ".ssh/id_ed25519_polima"
    if not private.is_file():
        private.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(private)],
                       check=True, capture_output=True)
    public = Path(f"{private}.pub").read_text().strip()
    if not re.fullmatch(r"ssh-[\w-]+ [A-Za-z0-9+/=]+( [^'\n]*)?", public):
        raise ValueError(f"{private}.pub is not an ssh public key")
    authorise = ("umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; "
                 f"grep -qxF '{public}' ~/.ssh/authorized_keys || echo '{public}' >> ~/.ssh/authorized_keys")
    with tempfile.TemporaryDirectory() as folder:
        env = {**os.environ, "SSH_ASKPASS": str(_askpass(Path(folder))), "SSH_ASKPASS_REQUIRE": "force",
               "DISPLAY": os.environ.get("DISPLAY", ":0"), "POLIMA_DEVKIT_PASSWORD": password}
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", "PubkeyAuthentication=no",
             "-o", "NumberOfPasswordPrompts=1", "-o", f"ConnectTimeout={timeout}", host.strip(), authorise],
            env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout + 30, check=False)
    if result.returncode:
        reason = (result.stderr.strip().splitlines() or ["ssh failed"])[-1]
        raise ValueError("the DevKit did not accept the password" if "denied" in reason.lower() else _hint(reason))
    return str(private)


def _askpass(folder: Path) -> Path:
    """A one-line program ssh runs to read the password from this process's environment."""
    if os.name == "nt":
        script = folder / "askpass.cmd"
        script.write_text(f'@"{sys.executable}" -c "import os; print(os.environ[\'POLIMA_DEVKIT_PASSWORD\'])"\n')
    else:
        script = folder / "askpass"
        script.write_text('#!/bin/sh\nprintf \'%s\\n\' "$POLIMA_DEVKIT_PASSWORD"\n')
        script.chmod(0o700)
    return script


def _hint(reason: str) -> str:
    if "Permission denied" in reason:
        return "SSH authentication failed. Select a key authorized on this DevKit or configure access with its password."
    if "IDENTIFICATION HAS CHANGED" in reason or "verification failed" in reason:
        return f"{reason}. If this DevKit was re-installed, run `ssh-keygen -R <its address>` once."
    return reason


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
        return {"reachable": False, "error": _hint((result.stderr.strip().splitlines() or ["ssh failed"])[-1])}
    facts = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    free = facts.get("FREE", "")
    return {"reachable": True, "modalix": facts.get("ARCH") == "aarch64",
            "free_gb": int(free) if free.isdigit() else None,
            "polima": "POLIMA" in facts, "server": "SERVER" in facts,
            "studio_installed": facts.get("STUDIO") == "loaded"}


def apply(board) -> None:
    """A board given by saved name (or saved address) takes the saved address and key."""
    entry = find(board.host) if board.host else None
    if entry:
        board.ssh_options = ssh_options(entry, tuple(board.ssh_options))
        board.host = entry["host"]
