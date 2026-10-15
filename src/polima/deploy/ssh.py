"""SSH and rsync to the DevKit, over one multiplexed connection.

BoardSession opens one ControlMaster socket and runs every step -- mkdir, each
rsync, the build, the launch, the health poll -- through it, rather than paying
a TCP handshake and key exchange per step. A deploy either has a working
connection or fails immediately, rather than dying halfway through with half
the bundle transferred.
"""

from __future__ import annotations

import atexit
import shlex
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

from polima.config.base import BoardConfig
from polima.util.logging import get
from polima.util.proc import Result, run

log = get("deploy.ssh")
_Session = TypeVar("_Session", bound="BoardSession")


class BoardError(RuntimeError):
    pass


class BoardSession:
    """A connection to one board. Use as a context manager."""

    def __init__(self, board: BoardConfig, *, dry_run: bool = False) -> None:
        self.board = board
        self.dry_run = dry_run
        self._control_path: str | None = None
        self._temp_dir: tempfile.TemporaryDirectory | None = None

    # ------------------------------------------------------------ lifecycle

    def __enter__(self: _Session) -> _Session:  # noqa: PYI019
        if self.board.control_master and not self.dry_run:
            self._open_master()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def _open_master(self) -> None:
        # OpenSSH appends a suffix to ControlPath; Unix socket paths are short.
        # Compiler TMPDIR paths can be much longer than that limit. This tiny,
        # private socket directory must not inherit the compiler scratch root.
        self._temp_dir = tempfile.TemporaryDirectory(prefix="polima-ssh-", dir="/tmp")
        self._control_path = str(Path(self._temp_dir.name) / "cm")
        result = subprocess.run(
            [
                "ssh", *self.board.ssh_options,
                "-o", f"ConnectTimeout={int(self.board.connect_timeout_s)}",
                "-o", "ControlMaster=yes",
                "-o", f"ControlPath={self._control_path}",
                "-o", "ControlPersist=120",
                "-fN", self.board.host,
            ],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            self._control_path = None
            raise BoardError(
                f"cannot reach {self.board.host}: "
                f"{result.stderr.strip() or 'ssh failed'}"
            )
        atexit.register(self.close)
        log.debug("ssh ControlMaster open at %s", self._control_path)

    def close(self) -> None:
        if self._control_path:
            subprocess.run(
                ["ssh", "-o", f"ControlPath={self._control_path}", "-O", "exit",
                 self.board.host],
                capture_output=True, check=False,
            )
            self._control_path = None
        if self._temp_dir:
            self._temp_dir.cleanup()
            self._temp_dir = None

    # -------------------------------------------------------------- plumbing

    def _ssh_prefix(self) -> list[str]:
        options = list(self.board.ssh_options)
        if self._control_path:
            options += ["-o", f"ControlPath={self._control_path}"]
        return ["ssh", *options]

    def _rsh(self) -> str:
        """The -e argument for rsync, so it reuses the same master socket."""
        return " ".join(shlex.quote(part) for part in self._ssh_prefix())

    # ---------------------------------------------------------------- public

    def run(
        self, command: str, *, check: bool = True, echo: bool = False, timeout: float | None = None
    ) -> Result:
        return run(
            [*self._ssh_prefix(), self.board.host, command],
            check=check, echo=echo, timeout=timeout, dry_run=self.dry_run,
        )

    def capture(self, command: str) -> str:
        return self.run(command, check=False, echo=False).stdout.strip()

    def exists(self, remote_path: str) -> bool:
        return self.run(
            f"test -e {shlex.quote(remote_path)}", check=False
        ).returncode == 0

    def mkdirs(self, *paths: str) -> None:
        if not paths:
            return
        quoted = " ".join(shlex.quote(p) for p in paths)
        self.run(f"mkdir -p {quoted}")

    def free_bytes(self, path: str) -> int:
        """Free bytes on the filesystem holding `path`.

        Walks up to the nearest existing ancestor, because preflight runs before
        the deploy root is created.
        """
        output = self.capture(
            f"p={shlex.quote(path)}; while [ ! -e \"$p\" ] && [ \"$p\" != / ]; "
            f"do p=$(dirname \"$p\"); done; "
            f"df -B1 --output=avail \"$p\" 2>/dev/null | tail -1"
        )
        try:
            return int(output.split()[0])
        except (ValueError, IndexError):
            return 0

    def rsync(
        self, source: str | Path, destination: str, *, delete: bool = False,
        excludes: tuple[str, ...] = (), echo: bool = True,
    ) -> Result:
        source = str(source)
        # A trailing slash means "contents of", which is what we always want.
        if Path(source).is_dir() and not source.endswith("/"):
            source += "/"
        argv = ["rsync", "-a", "--partial", "-e", self._rsh()]
        if delete:
            argv.append("--delete")
        for pattern in excludes:
            argv += ["--exclude", pattern]
        argv += [source, f"{self.board.host}:{destination}"]
        return run(argv, check=True, echo=echo, dry_run=self.dry_run)

    def push_file(self, source: str | Path, destination: str) -> Result:
        return run(
            ["rsync", "-a", "-e", self._rsh(), str(source),
             f"{self.board.host}:{destination}"],
            check=True, echo=False, dry_run=self.dry_run,
        )

    def fetch(self, remote_path: str, local_path: str | Path) -> Result:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        return run(
            ["rsync", "-a", "-e", self._rsh(),
             f"{self.board.host}:{remote_path}", str(local_path)],
            check=True, echo=False, dry_run=self.dry_run,
        )


@contextmanager
def connect(board: BoardConfig, *, dry_run: bool = False):
    session = BoardSession(board, dry_run=dry_run)
    try:
        with session:
            yield session
    finally:
        session.close()
