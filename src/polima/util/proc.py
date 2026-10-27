"""The one subprocess runner.

The log is streamed to its final path from the first byte, so an interrupted
run keeps its log.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Result:
    argv: list[str]
    returncode: int
    duration_s: float
    stdout: str = ""
    stderr: str = ""
    cwd: str | None = None
    log_path: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def command(self) -> str:
        return " ".join(shlex.quote(a) for a in self.argv)

    def tail(self, lines: int = 40) -> str:
        combined = (self.stdout + self.stderr).splitlines()
        return "\n".join(combined[-lines:])

    def to_dict(self) -> dict:
        return {
            "argv": self.argv,
            "returncode": self.returncode,
            "duration_s": round(self.duration_s, 3),
            "cwd": self.cwd,
            "log_path": self.log_path,
        }


class CommandFailed(RuntimeError):
    def __init__(self, result: Result):
        self.result = result
        super().__init__(
            f"command failed (exit {result.returncode}): {result.command}\n{result.tail()}"
        )


def run(
    argv: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    log_path: str | Path | None = None,
    echo: bool = True,
    check: bool = False,
    timeout: float | None = None,
    capture: bool = True,
    merge_stderr: bool = True,
    on_line: Callable[[str], None] | None = None,
    dry_run: bool = False,
) -> Result:
    """Run a command, streaming output to stdout and (optionally) a log file.

    stdout and stderr are merged by default so the log matches what a human
    sees. Set ``merge_stderr=False`` when stdout is machine-readable.

    `dry_run` returns a synthetic success without executing, which is what backs
    `polima train --dry-run` and `polima compile --dry-run`.
    """
    argv = [str(a) for a in argv]
    printable = " ".join(shlex.quote(a) for a in argv)

    if dry_run:
        if echo:
            print(f"[dry-run] {printable}")
        return Result(argv, 0, 0.0, cwd=str(cwd) if cwd else None)

    if echo:
        print(f"$ {printable}", flush=True)

    full_env = None
    if env is not None:
        full_env = {**os.environ, **{k: str(v) for k, v in env.items()}}

    log_handle = None
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        log_handle.write(f"$ {printable}\n")

    started = time.monotonic()
    chunks: list[str] = []
    error_chunks: list[str] = []
    process = subprocess.Popen(
        argv,
        cwd=str(cwd) if cwd else None,
        env=full_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    try:
        if merge_stderr:
            assert process.stdout is not None
            for line in process.stdout:
                if echo:
                    print(line, end="", flush=True)
                if log_handle:
                    log_handle.write(line)
                if capture:
                    chunks.append(line)
                if on_line:
                    on_line(line.rstrip("\n"))
                if timeout is not None and time.monotonic() - started > timeout:
                    process.send_signal(signal.SIGTERM)
                    raise subprocess.TimeoutExpired(argv, timeout)
            process.wait()
        else:
            stdout, stderr = process.communicate(timeout=timeout)
            if echo and stdout:
                print(stdout, end="", flush=True)
            if echo and stderr:
                print(stderr, end="", file=sys.stderr, flush=True)
            if log_handle:
                log_handle.write(stdout)
                log_handle.write(stderr)
            if capture:
                chunks.append(stdout)
                error_chunks.append(stderr)
            if on_line:
                for line in (stdout + stderr).splitlines():
                    on_line(line)
    except subprocess.TimeoutExpired:
        process.send_signal(signal.SIGTERM)
        process.wait()
        raise
    except KeyboardInterrupt:
        # Forward the interrupt and let the child clean up, then re-raise so the
        # caller's own handler (e.g. checkpoint retention) still runs.
        process.send_signal(signal.SIGINT)
        process.wait()
        raise
    finally:
        if log_handle:
            log_handle.close()

    result = Result(
        argv=argv,
        returncode=process.returncode,
        duration_s=time.monotonic() - started,
        stdout="".join(chunks),
        stderr="".join(error_chunks),
        cwd=str(cwd) if cwd else None,
        log_path=str(log_path) if log_path else None,
    )
    if check and not result.ok:
        raise CommandFailed(result)
    return result


def capture(argv: Sequence[str], *, cwd: str | Path | None = None,
            env: Mapping[str, str] | None = None, timeout: float | None = 60) -> Result:
    """Quiet variant for probes -- no echo, no log, output captured."""
    return run(argv, cwd=cwd, env=env, echo=False, capture=True, timeout=timeout)


def which(name: str) -> str | None:
    from shutil import which as _which

    return _which(name)


@dataclass
class Pipeline:
    """A recorded sequence of commands, for manifests and deploy reports."""

    records: list[Result] = field(default_factory=list)

    def run(self, argv: Sequence[str], **kwargs) -> Result:
        result = run(argv, **kwargs)
        self.records.append(result)
        return result

    def to_dict(self) -> list[dict]:
        return [r.to_dict() for r in self.records]
