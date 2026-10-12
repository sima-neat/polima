"""Modalix accelerator recovery.

Loading a model can fail even though the ELF is perfect and the board is
otherwise healthy. The MLA allocates from a reserved CMA pool, and a process
that is SIGKILLed -- which is exactly what `service.stop` escalates to when a
server ignores SIGTERM -- never releases its DMA-coherent buffers. Cycle a few
1.3 GB bundles that way and the pool is free but fragmented::

    CmaTotal:  1830912 kB
    CmaFree:   1031888 kB      <- a gigabyte free ...
    simaai-memory simaai,dms0-manager: dma_alloc_coherent alloc of 67108864 bytes failed
                                      <- ... that cannot serve 64 MB contiguous

The driver reports that as ``MLA_LOAD_FAILED errCode=1001``, which reads like a
corrupt ELF. It is not: the same bundle loaded fine an hour earlier. Only the
`dmesg` line names the real cause, so a deploy that just retries the load, or a
human reading the error, both draw the wrong conclusion.

The ladder below is taken from neat-genai-studio's ``run.sh``
(``reset_mla_dispatcher``), which resets the dispatcher on every launch and
again whenever a load fails. The ordering matters and is not obvious:
``fix_devkit_runtime.sh`` re-inits the MLA memory pool via
``mla_init_modalix.elf`` *before* restarting the services, and it is that step,
not the service restart, that defragments CMA. Restarting
``simaai-appcomplex.service`` alone leaves the pool exactly as fragmented as it
was found.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field

from polima.config.base import BoardConfig
from polima.deploy.ssh import BoardSession
from polima.util.logging import get

log = get("deploy.mla")

#: Substrings that mean "the accelerator is wedged", not "this bundle is bad".
#: The first is the dispatcher's own error name; the second is the message the
#: native server prints around it; the third is the kernel's allocator giving up.
WEDGE_MARKERS = (
    "MLA_LOAD_FAILED",
    "Failed to load model through MLASHM dispatcher",
    "Could not allocate buffer",
)

#: The SDK's official recovery script, preferred over anything hand-rolled.
#: Checked in PATH order, matching run.sh.
RECOVERY_SCRIPTS = (
    "/usr/bin/fix_devkit_runtime.sh",
    "/usr/local/bin/fix_devkit_runtime.sh",
)

#: Fallback when no recovery script is installed.
DISPATCHER_SERVICE = "simaai-appcomplex.service"
INIT_MLA_MEMORY = "/usr/bin/init_mla_memory.sh"

#: Escape hatch: run this instead of the ladder. Mirrors run.sh's MLA_RESET_CMD.
RESET_COMMAND_ENV = "POLIMA_MLA_RESET_CMD"
#: The board's sudo password, if passwordless sudo is not set up. No default
#: is shipped: without it only `sudo -n` is tried.
SUDO_PASSWORD_ENV = "POLIMA_MLA_SUDO_PASSWORD"
SUDO_HINT = (
    f"MLA reset failed; if sudo needed a password, set {SUDO_PASSWORD_ENV} or allow "
    f"passwordless sudo for {RECOVERY_SCRIPTS[0]} and {INIT_MLA_MEMORY} on the board."
)


def looks_wedged(text: str) -> bool:
    """Does this server output describe a wedged accelerator?"""
    return any(marker in text for marker in WEDGE_MARKERS)


@dataclass
class ResetReport:
    ok: bool
    method: str                       # override | recovery-script | services | none
    detail: str = ""
    cma_free_kb_before: int | None = None
    cma_free_kb_after: int | None = None
    steps: list[str] = field(default_factory=list)

    @property
    def reclaimed_kb(self) -> int | None:
        if self.cma_free_kb_before is None or self.cma_free_kb_after is None:
            return None
        return self.cma_free_kb_after - self.cma_free_kb_before

    def to_dict(self) -> dict:
        return {
            "ok": self.ok, "method": self.method, "detail": self.detail,
            "cma_free_kb_before": self.cma_free_kb_before,
            "cma_free_kb_after": self.cma_free_kb_after,
            "reclaimed_kb": self.reclaimed_kb, "steps": self.steps,
        }


def cma_free_kb(session: BoardSession) -> int | None:
    """Free bytes in the MLA's reserved pool, or None if the board has no CMA.

    Reported for the log line only. It is a weak health signal on its own --
    the wedge we hit had a gigabyte free and still could not place 64 MB -- so
    nothing branches on it.
    """
    text = session.capture("grep CmaFree /proc/meminfo 2>/dev/null")
    for token in text.split():
        if token.isdigit():
            return int(token)
    return None


def _sudo(command: str, password: str | None) -> str:
    """`command` under sudo, without ever blocking on a prompt.

    Tries passwordless sudo first so a board with NOPASSWD never sees the
    password at all; with a password, falls back to feeding it on stdin the way
    run.sh's mla_sudo does. `-p ''` suppresses the prompt so it cannot end up
    in a log.
    """
    if not password:
        return f"sudo -n {command}"
    quoted = shlex.quote(password)
    return (
        f"sudo -n {command} 2>/dev/null || "
        f"sudo -S -p '' {command} <<<{quoted}"
    )


def reset(
    session: BoardSession,
    board: BoardConfig,
    *,
    password: str | None = None,
    timeout: float = 180.0,
) -> ResetReport:
    """Reset the accelerator, following neat-genai-studio's ladder.

    Best-effort by design: a failed reset must not mask the load error that
    prompted it, so every rung reports rather than raises.
    """
    password = password or os.environ.get(SUDO_PASSWORD_ENV) or None
    before = cma_free_kb(session)
    steps: list[str] = []

    override = os.environ.get(RESET_COMMAND_ENV)
    if override:
        log.info("resetting MLA via %s: %s", RESET_COMMAND_ENV, override)
        result = session.run(override, check=False, timeout=timeout)
        steps.append(f"{RESET_COMMAND_ENV}: rc={result.returncode}")
        return _finish(session, result.returncode == 0, "override", steps, before)

    for script in RECOVERY_SCRIPTS:
        if session.run(f"test -x {shlex.quote(script)}", check=False).returncode != 0:
            continue
        log.info("resetting MLA runtime via %s", script)
        result = session.run(_sudo(shlex.quote(script), password),
                             check=False, timeout=timeout)
        steps.append(f"{script}: rc={result.returncode}")
        return _finish(session, result.returncode == 0, "recovery-script", steps, before,
                       password)

    # No recovery script: restart the dispatcher and re-init MLA memory. This is
    # strictly weaker -- init_mla_memory.sh alone does not do what the recovery
    # script's ordering does -- so it is the last rung, not the first.
    log.info("no recovery script on the board; restarting %s", DISPATCHER_SERVICE)
    restart = session.run(
        _sudo(f"systemctl restart {shlex.quote(DISPATCHER_SERVICE)}", password),
        check=False, timeout=timeout,
    )
    steps.append(f"systemctl restart {DISPATCHER_SERVICE}: rc={restart.returncode}")
    ok = restart.returncode == 0

    if session.run(f"test -x {shlex.quote(INIT_MLA_MEMORY)}", check=False).returncode == 0:
        init = session.run(_sudo(shlex.quote(INIT_MLA_MEMORY), password),
                           check=False, timeout=timeout)
        steps.append(f"{INIT_MLA_MEMORY}: rc={init.returncode}")
        ok = ok and init.returncode == 0

    return _finish(session, ok, "services", steps, before, password)


def _finish(session: BoardSession, ok: bool, method: str,
            steps: list[str], before: int | None,
            password: str | None = "") -> ResetReport:
    after = cma_free_kb(session)
    report = ResetReport(
        ok=ok, method=method, steps=steps,
        cma_free_kb_before=before, cma_free_kb_after=after,
        detail="; ".join(steps),
    )
    reclaimed = report.reclaimed_kb
    if reclaimed is not None:
        log.info("MLA reset via %s: CmaFree %d -> %d kB (%+d)",
                 method, before, after, reclaimed)
    else:
        log.info("MLA reset via %s: %s", method, "ok" if ok else "FAILED")
    if not ok and password is None:
        log.warning(SUDO_HINT)
    return report
