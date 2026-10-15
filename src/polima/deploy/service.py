"""Start, stop and health-check polima_server on the DevKit.

The server runs under `nohup` with a pid file -- no systemd unit, because the
DevKit is a development target and an operator needs to be able to kill and
restart the server by hand. The health poll is a real socket connect, which can
tell "refused" from "no route" and does not spawn a subshell per probe.
"""

from __future__ import annotations

import shlex
import time
from dataclasses import dataclass, field

from polima.config.base import BoardConfig
from polima.deploy import mla
from polima.deploy.ssh import BoardSession
from polima.util.logging import get
from polima.wire.client import wait_for_port

log = get("deploy.service")


@dataclass
class ServiceStatus:
    running: bool
    pid: int | None
    port: int | None
    bundle: str | None = None
    listening: bool = False
    #: Pid(s) actually bound to the port, which can differ from `pid`.
    bound: list[int] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        """Serving, and served by the process we think we started."""
        return self.listening and self.pid is not None and self.pid in self.bound

    def to_dict(self) -> dict:
        return {
            "running": self.running, "pid": self.pid, "port": self.port,
            "bundle": self.bundle, "listening": self.listening,
            "bound": self.bound, "healthy": self.healthy,
        }


def status(session: BoardSession, board: BoardConfig, port: int | None = None) -> ServiceStatus:
    """What is actually serving, which is not always what the pid file says.

    `bound` is the pid(s) holding the port, resolved independently of the pid
    file. The two can disagree -- a stale server keeps answering while every
    redeploy dies on bind -- so status reports both rather than trusting either.
    """
    pid_text = session.capture(f"cat {shlex.quote(board.path('var/run/server.pid'))} 2>/dev/null")
    pid = int(pid_text) if pid_text.strip().isdigit() else None
    alive = bool(pid) and session.run(f"kill -0 {pid} 2>/dev/null", check=False).returncode == 0
    bound = listening_pids(session, port) if port else []
    return ServiceStatus(running=alive, pid=pid if alive else None, port=port,
                         listening=bool(bound), bound=bound)


def listening_pids(session: BoardSession, port: int) -> list[int]:
    """Pids currently bound to `port`.

    The pid file alone is not enough. If a server is started while a stale one
    still holds the port, the new process dies on bind, the pid file records the
    corpse, and the stale process keeps answering -- so a redeploy silently has
    no effect and every subsequent measurement describes the OLD binary.
    """
    output = session.capture(
        f"ss -ltnpH 'sport = :{port}' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2"
    )
    return sorted({int(line) for line in output.split() if line.isdigit()})


def stop(session: BoardSession, board: BoardConfig, *, port: int | None = None,
         timeout: float | None = None) -> bool:
    """Stop the server. Returns True if anything was stopped.

    Targets both the recorded pid and whatever actually holds the port, because
    those can disagree.
    """
    timeout = board.stop_timeout_s if timeout is None else timeout
    pid_file = board.path("var/run/server.pid")
    targets: list[int] = []

    pid_text = session.capture(f"cat {shlex.quote(pid_file)} 2>/dev/null")
    if pid_text.strip().isdigit():
        pid = int(pid_text)
        # The pid file survives reboots, so its pid may now belong to another program.
        program = session.capture(
            f"tr '\\0' '\\n' < /proc/{pid}/cmdline 2>/dev/null | head -n 1")
        if program.strip().rsplit("/", 1)[-1] == "polima_server":
            targets.append(pid)
        else:
            log.info("ignoring stale server pid file (pid %d is not polima_server)", pid)
    if port:
        for pid in listening_pids(session, port):
            if pid not in targets:
                log.warning("port %d held by unrecorded pid %d; stopping it too", port, pid)
                targets.append(pid)

    alive = [
        pid for pid in targets
        if session.run(f"kill -0 {pid} 2>/dev/null", check=False).returncode == 0
    ]
    if not alive:
        session.run(f"rm -f {shlex.quote(pid_file)}", check=False)
        return False

    for pid in alive:
        session.run(f"kill {pid} 2>/dev/null || true", check=False)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        still = [
            pid for pid in alive
            if session.run(f"kill -0 {pid} 2>/dev/null", check=False).returncode == 0
        ]
        if not still:
            session.run(f"rm -f {shlex.quote(pid_file)}", check=False)
            log.info("stopped server pid(s) %s", ", ".join(str(p) for p in alive))
            return True
        time.sleep(0.5)
        alive = still

    raise RuntimeError(
        f"Policy server pid(s) {', '.join(str(p) for p in alive)} did not stop cleanly "
        f"within {timeout:.0f}s; model switch aborted. Stop robot motion and recover "
        "the accelerator from an authorized terminal."
    )


def start(
    session: BoardSession,
    board: BoardConfig,
    *,
    port: int,
    bundle_path: str | None = None,
    verbose: bool = False,
    health_timeout: float | None = None,
    reset_mla_on_failure: bool = True,
) -> ServiceStatus:
    """Launch polima_server, recovering once from a wedged accelerator.

    A load that fails with MLA_LOAD_FAILED is usually not about the bundle: the
    reserved CMA pool is fragmented, typically by an earlier server that was
    SIGKILLed before it could release its DMA buffers. Retrying the load
    unchanged fails identically forever, so the retry is worth nothing unless
    the accelerator is reset in between -- which is what neat-genai-studio does
    on exactly this error. Anything else propagates untouched on the first try.
    """
    try:
        return _launch(session, board, port=port, bundle_path=bundle_path,
                       verbose=verbose, health_timeout=health_timeout)
    except RuntimeError as error:
        if not reset_mla_on_failure or not mla.looks_wedged(str(error)):
            raise
        log.warning("server start failed on a wedged MLA; resetting the accelerator")
        report = mla.reset(session, board)
        if not report.ok:
            log.error("MLA reset did not succeed (%s); re-raising the load failure",
                      report.detail)
            raise
        reclaimed = report.reclaimed_kb
        log.info("MLA reset via %s%s; retrying the start", report.method,
                 f", reclaimed {reclaimed} kB of CMA" if reclaimed else "")
    return _launch(session, board, port=port, bundle_path=bundle_path,
                   verbose=verbose, health_timeout=health_timeout)


def _launch(
    session: BoardSession,
    board: BoardConfig,
    *,
    port: int,
    bundle_path: str | None = None,
    verbose: bool = False,
    health_timeout: float | None = None,
) -> ServiceStatus:
    """Launch polima_server against `current` (or an explicit bundle)."""
    bundle_path = bundle_path or board.current_link
    binary = board.path("bin", "polima_server")
    log_file = board.path("var/log", "server.log")
    pid_file = board.path("var/run", "server.pid")

    stop(session, board, port=port)

    # `< /dev/null` is what detaches this. Without it ssh waits for the
    # backgrounded process's inherited stdin to close, so the deploy hangs
    # forever even though the server started fine -- and with a ControlMaster
    # socket it holds the whole multiplexed connection open.
    #
    # Deliberately NOT `setsid`: setsid forks and exits, so `$!` would record a
    # pid that is already dead and every subsequent stop/status would miss the
    # real server. nohup alone survives the ssh session closing.
    #
    # The braces matter: `cd X && nohup Y & echo $!` parses as
    # `(cd X && nohup Y) & echo $!`, so $! would be the SUBSHELL's pid -- which
    # exits immediately, leaving every later stop/status unable to find the
    # server. Grouping puts the background job and the echo in the same shell.
    command = (
        f"cd {shlex.quote(board.root)} && "
        "{ "
        + ("POLIMA_LAYOUT_DIAGNOSTICS=1 " if verbose else "")
        + f"nohup {shlex.quote(binary)} "
        f"--bundle {shlex.quote(bundle_path)} --port {port}"
        + (" --verbose" if verbose else "")
        + f" < /dev/null >> {shlex.quote(log_file)} 2>&1 & "
        f"echo $! > {shlex.quote(pid_file)}; "
        "}"
    )
    session.run(command, timeout=30)

    launched_text = session.capture(f"cat {shlex.quote(pid_file)} 2>/dev/null")
    launched = int(launched_text) if launched_text.strip().isdigit() else None

    timeout = health_timeout if health_timeout is not None else board.health_timeout_s
    if not wait_for_port(board.address, port, timeout=timeout):
        tail = session.capture(f"tail -20 {shlex.quote(log_file)}")
        raise RuntimeError(
            f"polima_server did not accept connections on {board.address}:{port} "
            f"within {timeout:.0f}s\n--- server.log ---\n{tail}"
        )

    # A responding port is NOT proof that OUR server started -- a stale process
    # still holding the port answers just as happily while the new binary dies on
    # bind. Require that the process we launched is alive and is the one bound.
    if launched is None:
        raise RuntimeError("server start recorded no pid; cannot verify the launch")
    if session.run(f"kill -0 {launched} 2>/dev/null", check=False).returncode != 0:
        tail = session.capture(f"tail -20 {shlex.quote(log_file)}")
        raise RuntimeError(
            f"polima_server (pid {launched}) exited immediately, yet port {port} "
            f"answers -- something else is holding it.\n--- server.log ---\n{tail}"
        )
    bound = listening_pids(session, port)
    if bound and launched not in bound:
        raise RuntimeError(
            f"port {port} is served by pid(s) {bound}, not the server we started "
            f"({launched}). A stale process is shadowing this deploy."
        )

    result = status(session, board, port)
    result.pid = launched
    result.running = True
    result.listening = True
    log.info("server up on port %d (pid %d, verified bound)", port, launched)
    return result


def logs(session: BoardSession, board: BoardConfig, lines: int = 40) -> str:
    return session.capture(
        f"tail -{lines} {shlex.quote(board.path('var/log', 'server.log'))} 2>/dev/null"
    )
