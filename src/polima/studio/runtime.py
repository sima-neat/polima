from __future__ import annotations

import json
import os
import queue
import re
import secrets
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from polima.config.base import BoardConfig
from polima.deploy import mla

from . import transfer
from .models import RunConfig, StudioState
from .store import StudioStore

_PREVIEW_RE = re.compile(r"(?:Camera preview server|Live camera server):\s+(https?://\S+)")
_CALIBRATION_MISMATCH = re.compile(
    r"mismatch between calibration values in the motor and the calibration file"
    r"|no calibration file found",
    re.IGNORECASE,
)
_BENCHMARK_RE = re.compile(
    r"min (?P<min>[\d.]+) / mean (?P<mean>[\d.]+) / median (?P<median>[\d.]+) "
    r"/ p95 (?P<p95>[\d.]+) / p99 (?P<p99>[\d.]+) / max (?P<max>[\d.]+) ms "
    r"over (?P<iterations>\d+) runs \((?P<warmup>\d+) warm-up\)"
)
_STAGE_RE = re.compile(r"^\s+(?P<stage>[^\s=]+)=(?P<ms>[\d.]+)\s*$", re.MULTILINE)
_CHECK_RE = re.compile(
    r"(?P<status>PASS|FAIL)\s+cosine=(?P<cosine>[-\d.]+)\s+"
    r"mean_abs=(?P<mean_abs>[-\d.]+)\s+max_abs=(?P<max_abs>[-\d.]+)"
)
_START_TASK_RE = re.compile(r"POLIMA_TASK:-(?P<task>[^}\"\n]+)")
_TASK_ARGUMENT_RE = re.compile(r"--task=(?:\"(?P<double>[^\"]+)\"|'(?P<single>[^']+)'|(?P<bare>\S+))")


class Conflict(RuntimeError):
    pass


class _LocalBoard:
    """Runs `mla.reset`'s commands here instead of over ssh.

    The studio already IS the board, so the recovery ladder needs a local
    runner rather than a BoardSession. Same two methods, same contract --
    which is the whole reason `mla.reset` takes a session rather than
    building ssh commands itself.
    """

    def run(self, command: str, *, check: bool = True, echo: bool = False,
            timeout: float | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", "-c", command], capture_output=True,
                              text=True, timeout=timeout, check=False)

    def capture(self, command: str) -> str:
        return self.run(command).stdout.strip()


class StudioRuntime:
    def __init__(self, root: Path, command: Path | None = None) -> None:
        self.root = root
        self.command = command or root / "bin/polima_cli"
        self.logs = root / "var/log/studio"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.store = StudioStore(root / "var/lib/studio.sqlite3")
        self.state = StudioState.IDLE
        self.fault: str | None = None
        self.preview_url: str | None = None
        self._process: subprocess.Popen[str] | None = None
        self._kind: str | None = None
        self._run_id: int | None = None
        self._termination: str | None = None
        self._stop_server_pending = False
        self._lock = threading.RLock()
        self._events: deque[dict[str, Any]] = deque(maxlen=1000)
        self._subscribers: set[queue.Queue[dict[str, Any]]] = set()
        self._lease_token: str | None = None
        self._lease_owner: str | None = None
        self._lease_expires = 0.0
        self._arm_tokens: dict[str, tuple[float, str]] = {}
        self.started_at = time.time()
        self.uploads = transfer.Uploads(root / "var/uploads")
        self.uploads.prune()
        self._job: dict[str, Any] | None = None

    def close(self) -> None:
        with self._lock:
            process = self._process
        if process and process.poll() is None:
            self._termination = "halted"
            self._signal_process(signal.SIGUSR1)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._signal_process(signal.SIGKILL)

    def emit(self, kind: str, **data: Any) -> None:
        event = {"id": secrets.token_hex(6), "time": time.time(), "type": kind, **data}
        with self._lock:
            self._events.append(event)
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                pass

    def events(self) -> Iterator[dict[str, Any]]:
        channel: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=100)
        with self._lock:
            self._subscribers.add(channel)
        try:
            yield {"type": "snapshot", "time": time.time(), "snapshot": self.snapshot()}
            while True:
                try:
                    yield channel.get(timeout=15)
                except queue.Empty:
                    yield {"type": "heartbeat", "time": time.time()}
        finally:
            with self._lock:
                self._subscribers.discard(channel)

    def claim_lease(
        self, previous: str | None = None, owner: str | None = None
    ) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            same_browser = bool(owner and owner == self._lease_owner)
            held = bool(self._lease_token) and self._lease_expires > now
            taking_over = held and previous != self._lease_token and not same_browser

            # Takeover, not refusal. One controller at a time is a safety
            # property -- two tabs driving one arm is genuinely unsafe -- but
            # refusing the second browser outright made a stale tab, a reload
            # in a different window, or a second machine lock everyone out for
            # the rest of the 30s lease with no way to recover but waiting.
            # The newest claimant wins and the previous token stops validating,
            # so control still moves as one baton; it just is not nailed to the
            # first browser that touched it.
            if taking_over or not self._lease_token or (
                previous != self._lease_token and not same_browser
            ):
                self._lease_token = secrets.token_urlsafe(24)
            if owner:
                self._lease_owner = owner
            self._lease_expires = now + 30
            result = {"token": self._lease_token, "expires_at": self._lease_expires}
        # Outside the lock, and announced: the browser that just lost control
        # should be told, not left pressing buttons that quietly 409.
        if taking_over:
            self.emit("lease", event="taken_over", owner=owner or "")
        return result

    def require_lease(self, token: str | None) -> None:
        with self._lock:
            if not token or token != self._lease_token or self._lease_expires <= time.time():
                raise Conflict("a current controller lease is required")

    def arm(self, config: RunConfig) -> dict[str, Any]:
        with self._lock:
            if self.state == StudioState.ARMING:
                self._arm_tokens.clear()
                self.state = StudioState.IDLE
        self._require_idle()
        self._validate_paths(config)
        token = secrets.token_urlsafe(24)
        expires = time.time() + 15
        digest = json.dumps(config.to_json(), sort_keys=True)
        with self._lock:
            self._arm_tokens[token] = (expires, digest)
            self.state = StudioState.ARMING
        self.emit("state", state=self.state.value)
        return {"arming_token": token, "expires_at": expires, "preflight": self.preflight(config)}

    def start_robot(self, config: RunConfig, arming_token: str) -> dict[str, Any]:
        digest = json.dumps(config.to_json(), sort_keys=True)
        with self._lock:
            armed = self._arm_tokens.pop(arming_token, None)
            if not armed or armed[0] < time.time() or armed[1] != digest:
                self.state = StudioState.IDLE
                raise Conflict("arming token expired or does not match this run")
        args = [
            str(self.command), "robot", "run", "--yes", "--bundle", self._bundle_path(config.bundle),
            "--robot-port", config.robot_port, "--overhead-camera", config.overhead_camera,
            "--wrist-camera", config.wrist_camera,
        ]
        autocomplete = "1" if config.autocomplete else "0"
        env = {
            "POLIMA_TASK": config.task,
            "POLIMA_LIVE_PREVIEW": "1" if config.preview else "0",
            "POLIMA_AUTO_REPEAT": "1" if config.repeat else "0",
            "POLIMA_AUTO_COMPLETE": autocomplete,
            # The launcher maps POLIMA_AUTO_REPEAT but hardcodes its own
            # AUTO_COMPLETE as `${<POLICY>_AUTO_COMPLETE:-1}` -- no POLIMA_
            # spelling -- so the toggle was silently ignored and the arm always
            # ran the grasp-release detector. That `:-` does honour an inherited
            # value, so set the policy's own name too. The launchers live in the
            # legacy stacks, which PoLiMa must not edit, so this is the fix that
            # does not fork them. POLIMA_AUTO_COMPLETE above is for whenever
            # they do grow the generic spelling.
            f"{self._policy_of(config.bundle).upper()}_AUTO_COMPLETE": autocomplete,
        }
        if config.fps is not None:
            args += ["--fps", str(config.fps)]
        if config.actions_per_chunk is not None:
            args += ["--actions-per-chunk", str(config.actions_per_chunk)]
        if config.max_relative_target is not None:
            args += ["--max-relative-target", str(config.max_relative_target)]
        log_path = self.logs / f"run-{int(time.time())}.log"
        run_id = self.store.begin_run(config.to_json(), str(log_path))
        self._spawn("robot", args, StudioState.RUNNING, log_path, env, run_id)
        self.store.set("last_run_config", config.to_json())
        return {"run_id": run_id, "state": self.state.value}

    def start_preview(self, bundle: str, overhead: str, wrist: str) -> None:
        self._require_idle()
        if not Path(overhead).exists() or not Path(wrist).exists():
            raise ValueError("both camera devices must be connected")
        args = [
            str(self.command), "robot", "preview", "--bundle", self._bundle_path(bundle),
            "--overhead-camera", overhead, "--wrist-camera", wrist, "--preview-port", "5001",
        ]
        self._spawn("preview", args, StudioState.PREVIEWING, self.logs / "preview.log", {})

    def start_calibration(self, bundle: str, robot_port: str) -> dict[str, Any]:
        self._require_idle()
        bundle_path = Path(self._bundle_path(bundle))
        if not Path(robot_port).exists():
            raise ValueError("follower arm is not connected")
        calibration_dir = self._calibration_path(bundle).parent
        calibration_dir.mkdir(parents=True, exist_ok=True)
        backup = self._backup_calibration(bundle)
        args = [
            str(self.command), "robot", "--policy", self._policy_of(bundle), "calibrate", "--yes",
            "--robot-port", robot_port, "--calibration-dir", str(calibration_dir),
        ]
        self._spawn(
            "calibration", args, StudioState.CALIBRATING,
            self.logs / f"calibration-{int(time.time())}.log", {},
        )
        return {"state": self.state.value, "backup": backup}

    def start_benchmark(self, bundle: str, iterations: int = 50, warmup: int = 3) -> dict[str, Any]:
        self._require_idle()
        if self.server_status()["running"]:
            raise Conflict("stop the policy server before benchmarking")
        if not 1 <= iterations <= 1000:
            raise ValueError("iterations must be between 1 and 1000")
        if not 0 <= warmup <= 100:
            raise ValueError("warmup must be between 0 and 100")
        bundle_path = Path(self._bundle_path(bundle))
        if not (bundle_path / "fixtures/inputs").is_dir():
            raise ValueError("bundle has no fixture inputs for hardware-free benchmarking")
        if not self.command.exists() or not os.access(self.command, os.X_OK):
            raise ValueError("native control binary is unavailable")
        log_path = self.logs / f"benchmark-{int(time.time())}.log"
        benchmark_id = self.store.begin_benchmark(bundle, iterations, warmup, str(log_path))
        args = [
            str(self.command), "--models-dir", str(self.root / "models"),
            "--interactive", "--bundle", bundle, "--exclusive-control",
        ]
        self._spawn(
            "benchmark", args, StudioState.BENCHMARKING, log_path, {},
            benchmark_id=benchmark_id,
        )
        with self._lock:
            process = self._process
        assert process is not None and process.stdin is not None
        process.stdin.write(f"bench {iterations} {warmup}\nstages\ncheck\nquit\n")
        process.stdin.flush()
        self.emit(
            "benchmark", action="started", benchmark_id=benchmark_id,
            bundle=bundle, iterations=iterations, warmup=warmup,
        )
        return {"id": benchmark_id, "state": self.state.value}

    def calibration_input(self, value: str = "") -> None:
        with self._lock:
            process = self._process
            if self._kind != "calibration" or not process or process.poll() is not None:
                raise Conflict("calibration is not running")
            if process.stdin is None:
                raise RuntimeError("calibration input is unavailable")
            process.stdin.write(value + "\n")
            process.stdin.flush()

    def restore_calibration(self) -> dict[str, Any]:
        self._require_idle()
        entry = self.store.get("last_calibration_backup")
        if not entry or not Path(entry["backup"]).exists():
            raise FileNotFoundError("no calibration backup is available")
        shutil.copy2(entry["backup"], entry["source"])
        self.emit("calibration", action="restored", source=entry["backup"])
        return entry

    def stop(self) -> None:
        with self._lock:
            if not self._process or self._process.poll() is not None:
                self.state = StudioState.IDLE
                return
            self.state = StudioState.STOPPING
            self._termination = "stopped"
        self.emit("state", state=self.state.value, reason="operator stop")
        self._signal_process(signal.SIGINT)

    def stop_control(self) -> dict[str, Any]:
        """Stop the active client first, then release the model server.

        The client may still need the server while completing its graceful arm
        reset, so server shutdown must not race the client's SIGINT handler.
        """
        with self._lock:
            process = self._process
            running = bool(process and process.poll() is None)
            if running and self._stop_server_pending:
                return {"state": self.state.value, "server_stopping": True}
            if running:
                self._stop_server_pending = True

        self.stop()
        if running:
            assert process is not None
            threading.Thread(
                target=self._stop_server_after,
                args=(process,),
                daemon=True,
                name="polima-studio-stop-server",
            ).start()
            return {"state": self.state.value, "server_stopping": True}

        if not self.server_status()["running"]:
            return {
                "state": self.state.value,
                "server_stopping": False,
                "output": "model server was not running",
            }
        result = self.server("stop")
        return {"state": self.state.value, "server_stopping": False, **result}

    def _stop_server_after(self, process: subprocess.Popen[str]) -> None:
        process.wait()
        try:
            self.server("stop")
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            self.emit("server", action="stop", error=str(error))
        finally:
            with self._lock:
                self._stop_server_pending = False

    def halt(self) -> None:
        with self._lock:
            running = self._process and self._process.poll() is None
            self.state = StudioState.STOPPING if running else StudioState.IDLE
            if running:
                self._termination = "halted"
        self.emit("halt", reason="operator emergency halt")
        if running:
            self._signal_process(signal.SIGUSR1)

    def server(self, action: str, bundle: str | None = None) -> dict[str, Any]:
        if action not in {"start", "stop"}:
            raise ValueError("server action must be start or stop")
        args = [str(self.command), "server", action]
        if bundle:
            args += ["--bundle", self._bundle_path(bundle)]
        result = subprocess.run(
            args, text=True, capture_output=True, env=self._env(), timeout=45, check=False
        )
        self.emit("server", action=action, returncode=result.returncode, output=result.stdout.strip())
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout).strip())
        return {"output": result.stdout.strip()}

    def activate(self, bundle: str) -> dict[str, Any]:
        result = subprocess.run(
            [str(self.command), "activate", bundle, "--yes"], text=True, capture_output=True,
            env=self._env(), timeout=45, check=False,
        )
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout).strip())
        self.emit("bundle", action="activated", bundle=bundle)
        return {"output": result.stdout.strip()}

    def preflight(self, config: RunConfig) -> list[dict[str, Any]]:
        calibration = self._calibration_path(config.bundle)
        checks = [
            ("bundle", Path(self._bundle_path(config.bundle)).exists()),
            ("follower arm", Path(config.robot_port).exists()),
            ("overhead camera", Path(config.overhead_camera).exists()),
            ("wrist camera", Path(config.wrist_camera).exists()),
            ("follower-arm calibration", calibration.is_file()),
            ("native control binary", self.command.exists() and os.access(self.command, os.X_OK)),
        ]
        return [{"name": name, "ok": ok} for name, ok in checks]

    def _calibration_path(self, bundle: str) -> Path:
        """The per-bundle follower calibration Studio passes to LeRobot.

        A calibration is specific to the arm, so its absence is a hard safety
        failure for robot motion—not a warning a user can accidentally skip.
        """
        return Path(self._bundle_path(bundle)) / "robot_client/calibration/so-arm101.json"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            lease_active = bool(self._lease_token and self._lease_expires > time.time())
            data = {
                "state": self.state.value,
                "fault": self.fault,
                "process": self._kind if process and process.poll() is None else None,
                "preview_url": self.preview_url,
                "lease": {"held": lease_active, "expires_at": self._lease_expires},
                "uptime_s": round(time.time() - self.started_at, 1),
            }
        data["bundles"] = self.bundles()
        data["hardware"] = self.hardware()
        data["server"] = self.server_status()
        data["last_run_config"] = self.store.get("last_run_config", {})
        data["storage"] = self.storage()
        with self._lock:
            data["job"] = dict(self._job) if self._job else None
        benchmarks = self.store.list_benchmarks(limit=1)
        data["latest_benchmark"] = benchmarks[0] if benchmarks else None
        return data

    def bundles(self) -> list[dict[str, Any]]:
        models = self.root / "models"
        active = None
        try:
            active = (self.root / "current").resolve().name
        except OSError:
            pass
        result = []
        if not models.exists():
            return result
        for path in sorted(models.iterdir()):
            if not path.is_dir():
                continue
            metadata: dict[str, Any] = {}
            for filename in ("bundle.json", "robot.json"):
                try:
                    metadata.update(json.loads((path / filename).read_text()))
                except (OSError, ValueError):
                    pass
            tasks = self._bundle_tasks(path, metadata)
            result.append({
                "id": path.name, "active": path.name == active,
                "policy": metadata.get("policy", metadata.get("policy_type", "unknown")),
                "default_task": tasks[0],
                "tasks": tasks,
                "metadata": metadata,
            })
        return result

    @staticmethod
    def _bundle_tasks(path: Path, metadata: dict[str, Any]) -> list[str]:
        """Return the task instructions shipped with a policy bundle.

        Newer bundle manifests may declare ``tasks`` or ``default_task``
        directly.  Existing deployed bundles predate that field but already
        contain the authoritative launcher default as ``POLIMA_TASK``.  Reading
        both formats keeps Studio correct across in-place upgrades.
        """
        candidates: list[Any] = []
        declared = metadata.get("tasks", [])
        declared = declared if isinstance(declared, list) else [declared]
        default = metadata.get("default_task")
        # bundle.json records default_task as an index into tasks; robot.json
        # and older manifests spell the instruction itself.
        if type(default) is int:
            candidates += declared[default:default + 1]
        elif default:
            candidates.append(default)
        if metadata.get("task"):
            candidates.append(metadata["task"])
        candidates.extend(declared)

        if not candidates:
            try:
                launcher = (path / "robot_client/start.sh").read_text(encoding="utf-8")
                match = _START_TASK_RE.search(launcher)
                if match:
                    candidates.append(match.group("task"))
                else:
                    match = _TASK_ARGUMENT_RE.search(launcher)
                    if match:
                        candidates.append(next(value for value in match.groups() if value))
            except OSError:
                pass

        tasks: list[str] = []
        for candidate in candidates:
            task = str(candidate).strip()
            if task and task not in tasks:
                tasks.append(task)
        return tasks or ["Place the object in the basket."]

    def hardware(self) -> dict[str, Any]:
        serial_root = Path("/dev/serial/by-id")
        video_root = Path("/dev/v4l/by-id")
        arms = sorted(str(path) for path in serial_root.glob("*") if "Serial" in path.name)
        if not arms:
            arms = sorted(str(path) for path in Path("/dev").glob("ttyACM*"))
        cameras = sorted(str(path) for path in video_root.glob("*video-index0"))
        return {"arms": arms, "cameras": cameras}

    def server_status(self) -> dict[str, Any]:
        pid_path = self.root / "var/run/server.pid"
        try:
            pid = int(pid_path.read_text().strip())
            os.kill(pid, 0)
            return {"running": True, "pid": pid}
        except (OSError, ValueError):
            return {"running": False, "pid": None}

    def read_log(self, run_id: int | None = None, limit: int = 200_000) -> str:
        if run_id is None:
            path = self.root / "var/log/server.log"
        else:
            rows = [row for row in self.store.list_runs() if row["id"] == run_id]
            if not rows or not rows[0].get("log_path"):
                raise FileNotFoundError("run log not found")
            path = Path(rows[0]["log_path"])
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - limit))
            return handle.read().decode("utf-8", "replace")

    def read_benchmark_log(self, benchmark_id: int, limit: int = 200_000) -> str:
        rows = [row for row in self.store.list_benchmarks() if row["id"] == benchmark_id]
        if not rows or not rows[0].get("log_path"):
            raise FileNotFoundError("benchmark log not found")
        path = Path(rows[0]["log_path"])
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - limit))
            return handle.read().decode("utf-8", "replace")

    def _spawn(
        self, kind: str, args: list[str], state: StudioState, log_path: Path,
        extra_env: dict[str, str], run_id: int | None = None,
        benchmark_id: int | None = None,
    ) -> None:
        with self._lock:
            if self._process and self._process.poll() is None:
                raise Conflict(f"{self._kind} is already running")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            process = subprocess.Popen(
                args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                start_new_session=True, env={**self._env(), **extra_env},
            )
            self._process = process
            self._kind = kind
            self._run_id = run_id
            self._termination = None
            self.state = state
            self.fault = None
            self.preview_url = None
        self.emit("state", state=state.value, process=kind)
        threading.Thread(
            target=self._monitor,
            args=(process, kind, log_path, run_id, benchmark_id), daemon=True,
            name=f"polima-studio-{kind}",
        ).start()

    def _monitor(
        self, process: subprocess.Popen[str], kind: str, log_path: Path, run_id: int | None,
        benchmark_id: int | None,
    ) -> None:
        recent: deque[str] = deque(maxlen=20)
        with log_path.open("a", encoding="utf-8") as log:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                text = line.rstrip()
                if text.strip():
                    recent.append(text.strip())
                found = _PREVIEW_RE.search(text)
                if found:
                    preview_url = found.group(1)
                    with self._lock:
                        self.preview_url = preview_url
                    # The preview server becomes ready asynchronously, after
                    # the start request and its snapshot refresh have usually
                    # completed.  Give browser clients a state-bearing event
                    # instead of making them infer readiness from log text.
                    self.emit("preview", url=preview_url)
                if kind == "robot" and _CALIBRATION_MISMATCH.search(text):
                    # LeRobot reports this after connecting to the follower.
                    # Stop before it can accept a control command, and turn the
                    # otherwise cryptic library warning into an operator action.
                    fault = (
                        "Follower-arm calibration is required. Open Follower calibration, "
                        "then choose Use Existing (Enter) or Manual Calibration (c)."
                    )
                    with self._lock:
                        if self._process is process:
                            self._termination = "calibration_required"
                            self.state = StudioState.FAULT
                            self.fault = fault
                    self.emit("calibration", action="required", fault=fault)
                    self.emit("state", state=StudioState.FAULT.value, fault=fault)
                    self._signal_process(signal.SIGINT)
                self.emit("log", process=kind, line=text)
        code = process.wait()
        with self._lock:
            if self._process is process:
                self._process = None
                self._kind = None
                self.preview_url = None
                expected_stop = self._termination in {"stopped", "halted"}
                calibration_required = self._termination == "calibration_required"
                self.state = (
                    StudioState.FAULT
                    if calibration_required
                    else StudioState.IDLE
                    if expected_stop or code in (0, 130, 143)
                    else StudioState.FAULT
                )
                if self.state == StudioState.IDLE:
                    self.fault = None
                elif not calibration_required:
                    self.fault = process_failure(kind, code, recent)
        if run_id is not None:
            status = self._termination or ("completed" if code == 0 else "failed")
            self.store.finish_run(run_id, status, self.fault)
        if benchmark_id is not None:
            output = log_path.read_text(encoding="utf-8")
            result = parse_benchmark_output(output)
            error = self.fault
            if result is None and error is None:
                error = benchmark_failure(output)
            self.store.finish_benchmark(
                benchmark_id, "completed" if result is not None and code == 0 else "failed",
                result, error,
            )
            self.emit(
                "benchmark", action="completed" if result is not None else "failed",
                benchmark_id=benchmark_id, result=result, error=error,
            )
        self._prune_logs()
        self.emit("state", state=self.state.value, returncode=code, fault=self.fault)

    def _prune_logs(self, max_bytes: int = 500 * 1024 * 1024) -> None:
        files = sorted(
            (path for path in self.logs.glob("*.log") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        total = 0
        for path in files:
            size = path.stat().st_size
            if total + size <= max_bytes:
                total += size
            else:
                try:
                    path.unlink()
                except OSError:
                    pass

    def reset(self) -> dict[str, Any]:
        """Return the board to a clean slate: no child, no server, fresh MLA.

        Deliberately takes no lease. This is the control that gets reached for
        when the studio is already wedged -- a run that will not die, a server
        holding models after a failed load, an MLA whose CMA pool is too
        fragmented to place an ELF -- and a recovery path that can itself be
        blocked by the thing it recovers from is not a recovery path.
        """
        steps: list[str] = []

        with self._lock:
            child = self._process
        if child and child.poll() is None:
            self._termination = "reset"
            self._signal_process(signal.SIGINT)
            deadline = time.time() + 5
            while child.poll() is None and time.time() < deadline:
                time.sleep(0.2)
            if child.poll() is None:
                self._signal_process(signal.SIGKILL)
                steps.append("killed the running child")
            else:
                steps.append("stopped the running child")

        if self.server_status().get("running"):
            self.server("stop")
            steps.append("stopped the policy server")
        else:
            steps.append("no policy server was running")

        report = mla.reset(_LocalBoard(), BoardConfig(root=str(self.root)))
        steps.extend(report.steps)

        with self._lock:
            self.state = StudioState.IDLE
            self._termination = None
            self._arm_tokens.clear()
            # Drop the lease too: whoever recovers the studio should be able to
            # take control immediately rather than wait out a dead tab's 30s.
            self._lease_token = None
            self._lease_owner = None
            self._lease_expires = 0.0
        steps.append("cleared controller lease and arming tokens")

        self.emit("state", state=self.state.value, reason="reset to a clean slate")
        result = {"ok": report.ok, "state": self.state.value, "steps": steps,
                  "mla": report.to_dict()}
        return result

    # ------------------------------------------------------------ transfers

    def storage(self) -> dict[str, Any]:
        usage = shutil.disk_usage(self.root)
        return {"free_bytes": usage.free, "total_bytes": usage.total}

    def create_upload(self, files: list[dict[str, Any]]) -> dict[str, Any]:
        return self.uploads.create(files, self.storage()["free_bytes"])

    def _run_job(self, kind: str, work) -> dict[str, Any]:
        """Run one file job at a time in the background, reporting `job` events.

        Hashing or extracting gigabytes outlasts an HTTP request.
        """
        with self._lock:
            if self._job and self._job["status"] == "running":
                raise Conflict(f"a bundle {self._job['kind']} is still running")
            job = self._job = {"id": secrets.token_hex(6), "kind": kind, "status": "running",
                               "stage": "starting", "done": 0, "total": 0, "result": None,
                               "error": None}
        sent = [0.0]

        def progress(stage: str, done: int, total: int) -> None:
            changed = stage != job["stage"]
            job.update(stage=stage, done=done, total=total)
            if changed or done >= total or time.time() - sent[0] > 0.5:
                sent[0] = time.time()
                self.emit("job", job=dict(job))

        def run() -> None:
            try:
                job.update(status="completed", stage="done", result=work(progress))
            except Exception as error:  # noqa: BLE001 - reported to the operator
                job.update(status="failed", error=str(error))
            self.emit("job", job=dict(job))

        self.emit("job", job=dict(job))
        threading.Thread(target=run, daemon=True, name=f"polima-studio-{kind}").start()
        return dict(job)

    def install_upload(self, upload_id: str, activate: bool = False) -> dict[str, Any]:
        if not self.uploads.status(upload_id)["complete"]:
            raise ValueError("the upload is incomplete")

        def work(progress) -> dict[str, Any]:
            try:
                result = transfer.install_bundle(
                    self.uploads.tree(upload_id, progress), self.root / "models",
                    self.uploads.path(upload_id), progress, self._require_replaceable,
                )
            finally:
                self.uploads.discard(upload_id)
            self.emit("bundle", action="installed", bundle=result["bundle"])
            result["calibration"] = self.calibration_status(result["bundle"])["compared_to_active"]
            if activate:
                result.update(self._activate_when_idle(result["bundle"]))
            return result

        return self._run_job("install", work)

    def _activate_when_idle(self, bundle: str) -> dict[str, Any]:
        if self.state not in {StudioState.IDLE, StudioState.FAULT} or (
            self.server_status()["running"]
        ):
            return {"activation_skipped": "Studio is busy or the policy server is running"}
        try:
            self.activate(bundle)
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            return {"activation_error": str(error)}
        return {"activated": True}

    def verify_bundle(self, bundle_id: str) -> dict[str, Any]:
        path = self._model_dir(bundle_id)
        manifest = transfer.read_manifest(path)

        def work(progress) -> dict[str, Any]:
            problems = transfer.verify_elfs(path, manifest, progress) + transfer.archives_in(path)
            return {"bundle": bundle_id, "ok": not problems, "problems": problems,
                    "graphs": len(manifest["graphs"])}

        return self._run_job("verify", work)

    def delete_bundle(self, bundle_id: str) -> dict[str, Any]:
        """Delete an inactive PoLiMa bundle. Hand-built deploy roots are never deleted."""
        path = self._model_dir(bundle_id)
        if self._active_bundle() == bundle_id:
            raise Conflict("activate another bundle before deleting the active one")
        if _manifest(path).get("format") != "polima-bundle-v1":
            raise ValueError("only PoLiMa bundles can be deleted from Studio")
        self._require_replaceable(bundle_id)
        backup = self._backup_calibration(bundle_id)
        shutil.rmtree(path)
        self.emit("bundle", action="deleted", bundle=bundle_id)
        return {"deleted": bundle_id, "calibration_backup": backup}

    def bundle_details(self, bundle_id: str) -> dict[str, Any]:
        path = self._model_dir(bundle_id)
        manifest = _manifest(path)
        size, files = transfer.tree_size(path)
        return {
            "id": bundle_id, "size_bytes": size, "files": files, "modified": path.stat().st_mtime,
            "managed": manifest.get("format") == "polima-bundle-v1",
            "checkpoint": manifest.get("checkpoint"),
            "graphs": [{key: graph.get(key) for key in ("name", "precision", "elf_bytes")}
                       for graph in manifest.get("graphs", [])],
            "robot_client": (path / "robot_client/start.sh").is_file(),
            "calibration": self.calibration_status(bundle_id),
        }

    def calibration_status(self, bundle_id: str) -> dict[str, Any]:
        """This bundle's SO-ARM101 calibration, compared by value with the active bundle's."""
        active, robot = self._active_bundle(), self._calibration_id(bundle_id)
        if robot not in ("", "so-arm101"):
            return {"applicable": False, "present": False, "robot": robot,
                    "active_bundle": active, "compared_to_active": "not-applicable"}
        own = self._load_calibration(bundle_id)
        theirs = self._load_calibration(active) if active and active != bundle_id else None
        compared = (
            "is-active" if active == bundle_id else "no-active" if not active
            else "missing" if own is None else "no-active-calibration" if theirs is None
            else "same" if own == theirs else "different"
        )
        return {"applicable": True, "present": own is not None, "robot": "so-arm101",
                "active_bundle": active, "compared_to_active": compared}

    def read_calibration(self, bundle_id: str) -> Path:
        path = self._calibration_path(bundle_id)
        if not path.is_file():
            raise FileNotFoundError(f"{bundle_id} has no follower-arm calibration")
        return path

    def set_calibration(
        self, bundle_id: str, calibration: Any = None, from_bundle: str | None = None
    ) -> dict[str, Any]:
        """Replace a bundle's SO-ARM101 calibration, keeping a restorable backup."""
        self._require_idle()
        if self._calibration_id(bundle_id) not in ("", "so-arm101"):
            raise ValueError("Studio manages SO-ARM101 calibration only")
        if from_bundle:
            calibration = json.loads(self.read_calibration(from_bundle).read_text())
        transfer.validate_calibration(calibration, self._load_calibration(bundle_id))
        backup = self._backup_calibration(bundle_id)
        target = self._calibration_path(bundle_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(calibration, indent=2) + "\n")
        self.emit("calibration", action="updated", bundle=bundle_id, backup=backup)
        return {"bundle": bundle_id, "backup": backup}

    def _load_calibration(self, bundle: str) -> Any:
        try:
            return json.loads(self._calibration_path(bundle).read_text())
        except (OSError, ValueError):
            return None

    def _backup_calibration(self, bundle: str) -> str | None:
        current = self._calibration_path(bundle)
        if not current.is_file():
            return None
        backup = self.root / "var/lib/calibration-backups" / (
            f"{bundle}-{int(time.time())}-so-arm101.json"
        )
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current, backup)
        self.store.set("last_calibration_backup", {"source": str(current), "backup": str(backup)})
        return str(backup)

    def _calibration_id(self, bundle: str) -> str:
        """The robot plan.json says a bundle drives ('' when it does not say)."""
        plan = _manifest(self.root / "models" / bundle, "plan.json")
        blocks = (plan.get(key) for key in ("robot", "profile"))
        return next((str(block["calibration_id"]) for block in blocks
                     if isinstance(block, dict) and block.get("calibration_id")), "")

    def _model_dir(self, bundle_id: str) -> Path:
        path = self.root / "models" / str(bundle_id)
        if not transfer.SAFE_ID.match(str(bundle_id)) or path.is_symlink() or not path.is_dir():
            raise FileNotFoundError(f"bundle {bundle_id} is not installed")
        return path

    def _active_bundle(self) -> str | None:
        current = self.root / "current"
        return current.resolve().name if current.exists() else None

    def _require_replaceable(self, bundle_id: str) -> None:
        with self._lock:
            kind = self._kind if self._process and self._process.poll() is None else None
        if kind:
            raise Conflict(f"stop the {kind} before replacing or deleting a bundle")
        if self._active_bundle() == bundle_id and self.server_status()["running"]:
            raise Conflict(f"{bundle_id} is being served; stop the policy server first")

    def _signal_process(self, sig: signal.Signals) -> None:
        with self._lock:
            process = self._process
        if process and process.poll() is None:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass

    def _require_idle(self) -> None:
        with self._lock:
            if self.state not in {StudioState.IDLE, StudioState.FAULT}:
                raise Conflict(f"Studio is {self.state.value}")

    def _validate_paths(self, config: RunConfig) -> None:
        failed = [item["name"] for item in self.preflight(config) if not item["ok"]]
        if failed:
            raise ValueError("preflight failed: " + ", ".join(failed))

    def _policy_of(self, bundle: str) -> str:
        """The bundle's policy name, for the launcher's env var spelling."""
        try:
            manifest = json.loads(
                (Path(self._bundle_path(bundle)) / "bundle.json").read_text()
            )
        except (OSError, ValueError):
            return "polima"
        return str(manifest.get("policy") or "polima")

    def _bundle_path(self, bundle: str) -> str:
        candidate = Path(bundle)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.root / "models" / bundle).resolve()
        allowed = (self.root / "models").resolve()
        current = (self.root / "current").resolve() if (self.root / "current").exists() else allowed
        if resolved != current and allowed not in resolved.parents:
            raise ValueError("bundle must be installed below the PoLiMa model store")
        return str(resolved)

    def _env(self) -> dict[str, str]:
        return {**os.environ, "POLIMA_ROOT": str(self.root)}


def _manifest(bundle: Path, name: str = "bundle.json") -> dict[str, Any]:
    try:
        value = json.loads((bundle / name).read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def parse_benchmark_output(output: str) -> dict[str, Any] | None:
    summary = _BENCHMARK_RE.search(output)
    if summary is None:
        return None
    result: dict[str, Any] = {
        key: float(summary.group(key))
        for key in ("min", "mean", "median", "p95", "p99", "max")
    }
    result["iterations"] = int(summary.group("iterations"))
    result["warmup"] = int(summary.group("warmup"))
    result["throughput_hz"] = round(1000.0 / result["mean"], 3) if result["mean"] else None
    result["stages"] = [
        {"name": match.group("stage"), "ms": float(match.group("ms"))}
        for match in _STAGE_RE.finditer(output)
    ]
    check = _CHECK_RE.search(output)
    result["validation"] = (
        {
            "status": check.group("status").lower(),
            "cosine": float(check.group("cosine")),
            "mean_abs": float(check.group("mean_abs")),
            "max_abs": float(check.group("max_abs")),
        }
        if check
        else {"status": "unavailable"}
    )
    return result


def benchmark_failure(output: str) -> str:
    """Preserve the native diagnostic when a benchmark has no summary."""
    diagnostics = (
        "failed to load ",
        "no usable fixture inputs ",
        "no model loaded ",
    )
    for line in output.splitlines():
        message = line.strip()
        if any(marker in message for marker in diagnostics):
            return message
    return "native benchmark did not produce a result"


def process_failure(kind: str, code: int, recent: list[str] | deque[str]) -> str:
    """Turn a failed child's recent output into an operator-facing reason."""
    markers = (
        "error",
        "failed",
        "cannot",
        "can't",
        "missing",
        "not found",
        "refused",
        "denied",
        "unavailable",
    )
    for line in reversed(recent):
        if any(marker in line.lower() for marker in markers):
            return f"{kind}: {line}"
    return f"{kind} exited with code {code}; check Runtime Events for details"
