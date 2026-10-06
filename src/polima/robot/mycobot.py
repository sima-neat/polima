"""Fail-closed MyCobot 280 client for compiled PoLiMa policy bundles.

The Modalix board serves the model and runs this client beside the USB-attached
arm and cameras, exactly as it does for SO-101. Importing the module remains
numpy-only; LeRobot, OpenCV, torch, and pymycobot are loaded after the bundle
contract has been validated.
"""

from __future__ import annotations

import argparse
import importlib
import json
import signal
import sys
import time
import types
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from polima.bundle.layout import Bundle
from polima.policies.base import PolicySpec
from polima.robot.policy import (
    CompiledPolicyAdapter as SharedPolicyAdapter,
)
from polima.robot.policy import (
    PolicyPrediction,
    prepare_groot_image,
    prepare_smolvla_image,
)
from polima.wire.client import PolimaSOMClient

JOINT_KEYS = tuple(f"joint_{index}.pos" for index in range(1, 7))
GRIPPER_KEY = "gripper.pos"
ACTION_KEYS = (*JOINT_KEYS, GRIPPER_KEY)



def _json_safe(value: Any) -> Any:
    """Convert driver snapshots and numpy scalars to strict JSON values."""

    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


class DiagnosticRecorder:
    """Persist replayable cycles and timestamped robot feedback off the hot path."""

    SCHEMA_VERSION = 1

    def __init__(self, root: Path, metadata: dict[str, Any]) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.cycles_dir = self.root / "cycles"
        self.cycles_dir.mkdir()
        self.started_perf_s = time.perf_counter()
        self._events = (self.root / "events.jsonl").open("x", encoding="utf-8", buffering=1)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="polima-diagnostic")
        self._pending: list[Future[None]] = []
        self._write_json(
            self.root / "run.json",
            {
                "schema_version": self.SCHEMA_VERSION,
                "created_utc": datetime.now(UTC).isoformat(),
                **metadata,
            },
        )
        self.event("recording_started")

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(_json_safe(payload), stream, indent=2, allow_nan=False, sort_keys=True)
            stream.write("\n")

    @staticmethod
    def _write_cycle(path: Path, payload: dict[str, np.ndarray]) -> None:
        np.savez(path, **payload)

    def elapsed_s(self) -> float:
        return time.perf_counter() - self.started_perf_s

    def event(self, kind: str, **fields: Any) -> None:
        payload = {"elapsed_s": self.elapsed_s(), "event": kind, **fields}
        line = json.dumps(_json_safe(payload), allow_nan=False, sort_keys=True)
        self._events.write(line + "\n")

    def record_cycle(
        self,
        cycle: int,
        observation: dict[str, Any],
        prediction: PolicyPrediction,
        safe_actions: np.ndarray | None,
        *,
        observation_ms: float,
        total_before_execution_ms: float,
        validation_error: str | None = None,
    ) -> None:
        archive = self.cycles_dir / f"cycle_{cycle:06d}.npz"
        payload: dict[str, np.ndarray] = {
            "measured_state": np.asarray(
                [observation[name] for name in ACTION_KEYS], dtype=np.float32
            ).copy(),
            "raw_server_actions": np.asarray(prediction.raw_actions).copy(),
            "decoded_actions": np.asarray(prediction.actions).copy(),
            "inference_ms": np.asarray(prediction.latency_ms, dtype=np.float64),
            "observation_ms": np.asarray(observation_ms, dtype=np.float64),
            "total_before_execution_ms": np.asarray(total_before_execution_ms, dtype=np.float64),
        }
        if safe_actions is not None:
            payload["safe_actions"] = np.asarray(safe_actions).copy()
        if validation_error is not None:
            payload["validation_error"] = np.asarray(validation_error)
        for role in ("front", "wrist"):
            payload[f"camera_{role}_rgb"] = np.asarray(observation[role]).copy()
        for name, tensor in prediction.request_tensors.items():
            payload[f"request_{name}"] = np.asarray(tensor).copy()
        self._pending.append(self._executor.submit(self._write_cycle, archive, payload))
        if len(self._pending) > 2:
            self._pending.pop(0).result()
        self.event(
            "cycle_rejected" if validation_error is not None else "cycle_recorded",
            cycle=cycle,
            archive=str(archive.relative_to(self.root)),
            action_count=0 if safe_actions is None else len(safe_actions),
            inference_ms=prediction.latency_ms,
        )

    def finish(self, *, status: str, **fields: Any) -> None:
        try:
            for pending in self._pending:
                pending.result()
            self._executor.shutdown(wait=True)
            self.event("recording_finished", status=status, **fields)
            self._write_json(
                self.root / "result.json",
                {
                    "finished_utc": datetime.now(UTC).isoformat(),
                    "status": status,
                    **fields,
                },
            )
        finally:
            self._events.close()


@dataclass
class GraspReleaseCompletion:
    """Recognize one SO-101-style open, grasp, release task cycle."""

    close_threshold: float = 10.0
    open_threshold: float = 20.0
    close_frames: int = 5
    open_frames: int = 8
    min_runtime_s: float = 2.0

    def __post_init__(self) -> None:
        if self.open_threshold <= self.close_threshold:
            raise ValueError("completion open threshold must exceed close threshold")
        if self.close_frames < 1 or self.open_frames < 1:
            raise ValueError("completion debounce frame counts must be positive")
        if self.min_runtime_s < 0:
            raise ValueError("completion minimum runtime must be non-negative")
        self.started_s: float | None = None
        self.open_count = 0
        self.close_count = 0
        self.initial_open_seen = False
        self.grasp_seen = False
        self.completed = False

    def observe(self, gripper_position: float, *, now_s: float | None = None) -> str | None:
        """Return a stage name only when the commanded gripper stage changes."""

        if self.completed:
            return None
        now = time.monotonic() if now_s is None else float(now_s)
        if self.started_s is None:
            self.started_s = now
        value = float(gripper_position)

        # The recorded rest pose is closed. As on SO-101, an initial opening
        # must happen before a later close can count as a grasp.
        if not self.initial_open_seen:
            self.open_count = self.open_count + 1 if value >= self.open_threshold else 0
            if self.open_count >= self.open_frames:
                self.initial_open_seen = True
                self.open_count = 0
                return "opened"
            return None

        if not self.grasp_seen:
            self.close_count = self.close_count + 1 if value <= self.close_threshold else 0
            if self.close_count >= self.close_frames:
                self.grasp_seen = True
                self.open_count = 0
                return "grasped"
            return None

        runtime_s = now - self.started_s
        self.open_count = (
            self.open_count + 1
            if runtime_s >= self.min_runtime_s and value >= self.open_threshold
            else 0
        )
        if self.open_count >= self.open_frames:
            self.completed = True
            return "released"
        return None


def resolve_task_id(value: str | int | None, bundle: Bundle) -> int:
    """Resolve a task number, exact instruction, or short color-bin alias."""

    if not bundle.tasks:
        return 0
    if value is None:
        if len(bundle.tasks) == 1:
            return int(bundle.default_task)
        choices = "\n".join(
            f"  {index}: {instruction}" for index, instruction in enumerate(bundle.tasks)
        )
        raise ValueError("task is required for this multi-task bundle:\n" + choices)
    text = str(value).strip()
    try:
        task_id = int(text)
    except ValueError:
        normalized = text.casefold().replace("_", "-").replace(" ", "-")
        aliases: dict[str, int] = {}
        for index, instruction in enumerate(bundle.tasks):
            words = instruction.casefold().split()
            if "cube" in words and "bin" in words:
                color = words[words.index("cube") - 1]
                destination = words[words.index("bin") + 1]
                aliases[f"{color}-{destination}"] = index
            aliases[instruction.casefold()] = index
        if normalized in aliases:
            task_id = aliases[normalized]
        elif text.casefold() in aliases:
            task_id = aliases[text.casefold()]
        else:
            known = ", ".join(sorted(key for key in aliases if "-" in key and " " not in key))
            raise ValueError(f"unknown task {value!r}; use one of: {known}") from None
    if not 0 <= task_id < len(bundle.tasks):
        raise ValueError(f"task id {task_id} outside [0, {len(bundle.tasks)})")
    return task_id


def validate_profile(spec: PolicySpec) -> None:
    """Reject a wrong robot/bundle pairing before opening cameras or serial."""

    problems = []
    if spec.robot.calibration_id != "mycobot280":
        problems.append(f"robot profile is {spec.robot.calibration_id!r}, not 'mycobot280'")
    if spec.dataset.state_names != ACTION_KEYS:
        problems.append(f"joint order is {spec.dataset.state_names!r}, expected {ACTION_KEYS!r}")
    if spec.dataset.state_dim != 7 or spec.dataset.action_dim != 7:
        problems.append(
            f"state/action dimensions are {spec.dataset.state_dim}/{spec.dataset.action_dim}, "
            "expected 7/7"
        )
    if spec.robot.camera_keys != ("front", "wrist"):
        problems.append(f"camera order is {spec.robot.camera_keys!r}, expected ('front', 'wrist')")
    if problems:
        raise ValueError(
            "bundle is not compatible with MyCobot 280:\n  - " + "\n  - ".join(problems)
        )


def validate_action_chunk(
    actions: np.ndarray,
    observed_state: np.ndarray,
    joint_limits: tuple[tuple[float, float], ...] | list[tuple[float, float]],
    *,
    max_initial_error: float,
    max_trajectory_step: float,
    max_chunk_error: float,
) -> np.ndarray:
    """Validate the complete chunk before allowing its first physical command.

    The initial-target and consecutive-step checks do not bound cumulative
    motion across a chunk. ``max_chunk_error`` rejects a trajectory whose
    displacement from the measured chunk-start pose becomes excessive.
    """

    chunk = np.asarray(actions, dtype=np.float32)
    observed = np.asarray(observed_state, dtype=np.float32).reshape(-1)
    if chunk.ndim != 2 or chunk.shape[1] != len(ACTION_KEYS):
        raise ValueError(f"expected an action chunk shaped (steps, 7), got {chunk.shape}")
    if chunk.shape[0] == 0:
        raise ValueError("policy returned an empty action chunk")
    if observed.shape != (len(ACTION_KEYS),):
        raise ValueError(f"expected a measured 7-channel state, got {observed.shape}")
    if not np.isfinite(chunk).all():
        raise ValueError("policy returned NaN or infinite action values")
    if len(joint_limits) != len(JOINT_KEYS):
        raise ValueError(f"MyCobot driver exposed {len(joint_limits)} joint limits, expected 6")

    safe = chunk.copy()
    start = observed[:6].astype(np.float64)
    previous = start
    for action_index, row in enumerate(safe):
        arm = row[:6].astype(np.float64)
        for joint_index, (target, limits) in enumerate(
            zip(arm, joint_limits, strict=True), start=1
        ):
            low, high = float(limits[0]), float(limits[1])
            if not low <= target <= high:
                raise ValueError(
                    f"action[{action_index}] J{joint_index}={target:.2f} is outside "
                    f"[{low:.2f}, {high:.2f}]"
                )
        errors = np.abs(arm - previous)
        if action_index == 0 and np.any(errors > max_initial_error):
            joint_index = int(np.argmax(errors))
            raise ValueError(
                f"action[{action_index}] J{joint_index + 1} differs from the "
                f"measured pose by {errors[joint_index]:.2f} degrees; "
                f"limit is {max_initial_error:.2f}"
            )
        if action_index > 0:
            arm = previous + np.clip(
                arm - previous,
                -max_trajectory_step,
                max_trajectory_step,
            )
            safe[action_index, :6] = arm
        chunk_errors = np.abs(arm - start)
        if np.any(chunk_errors > max_chunk_error):
            joint_index = int(np.argmax(chunk_errors))
            raise ValueError(
                f"action[{action_index}] J{joint_index + 1} differs from the "
                f"chunk-start measured pose by {chunk_errors[joint_index]:.2f} "
                f"degrees; cumulative limit is {max_chunk_error:.2f}"
            )
        safe[action_index, 6] = float(np.clip(row[6], 0.0, 100.0))
        previous = arm
    return safe


class CompiledPolicyAdapter(SharedPolicyAdapter):
    """MyCobot validation around the shared policy transport."""

    def __init__(self, spec, bundle, address, **kwargs):
        validate_profile(spec)
        super().__init__(spec, bundle, address, client_factory=PolimaSOMClient, **kwargs)


def build_robot(config: Any):
    """Instantiate the installed MyCobot plugin through its public class."""

    _install_lerobot_types_compat()
    try:
        from lerobot_robot_mycobot280 import MyCobot280M5
    except ImportError as error:
        raise RuntimeError(
            "MyCobot support needs the lerobot_robot_mycobot280 package inside the deployed bundle"
        ) from error
    return MyCobot280M5(config)


def _camera_fps(args: Any, role: str) -> int:
    configured = getattr(args, f"{role}_camera_fps", None)
    return args.fps if configured is None else configured


def _cycle_limit_reached(args: Any, completed_cycles: int) -> bool:
    """Return whether a bounded robot run has completed its requested work."""

    return bool(args.once) or (args.max_cycles is not None and completed_cycles >= args.max_cycles)


def build_robot_config(args: Any):
    _install_lerobot_types_compat()
    try:
        from lerobot.cameras import Cv2Backends
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot_robot_mycobot280 import MyCobot280M5Config
        from lerobot_robot_mycobot280.mycobot280 import GRIPPER_KEY as plugin_gripper_key
        from lerobot_robot_mycobot280.mycobot280 import JOINT_KEYS as plugin_joint_keys
    except ImportError as error:
        raise RuntimeError("MyCobot support needs the lerobot_robot_mycobot280 package") from error
    if tuple(plugin_joint_keys) != JOINT_KEYS or plugin_gripper_key != GRIPPER_KEY:
        raise RuntimeError("installed MyCobot plugin uses an unexpected joint order")
    # Camera capture rates are hardware capabilities, independent of the policy
    # control rate. Keep args.fps for command timing and configure each camera
    # with the rate that its V4L2 mode actually exposes.
    cameras = {
        "front": OpenCVCameraConfig(
            index_or_path=Path(args.front_camera),
            width=args.camera_width,
            height=args.camera_height,
            fps=_camera_fps(args, "front"),
            fourcc=args.camera_fourcc,
            backend=Cv2Backends.V4L2,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=Path(args.wrist_camera),
            width=args.camera_width,
            height=args.camera_height,
            fps=_camera_fps(args, "wrist"),
            fourcc=args.camera_fourcc,
            backend=Cv2Backends.V4L2,
        ),
    }
    return MyCobot280M5Config(
        port=args.robot_port,
        command_speed=args.command_speed,
        fresh_mode=1,
        power_on_at_connect=False,
        connect_read_retries=10,
        read_retries=1,
        connection_settle_s=0.5,
        use_gripper=True,
        gripper_speed=args.gripper_speed,
        use_firmware_joint_limits=True,
        joint_limit_margin_deg=3.0,
        joint_limits_path=(
            str(Path(args.calibration_path).resolve()) if args.calibration_path else ""
        ),
        startup_recovery_to_neutral=False,
        cameras=cameras,
    )


def _install_lerobot_types_compat() -> None:
    """Provide type-only aliases omitted by some compatible LeRobot releases.

    The bundled MyCobot plugin imports these aliases for annotations only. Some
    LeRobot releases expose the robot and camera APIs we use but do not ship
    ``lerobot.types``. Keep that otherwise-compatible environment intact and
    supply only the two missing aliases instead of pinning or downgrading it.
    """

    try:
        importlib.import_module("lerobot.types")
        return
    except ModuleNotFoundError as error:
        if error.name != "lerobot.types":
            raise

    lerobot = importlib.import_module("lerobot")
    compatibility = types.ModuleType("lerobot.types")
    compatibility.RobotAction = dict[str, Any]
    compatibility.RobotObservation = dict[str, Any]
    sys.modules[compatibility.__name__] = compatibility
    lerobot.types = compatibility


def _connect_linux_cameras(robot: Any) -> list[Any]:
    """Open V4L2 cameras before the copied driver reaches its Windows fallback."""

    connected: list[Any] = []
    try:
        for camera in robot.cameras.values():
            camera.connect()
            connected.append(camera)
    except Exception:
        for camera in reversed(connected):
            if camera.is_connected:
                camera.disconnect()
        raise
    return connected


def _full_episode(args: Any) -> bool:
    """Whether this is a complete task run rather than a bounded diagnostic."""

    return bool(args.execute and not args.once)


def _completion_detector(args: Any) -> GraspReleaseCompletion | None:
    if not _full_episode(args) or getattr(args, "no_auto_complete", False):
        return None
    return GraspReleaseCompletion(
        close_threshold=getattr(args, "completion_close_threshold", 10.0),
        open_threshold=getattr(args, "completion_open_threshold", 20.0),
        close_frames=getattr(args, "completion_close_frames", 5),
        open_frames=getattr(args, "completion_open_frames", 8),
        min_runtime_s=getattr(args, "completion_min_runtime", 2.0),
    )


def _fresh_control_snapshot(robot: Any, *, phase: str) -> Any:
    """Pause on a brief feedback gap and fail before motion if it persists."""

    stale_s = float(getattr(robot.config, "feedback_stale_s", 0.15))
    abort_s = float(getattr(robot.config, "feedback_abort_s", 1.0))
    deadline = time.monotonic() + abort_s
    snapshot = None
    while time.monotonic() < deadline:
        snapshot = robot.get_control_snapshot()
        if snapshot.worker_fault is not None:
            raise RuntimeError(
                f"MyCobot control worker failed during {phase}: {snapshot.worker_fault}"
            )
        age_s = float(snapshot.joint_age_s)
        if (
            snapshot.joints_valid
            and snapshot.joints is not None
            and np.isfinite(age_s)
            and age_s <= stale_s
        ):
            return snapshot
        time.sleep(min(0.02, stale_s))

    age_s = float("inf") if snapshot is None else float(snapshot.joint_age_s)
    raise RuntimeError(
        f"MyCobot joint feedback remained unavailable during {phase} for "
        f"{abort_s:.2f}s (last age {age_s:.2f}s)"
    )


def _return_to_rest(
    robot: Any,
    args: Any,
    *,
    reason: str,
    recorder: DiagnosticRecorder | None = None,
) -> None:
    """Use the driver recorded neutral target and measured feedback, as during data capture."""

    neutral = tuple(float(value) for value in robot.neutral_position)
    gripper = robot.neutral_gripper_position
    if len(neutral) != len(JOINT_KEYS) or gripper is None:
        raise RuntimeError(
            "MyCobot calibration must contain neutral_position and neutral_gripper_position"
        )
    invalid = [
        (index, value, limits)
        for index, (value, limits) in enumerate(
            zip(neutral, robot.joint_limits, strict=True), start=1
        )
        if not float(limits[0]) <= value <= float(limits[1])
    ]
    if invalid:
        raise RuntimeError(f"recorded rest pose is outside joint limits: {invalid}")

    target = {
        **dict(zip(JOINT_KEYS, neutral, strict=True)),
        GRIPPER_KEY: float(gripper),
    }
    print(f"REST | {reason} | target {[round(value, 2) for value in neutral]}")
    if recorder is not None:
        recorder.event("rest_command", reason=reason, target=target)
    robot.next_joint_command_speed = max(1, round(args.command_speed * 0.5))
    robot.send_action(target)

    tolerance = float(getattr(args, "rest_tolerance_deg", 3.0))
    stable_needed = int(getattr(args, "rest_stable_samples", 3))
    rest_timeout = float(getattr(args, "rest_timeout", 30.0))
    deadline = time.monotonic() + rest_timeout
    stable = 0
    while time.monotonic() < deadline:
        snapshot = robot.get_control_snapshot()
        if recorder is not None:
            recorder.event("rest_feedback", reason=reason, snapshot=snapshot)
        if snapshot.worker_fault is not None:
            raise RuntimeError(
                f"MyCobot control worker failed during reset: {snapshot.worker_fault}"
            )
        joints_reached = (
            snapshot.joints_valid
            and snapshot.joints is not None
            and all(
                abs(float(current) - target_value) <= tolerance
                for current, target_value in zip(snapshot.joints, neutral, strict=True)
            )
        )
        gripper_reached = (
            snapshot.gripper_valid
            and snapshot.gripper_position is not None
            and abs(float(snapshot.gripper_position) - float(gripper)) <= 5.0
        )
        stable = stable + 1 if joints_reached and gripper_reached else 0
        if stable >= stable_needed:
            print("REST | reached")
            if recorder is not None:
                recorder.event("rest_reached", reason=reason, snapshot=snapshot)
            return
        time.sleep(1.0 / args.fps)
    raise TimeoutError(f"MyCobot did not reach the recorded rest pose within {rest_timeout:g}s")


def _wait_for_gripper_release(
    robot: Any,
    target: float,
    timeout_s: float = 5.0,
    *,
    recorder: DiagnosticRecorder | None = None,
) -> None:
    """Do not close for rest until the release command reached the hardware."""

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        snapshot = robot.get_control_snapshot()
        if recorder is not None:
            recorder.event("gripper_release_feedback", target=target, snapshot=snapshot)
        if snapshot.worker_fault is not None:
            raise RuntimeError(f"MyCobot gripper release failed: {snapshot.worker_fault}")
        target_matches = (
            snapshot.gripper_command_target is not None
            and abs(float(snapshot.gripper_command_target) - target) <= 1.0
        )
        measured_open = (
            snapshot.gripper_valid
            and snapshot.gripper_position is not None
            and abs(float(snapshot.gripper_position) - target) <= 5.0
        )
        if target_matches and snapshot.gripper_command_state == "complete" and measured_open:
            print("TASK | physical gripper release confirmed")
            if recorder is not None:
                recorder.event("gripper_release_confirmed", target=target, snapshot=snapshot)
            return
        time.sleep(0.05)
    raise TimeoutError("MyCobot gripper did not physically complete the release")


def run_client(spec: PolicySpec, bundle: Bundle, args: Any, *, task_id: int = 0) -> int:
    """Run preview/inference, moving hardware only when args.execute is true."""

    validate_profile(spec)
    full_episode = _full_episode(args)
    if args.calibration_path and not Path(args.calibration_path).is_file():
        raise SystemExit(f"MyCobot joint-limits file not found: {args.calibration_path}")
    if full_episode and not args.calibration_path:
        raise SystemExit(
            "full MyCobot task execution requires the recorded calibration/rest pose; "
            "pass --calibration-path or redeploy a bundle that includes it"
        )
    completion = _completion_detector(args)
    if args.front_camera == args.wrist_camera:
        raise SystemExit("front and wrist cameras must use different devices")
    if args.camera_width <= 0 or args.camera_height <= 0 or args.fps <= 0:
        raise SystemExit("camera size and FPS must be positive")
    if any(_camera_fps(args, role) <= 0 for role in ("front", "wrist")):
        raise SystemExit("camera capture FPS must be positive")
    if len(args.camera_fourcc) != 4:
        raise SystemExit("--camera-fourcc must contain four characters")
    for role in ("front", "wrist"):
        camera = Path(getattr(args, f"{role}_camera"))
        if not camera.exists():
            raise SystemExit(f"{role} camera not found: {camera}")
    if not Path(args.robot_port).exists():
        raise SystemExit(f"MyCobot serial port not found: {args.robot_port}")
    if args.actions_per_chunk is not None and not (
        1 <= args.actions_per_chunk <= spec.robot.actions_per_chunk
    ):
        raise SystemExit(
            f"--actions-per-chunk must be between 1 and {spec.robot.actions_per_chunk}"
        )
    if not 1 <= args.command_speed <= 100 or not 1 <= args.gripper_speed <= 100:
        raise SystemExit("robot speeds must be between 1 and 100")
    if (
        args.max_initial_target_error_deg <= 0
        or args.max_trajectory_step_deg <= 0
        or args.max_chunk_error_deg <= 0
    ):
        raise SystemExit("safety thresholds must be positive")
    if full_episode and (
        args.rest_timeout <= 0 or args.rest_tolerance_deg <= 0 or args.rest_stable_samples <= 0
    ):
        raise SystemExit("rest timeout, tolerance, and stable samples must be positive")
    if args.max_cycles is not None and args.max_cycles <= 0:
        raise SystemExit("--max-cycles must be positive")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")

    adapter = CompiledPolicyAdapter(
        spec,
        bundle,
        args.server,
        task_id=task_id,
        timeout=args.timeout,
        seed=args.seed,
    )
    if not adapter.client.ping():
        raise SystemExit(f"compiled policy server is not reachable at {adapter.client.address}")

    recorder = None
    diagnostic_dir = getattr(args, "diagnostic_dir", None)
    if diagnostic_dir is not None:
        recorder = DiagnosticRecorder(
            Path(diagnostic_dir),
            {
                "bundle_id": bundle.bundle_id,
                "bundle_root": str(bundle.root.resolve()),
                "policy": spec.name,
                "robot": spec.robot.calibration_id,
                "server": adapter.client.address,
                "task_id": task_id,
                "task_instruction": bundle.tasks[task_id] if bundle.tasks else None,
                "execute": bool(args.execute),
                "auto_complete": completion is not None,
                "seed": args.seed,
                "control_fps": args.fps,
                "camera_width": args.camera_width,
                "camera_height": args.camera_height,
                "camera_fourcc": args.camera_fourcc,
                "front_camera": args.front_camera,
                "front_camera_fps": _camera_fps(args, "front"),
                "wrist_camera": args.wrist_camera,
                "wrist_camera_fps": _camera_fps(args, "wrist"),
                "actions_per_chunk": args.actions_per_chunk or spec.robot.actions_per_chunk,
                "command_speed": args.command_speed,
                "gripper_speed": args.gripper_speed,
                "max_initial_target_error_deg": args.max_initial_target_error_deg,
                "max_trajectory_step_deg": args.max_trajectory_step_deg,
                "max_chunk_error_deg": args.max_chunk_error_deg,
                "max_cycles": args.max_cycles,
            },
        )

    mode = "EXECUTE - ROBOT MAY MOVE" if args.execute else "PREVIEW ONLY"
    print(f"MyCobot 280 | {spec.display_name} | {mode}")
    print(f"  bundle: {bundle.bundle_id}")
    print(f"  server: {adapter.client.address}")
    if bundle.tasks:
        print(f"  task: {task_id} - {bundle.tasks[task_id]}")
    if completion is not None:
        print("  completion: gripper open -> grasp/close -> release/open")
    print(
        f"  safety: first target <= {args.max_initial_target_error_deg:g} deg; "
        f"trajectory step <= {args.max_trajectory_step_deg:g} deg; "
        f"chunk drift <= {args.max_chunk_error_deg:g} deg"
    )
    if not args.execute:
        print("  motion disabled; add --execute only after checking preview actions")
    else:
        print("  WARNING: keep the workspace clear and emergency stop available")
    if args.max_cycles is not None:
        print(f"  limit: {args.max_cycles} inference cycle(s)")
    if recorder is not None:
        print(f"  diagnostics: {recorder.root}")

    robot = build_robot(build_robot_config(args))
    preconnected = _connect_linux_cameras(robot)
    stopping = False
    completed_cycles = 0
    task_completed = False
    at_rest = False
    policy_motion_started = False
    stop_reason = "running"

    def request_stop(signum, _frame) -> None:
        nonlocal stopping, stop_reason
        stopping = True
        stop_reason = f"signal:{signal.Signals(signum).name}"
        if recorder is not None:
            recorder.event("stop_requested", reason=stop_reason)

    old_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        old_handlers[signum] = signal.signal(signum, request_stop)
    try:
        robot.connect()
        if full_episode:
            _return_to_rest(robot, args, reason="startup", recorder=recorder)
            at_rest = True
        while not stopping:
            cycle_started = time.perf_counter()
            observation_started = time.perf_counter()
            _fresh_control_snapshot(robot, phase=f"cycle {completed_cycles} observation")
            observation = robot.get_observation()
            observation_ms = (time.perf_counter() - observation_started) * 1000.0
            state = np.asarray([observation[name] for name in ACTION_KEYS], dtype=np.float32)
            prediction = None
            if recorder is not None:
                prediction = adapter.predict_detailed(observation)
                actions = prediction.actions
                inference_ms = prediction.latency_ms
            else:
                actions, inference_ms = adapter.predict(observation)
            count = min(args.actions_per_chunk or len(actions), len(actions))
            try:
                safe = validate_action_chunk(
                    actions[:count],
                    state,
                    robot.joint_limits,
                    max_initial_error=args.max_initial_target_error_deg,
                    max_trajectory_step=args.max_trajectory_step_deg,
                    max_chunk_error=args.max_chunk_error_deg,
                )
            except ValueError as error:
                if recorder is not None:
                    assert prediction is not None
                    recorder.record_cycle(
                        completed_cycles,
                        observation,
                        prediction,
                        None,
                        observation_ms=observation_ms,
                        total_before_execution_ms=(time.perf_counter() - cycle_started) * 1000.0,
                        validation_error=str(error),
                    )
                raise
            requested = np.asarray(actions[:count], dtype=np.float32)
            if recorder is not None:
                assert prediction is not None
                recorder.record_cycle(
                    completed_cycles,
                    observation,
                    prediction,
                    safe,
                    observation_ms=observation_ms,
                    total_before_execution_ms=(time.perf_counter() - cycle_started) * 1000.0,
                )
            smoothed = np.abs(safe[:, :6] - requested[:, :6]) > 1e-5
            if np.any(smoothed):
                print(
                    "SAFETY | smoothed "
                    f"{int(np.count_nonzero(smoothed))} joint target(s) to "
                    f"<={args.max_trajectory_step_deg:g} deg per action"
                )
            first = ", ".join(f"{value:.2f}" for value in safe[0])
            print(f"inference {inference_ms:.1f} ms | {len(safe)} actions | first [{first}]")
            if args.log_actions:
                for index, action in enumerate(safe):
                    print(f"  action[{index}]: {action.tolist()}")

            if args.execute:
                for action_index, values in enumerate(safe):
                    if stopping:
                        break
                    tick = time.perf_counter()
                    snapshot_before = _fresh_control_snapshot(
                        robot,
                        phase=f"cycle {completed_cycles} action {action_index}",
                    )
                    command = {
                        **{name: float(values[index]) for index, name in enumerate(JOINT_KEYS)},
                        # MyCobot demonstrations use binary gripper targets.
                        # Threshold model interpolation to avoid queueing a slow
                        # serial edge for every intermediate prediction.
                        GRIPPER_KEY: 100.0 if float(values[6]) >= 50.0 else 0.0,
                    }
                    returned_action = robot.send_action(command)
                    policy_motion_started = True
                    at_rest = False
                    if completion is not None:
                        transition = completion.observe(command[GRIPPER_KEY])
                        if transition == "opened":
                            print("TASK | gripper opened; seeking grasp")
                        elif transition == "grasped":
                            print("TASK | sustained close detected; carrying")
                        elif transition == "released":
                            print("TASK | release detected; waiting for physical gripper")
                            _wait_for_gripper_release(
                                robot, command[GRIPPER_KEY], recorder=recorder
                            )
                            print("TASK | task completion inferred from grasp-release sequence")
                            task_completed = True
                            stopping = True
                            stop_reason = "commanded_grasp_release_sequence"
                    remaining = 1.0 / args.fps - (time.perf_counter() - tick)
                    if remaining > 0:
                        time.sleep(remaining)
                    if recorder is not None:
                        recorder.event(
                            "action_command",
                            cycle=completed_cycles,
                            action=action_index,
                            decoded_safe_action=values,
                            binary_gripper_target=command[GRIPPER_KEY],
                            command=command,
                            returned_action=returned_action,
                            tick_ms=(time.perf_counter() - tick) * 1000.0,
                            snapshot_before=snapshot_before,
                            snapshot_after=robot.get_control_snapshot(),
                        )
            completed_cycles += 1
            if _cycle_limit_reached(args, completed_cycles):
                stop_reason = "once" if args.once else "max_cycles"
                if recorder is not None:
                    recorder.event(
                        "stop_requested",
                        reason=stop_reason,
                        completed_cycles=completed_cycles,
                    )
                break
            if not args.execute:
                remaining = 1.0 / args.fps - (time.perf_counter() - cycle_started)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        active_error = sys.exc_info()[1]
        if active_error is not None and stop_reason == "running":
            stop_reason = f"exception:{type(active_error).__name__}"
        if recorder is not None and active_error is not None:
            recorder.event(
                "exception",
                error_type=type(active_error).__name__,
                message=str(active_error),
                completed_cycles=completed_cycles,
            )
        if getattr(robot, "is_connected", False):
            try:
                if full_episode and policy_motion_started and not at_rest:
                    _return_to_rest(robot, args, reason="task finished", recorder=recorder)
                    at_rest = True
            finally:
                robot.disconnect()
        else:
            for camera in reversed(preconnected):
                if camera.is_connected:
                    camera.disconnect()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if recorder is not None:
            recorder.finish(
                status=(
                    "error"
                    if active_error is not None
                    else "complete"
                    if task_completed
                    else "stopped"
                ),
                stop_reason=stop_reason,
                completed_cycles=completed_cycles,
                task_completed=task_completed,
                returned_to_rest=at_rest,
                error=(
                    None
                    if active_error is None
                    else f"{type(active_error).__name__}: {active_error}"
                ),
            )
            print(f"diagnostics saved to {recorder.root}")
    if task_completed:
        print("TASK COMPLETE | robot returned to rest")
    elif full_episode:
        print("TASK STOPPED before completion | robot returned to rest")
    print("stopped MyCobot client")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Bundle-side entry point used by ``robot_client/start.sh`` on Modalix."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--robot-port", required=True)
    parser.add_argument("--front-camera", required=True)
    parser.add_argument("--wrist-camera", required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--front-camera-fps", type=int)
    parser.add_argument("--wrist-camera-fps", type=int)
    parser.add_argument("--camera-fourcc")
    parser.add_argument("--calibration-path", type=Path)
    parser.add_argument("--actions-per-chunk", type=int)
    parser.add_argument("--command-speed", type=int, default=30)
    parser.add_argument("--gripper-speed", type=int, default=25)
    parser.add_argument("--max-initial-target-error-deg", type=float, default=12.0)
    parser.add_argument("--max-trajectory-step-deg", type=float, default=5.0)
    parser.add_argument(
        "--max-chunk-error-deg",
        type=float,
        default=15.0,
        help="maximum cumulative joint displacement from the measured chunk-start pose",
    )
    parser.add_argument("--task")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--no-auto-complete", action="store_true")
    parser.add_argument("--completion-close-threshold", type=float, default=10.0)
    parser.add_argument("--completion-open-threshold", type=float, default=20.0)
    parser.add_argument("--completion-close-frames", type=int, default=5)
    parser.add_argument("--completion-open-frames", type=int, default=8)
    parser.add_argument("--completion-min-runtime", type=float, default=2.0)
    parser.add_argument("--rest-timeout", type=float, default=30.0)
    parser.add_argument("--rest-tolerance-deg", type=float, default=3.0)
    parser.add_argument("--rest-stable-samples", type=int, default=3)
    parser.add_argument("--log-actions", action="store_true")
    parser.add_argument(
        "--diagnostic-dir",
        type=Path,
        help="new directory for replayable camera, tensor, action, and feedback logs",
    )
    args = parser.parse_args(argv)

    from polima.bundle.contract import spec_from_bundle
    from polima.policies.registry import get_policy
    from polima.util.jsonio import read_json

    manifest = args.bundle / "bundle.json"
    if not manifest.is_file():
        parser.error(f"no bundle manifest at {manifest}")
    bundle = Bundle.from_dict(read_json(manifest), args.bundle)
    try:
        task_id = resolve_task_id(args.task, bundle)
    except ValueError as error:
        parser.error(str(error))
    spec = spec_from_bundle(get_policy(bundle.policy), bundle)
    args.camera_fourcc = args.camera_fourcc or spec.robot.camera_fourcc
    for role in ("front", "wrist"):
        attribute = f"{role}_camera_fps"
        if getattr(args, attribute) is None:
            setattr(args, attribute, spec.robot.camera_capture_fps.get(role))
    return run_client(spec, bundle, args, task_id=task_id)


__all__ = [
    "ACTION_KEYS",
    "CompiledPolicyAdapter",
    "prepare_groot_image",
    "prepare_smolvla_image",
    "run_client",
    "validate_action_chunk",
    "validate_profile",
]


if __name__ == "__main__":
    raise SystemExit(main())
