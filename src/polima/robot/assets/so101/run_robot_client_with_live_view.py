

"""Run LeRobot's async client with a headless Flask live camera view."""

from __future__ import annotations

import json
import logging
import os
import secrets
import signal
import socket
import threading
import time
from collections.abc import Iterator
from queue import Empty, Full, Queue

import cv2
import numpy as np
import torch

_DIRECT_SOM_IMPORT = os.environ.get("ACT_SOM_DIRECT", "0") == "1"
if not _DIRECT_SOM_IMPORT:
    from flask import Flask, Response, abort, redirect, render_template_string
    from werkzeug.serving import make_server

import lerobot.robots.so_follower  # noqa: F401 - register the SO-101 configuration
from lerobot.async_inference.helpers import TimedAction, TimedObservation
from lerobot.async_inference.robot_client import RobotClient, async_client
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.utils.import_utils import register_third_party_plugins

from polima.robot.so101 import SO101PolicyClient as ACTSOMClient
from polima.robot.so101 import record

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ACT live camera feeds</title>
  <style>
    body { margin: 0; padding: 24px; color: #eee; background: #111; font: 16px system-ui, sans-serif; }
    h1 { margin-top: 0; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
    figure { margin: 0; padding: 12px; background: #222; border-radius: 8px; }
    img { display: block; width: 100%; height: auto; background: #000; }
    figcaption { margin-top: 8px; }
  </style>
</head>
<body>
  <h1>ACT live camera feeds</h1>
  <p>The robot client is running. Stop it from its terminal with Ctrl-C.</p>
  <div class="grid">
    {% for camera_id, label in cameras %}
    <figure><img src="{{ url_for("stream", token=token, camera_id=camera_id) }}" alt="{{ label }}"><figcaption>{{ label }}</figcaption></figure>
    {% endfor %}
  </div>
</body>
</html>"""

CAMERAS = (
    ("overhead", "Overhead"),
    ("wrist", "Wrist"),
)

_original_control_loop_observation = RobotClient.control_loop_observation
_original_control_loop_action = RobotClient.control_loop_action
_original_aggregate_action_queues = RobotClient._aggregate_action_queues
_original_init = RobotClient.__init__
_original_start = RobotClient.start
_original_stop = RobotClient.stop
_original_action_tensor_to_dict = RobotClient._action_tensor_to_action_dict
_original_camera_configure = OpenCVCamera._configure_capture_settings
_frames: dict[str, bytes] = {}
_frame_condition = threading.Condition()
_active_client: RobotClient | None = None
_last_web_encode_s = 0.0
_direct_som = _DIRECT_SOM_IMPORT
_JOINT_NAMES = (
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
)


def _som_init(self: RobotClient, config) -> None:
    _init_with_reset_position(self, config)
    stats = os.environ.get("ACT_SOM_STATS")
    if not stats:
        raise ValueError("ACT_SOM_STATS must point to normalization_stats.npz")
    self.som_client = ACTSOMClient(self.server_address, stats)
    self.som_observations: Queue[TimedObservation] = Queue(maxsize=1)
    self.logger.info("Using direct Modalix transport at %s", self.server_address)


def _som_start(self: RobotClient) -> bool:
    self.logger.info("Startup safety reset: moving to the dataset resting position")
    if not _return_to_reset_position(self, stop_inference=False, force=True):
        record("error", message="The robot did not reach rest during startup")
        self.logger.error("Client startup aborted because the robot did not reach rest")
        _original_stop(self)
        return False
    self.reset_completed = False
    self.shutdown_event.clear()
    try:
        with socket.create_connection((self.som_client.host, self.som_client.port), timeout=2.0):
            pass
    except OSError as error:
        record("error", message=str(error))
        self.logger.error("Cannot reach Modalix server: %s", error)
        return False
    self.logger.info("Connected to Modalix server")
    return True


def _som_send_observation(self: RobotClient, observation: TimedObservation) -> bool:
    if not self.running:
        raise RuntimeError("Client is not running")
    try:
        self.som_observations.put_nowait(observation)
    except Full:
        try:
            self.som_observations.get_nowait()
        except Empty:
            pass
        self.som_observations.put_nowait(observation)
    return True


def _som_receive_actions(self: RobotClient, verbose: bool = False) -> None:
    self.start_barrier.wait()
    self.logger.info("Modalix action receiver starting")
    joint_names = tuple(self.robot.action_features)
    if joint_names != _JOINT_NAMES:
        record("error", message="Robot joint order does not match the policy")
        self.logger.error("ACT joint order must be %s, robot exposes %s", _JOINT_NAMES, joint_names)
        self.shutdown_event.set()
        return
    while self.running:
        try:
            observation = self.som_observations.get(timeout=0.2)
        except Empty:
            continue
        raw = observation.get_observation()
        try:
            state = np.asarray([raw[name] for name in joint_names], dtype=np.float32)
            actions, inference_ms = self.som_client.predict(raw["overhead"], raw["wrist"], state)
            count = min(int(self.config.actions_per_chunk), actions.shape[0])
            first_step = max(observation.get_timestep(), 0)
            timestamp = observation.get_timestamp()
            timed_actions = [
                TimedAction(timestamp=timestamp, timestep=first_step + index,
                            action=torch.from_numpy(actions[index].copy()))
                for index in range(count)
            ]
            self.action_chunk_size = max(self.action_chunk_size, count)
            self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
            self.must_go.set()
            record("inference_completed", model_ms=float(inference_ms),
                   queue_ms=(time.time() - timestamp) * 1000.0, actions=count)
            self.logger.info(
                "Modalix produced %d actions: model %.1f ms, observation-to-queue %.1f ms",
                count, inference_ms, (time.time() - timestamp) * 1000.0,
            )
        except Exception as error:
            record("error", message=str(error))
            self.logger.exception("Modalix inference request failed: %s", error)  # noqa: TRY401 - preserve driver diagnostics
            self.shutdown_event.set()
            return


def _configure_camera_with_perspective_autofocus(self: OpenCVCamera) -> None:
    """Restore the focus setting chosen on the preflight page."""
    _original_camera_configure(self)
    camera_ids = {
        os.environ.get("ACT_PERSPECTIVE_CAMERA", "2"): "perspective",
        os.environ.get("ACT_WRIST_CAMERA", "0"): "wrist",
    }
    camera_id = camera_ids.get(str(self.index_or_path))
    if camera_id is None or self.videocapture is None:
        return
    autofocus = True
    focus = 0
    config_path = os.environ.get("ACT_CAMERA_FOCUS_CONFIG")
    if config_path:
        try:
            with open(config_path, encoding="utf-8") as config_file:
                setting = json.load(config_file).get(camera_id, {})
            autofocus = bool(setting.get("autofocus", True))
            focus = int(setting.get("focus", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            logging.getLogger(__name__).warning("Could not read camera focus config; using autofocus")
    accepted = self.videocapture.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
    if accepted and (autofocus or self.videocapture.set(cv2.CAP_PROP_FOCUS, focus)):
        mode = "autofocus" if autofocus else f"manual focus {focus}"
        logging.getLogger(__name__).info("%s camera %s enabled", camera_id.title(), mode)
    else:
        logging.getLogger(__name__).warning(
            "Perspective camera backend did not accept autofocus"
        )


def _load_dataset_rest_position() -> dict[str, float]:
    default_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "robot_rest_position.json",
    )
    path = os.environ.get("ACT_REST_POSITION", default_path)
    with open(path, encoding="utf-8") as rest_file:
        payload = json.load(rest_file)
    positions = payload["positions"]
    if not positions or any(not key.endswith(".pos") for key in positions):
        raise ValueError(f"Invalid rest position file: {path}")
    return {key: float(value) for key, value in positions.items()}


def _begin_task_cycle(self: RobotClient) -> None:
    """Clear completion and rest-hold state before a new task cycle."""
    self.reset_completed = False
    self._auto_complete_started_s = None
    self._auto_complete_close_count = 0
    self._auto_complete_open_count = 0
    self._auto_complete_initial_open_seen = False
    self._auto_complete_grasp_seen = False
    self._auto_complete_triggered = False
    self._rest_snap_armed = False
    self._rest_snap_latched = False
    self._rest_snap_latched_s = None


def _action_tensor_to_dict_with_rest_snap(self: RobotClient, action_tensor):
    action = _original_action_tensor_to_dict(self, action_tensor)
    arm_keys = [
        key for key in self.reset_position if key != "gripper.pos" and key in action
    ]
    if not arm_keys or self.rest_snap_tolerance <= 0:
        return action

    if self._rest_snap_latched:
        if (
            self.auto_repeat_enabled
            and self._rest_snap_latched_s is not None
            and time.monotonic() - self._rest_snap_latched_s
            >= self.auto_repeat_pause_s
        ):

            with self.action_queue_lock:
                self.action_queue.queue.clear()
            self.must_go.set()
            _begin_task_cycle(self)
            self.logger.info(
                "Rest hold complete after %.1fs; starting the next task cycle",
                self.auto_repeat_pause_s,
            )
            action.update(self.reset_position)
            return action
        else:
            action.update(self.reset_position)
            return action

    max_rest_error = max(
        abs(float(action[key]) - float(self.reset_position[key])) for key in arm_keys
    )
    if not self._rest_snap_armed:
        if max_rest_error >= self.rest_snap_arm_distance:
            self._rest_snap_armed = True
            self.logger.info(
                "Rest snap armed after policy moved %.1f degrees away from rest",
                max_rest_error,
            )
        return action

    if not self._rest_snap_latched and max_rest_error <= self.rest_snap_tolerance:
        self._rest_snap_latched = True
        self._rest_snap_latched_s = time.monotonic()
        self.logger.info(
            "Policy returned within %.1f degrees of rest; snapping to exact rest pose",
            max_rest_error,
        )

    if self._rest_snap_latched:

        action.update(self.reset_position)
    return action


def _init_with_reset_position(self: RobotClient, config) -> None:
    global _active_client
    _original_init(self, config)
    self.reset_position = _load_dataset_rest_position()
    self.reset_completed = False
    self.rest_snap_tolerance = float(
        os.environ.get("ACT_REST_SNAP_TOLERANCE", "5")
    )
    self.rest_snap_arm_distance = max(15.0, self.rest_snap_tolerance * 3.0)
    self._rest_snap_armed = False
    self._rest_snap_latched = False
    self._rest_snap_latched_s = None
    self.auto_complete_enabled = os.environ.get("ACT_AUTO_COMPLETE", "1") != "0"
    self.auto_complete_close_threshold = float(
        os.environ.get("ACT_AUTO_CLOSE_THRESHOLD", "10")
    )
    self.auto_complete_open_threshold = float(
        os.environ.get("ACT_AUTO_OPEN_THRESHOLD", "20")
    )
    self.auto_complete_min_runtime_s = float(
        os.environ.get("ACT_AUTO_MIN_RUNTIME", "2")
    )
    self.auto_complete_close_frames = int(
        os.environ.get("ACT_AUTO_CLOSE_FRAMES", "5")
    )
    self.auto_complete_open_frames = int(
        os.environ.get("ACT_AUTO_OPEN_FRAMES", "8")
    )
    self.auto_repeat_enabled = os.environ.get("ACT_AUTO_REPEAT", "1") != "0"
    self.auto_repeat_pause_s = float(
        os.environ.get("ACT_AUTO_REPEAT_PAUSE", "2")
    )
    if self.auto_complete_open_threshold <= self.auto_complete_close_threshold:
        raise ValueError(
            "ACT_AUTO_OPEN_THRESHOLD must be higher than "
            "ACT_AUTO_CLOSE_THRESHOLD"
        )
    if self.auto_complete_min_runtime_s < 0:
        raise ValueError("ACT_AUTO_MIN_RUNTIME must be non-negative")
    if self.rest_snap_tolerance < 0:
        raise ValueError("ACT_REST_SNAP_TOLERANCE must be non-negative")
    if self.auto_repeat_pause_s < 0:
        raise ValueError("ACT_AUTO_REPEAT_PAUSE must be non-negative")
    if self.auto_complete_close_frames < 1 or self.auto_complete_open_frames < 1:
        raise ValueError("Auto-completion debounce frame counts must be positive")
    self._auto_complete_started_s = None
    self._auto_complete_close_count = 0
    self._auto_complete_open_count = 0
    self._auto_complete_initial_open_seen = False
    self._auto_complete_grasp_seen = False
    self._auto_complete_triggered = False
    self.logger.info(
        "Loaded dataset rest position (%d joints): %s",
        len(self.reset_position),
        self.reset_position,
    )
    if self.auto_complete_enabled:
        self.logger.info(
            "Automatic completion enabled: ignore initial closed rest pose, arm after "
            "opening at >= %.1f for %d frames, detect grasp after closing at <= %.1f "
            "for %d frames, then complete on the next opening (minimum runtime %.1fs)",
            self.auto_complete_open_threshold,
            self.auto_complete_open_frames,
            self.auto_complete_close_threshold,
            self.auto_complete_close_frames,
            self.auto_complete_min_runtime_s,
        )
        if self.auto_repeat_enabled:
            self.logger.info(
                "Automatic repeat enabled: return to rest after release, wait %.1fs "
                "for scene reset, then begin the next cycle",
                self.auto_repeat_pause_s,
            )
    else:
        self.logger.info("Automatic grasp-release completion detection disabled")
    _active_client = self


def _return_to_reset_position(
    self: RobotClient,
    duration_s: float = 5.0,
    lift_duration_s: float = 1.5,
    lift_degrees: float = 20.0,
    fps: int = 50,
    tolerance_degrees: float = 2.0,
    settle_timeout_s: float = 5.0,
    *,
    stop_inference: bool = True,
    force: bool = False,
) -> bool:
    if (self.reset_completed and not force) or not self.reset_position:
        return True

    if stop_inference:

        self.shutdown_event.set()
    self.logger.info("Returning robot to reset position before shutdown...")
    try:
        current_observation = self.robot.get_observation()
        current_position = {
            key: value
            for key, value in current_observation.items()
            if key in self.reset_position
        }

        lift_position = dict(current_position)
        for key in ("shoulder_lift.pos", "wrist_flex.pos"):
            if key not in lift_position:
                continue
            current = float(lift_position[key])
            target = float(self.reset_position[key])
            delta = max(-lift_degrees, min(lift_degrees, target - current))
            lift_position[key] = current + delta

        self.logger.info(
            "Reset stage 1/2: lifting clear of the basket with shoulder pan held at %.2f",
            float(current_position.get("shoulder_pan.pos", 0.0)),
        )
        lift_steps = max(int(lift_duration_s * fps), 1)
        for step in range(1, lift_steps + 1):
            ratio = step / lift_steps
            action = {
                key: current_position[key] * (1 - ratio)
                + lift_position[key] * ratio
                for key in current_position
            }
            self.robot.send_action(action)
            time.sleep(1 / fps)

        self.logger.info("Reset stage 2/2: returning to the resting position")
        steps = max(int(duration_s * fps), 1)
        for step in range(1, steps + 1):
            ratio = step / steps
            action = {
                key: lift_position[key] * (1 - ratio)
                + self.reset_position[key] * ratio
                for key in lift_position
            }
            self.robot.send_action(action)
            time.sleep(1 / fps)

        settle_deadline = time.monotonic() + settle_timeout_s
        while True:
            current_observation = self.robot.get_observation()
            errors = {
                key: abs(float(current_observation[key]) - target)
                for key, target in self.reset_position.items()
            }
            max_error = max(errors.values())
            if max_error <= tolerance_degrees:
                break
            if time.monotonic() >= settle_deadline:
                raise TimeoutError(
                    f"Rest position did not settle within {tolerance_degrees} degrees; "
                    f"maximum remaining error is {max_error:.2f} degrees"
                )
            self.robot.send_action(dict(self.reset_position))
            time.sleep(1 / fps)

        self.robot.send_action(dict(self.reset_position))
        self.reset_completed = True
        self.logger.info("Robot reached reset position")
        return True
    except Exception as error:  # noqa: BLE001 - driver failures must stop the reset
        record("error", message=str(error))
        self.logger.warning("Could not return robot to reset position: %s", error)
        return False


def _stop_after_reset(self: RobotClient) -> None:
    _return_to_reset_position(self)
    _original_stop(self)


def _start_after_reset(self: RobotClient) -> bool:

    self.logger.info(
        "Startup safety reset: moving to the dataset resting position before inference"
    )
    if not _return_to_reset_position(
        self,
        stop_inference=False,
        force=True,
    ):
        self.logger.error(
            "Client startup aborted because the robot did not reach the resting position"
        )
        _original_stop(self)
        return False

    self.reset_completed = False
    self.logger.info("Startup reset complete; connecting to the policy server")
    return _original_start(self)


def _request_graceful_shutdown(signum: int, _frame) -> None:
    record("operator_stop", signal=signum)
    if _active_client is None:
        raise KeyboardInterrupt
    _active_client.logger.info(
        "Received signal %s; requesting reset and shutdown", signum
    )
    _active_client.shutdown_event.set()
    if signum == signal.SIGINT:

        raise KeyboardInterrupt


def _aggregate_action_queues_with_latency(self, incoming_actions, aggregate_fn=None):
    if incoming_actions:
        latency_ms = max(0.0, (time.time() - incoming_actions[0].get_timestamp()) * 1000)
        self.live_latency_ms = latency_ms
        previous = getattr(self, "live_latency_ema_ms", latency_ms)
        self.live_latency_ema_ms = 0.85 * previous + 0.15 * latency_ms
        now = time.monotonic()
        if now - getattr(self, "_last_latency_print_s", 0.0) >= 1.0:
            with self.action_queue_lock:
                queue_size = self.action_queue.qsize()
            print(
                f"[latency] observation-to-actions={latency_ms:.1f} ms "
                f"(rolling={self.live_latency_ema_ms:.1f} ms), queue={queue_size}"
            )
            self._last_latency_print_s = now
    return _original_aggregate_action_queues(self, incoming_actions, aggregate_fn)


def _control_loop_action_with_completion(
    self: RobotClient,
    verbose: bool = False,
) -> dict[str, float]:
    performed_action = _original_control_loop_action(self, verbose)
    record("action_sent", positions=performed_action)
    if not self.auto_complete_enabled or self._auto_complete_triggered:
        return performed_action

    gripper_position = performed_action.get("gripper.pos")
    if gripper_position is None:
        return performed_action

    now = time.monotonic()
    if self._auto_complete_started_s is None:
        self._auto_complete_started_s = now

    if not self._auto_complete_initial_open_seen:
        if gripper_position >= self.auto_complete_open_threshold:
            self._auto_complete_open_count += 1
        else:
            self._auto_complete_open_count = 0
        if self._auto_complete_open_count >= self.auto_complete_open_frames:
            self._auto_complete_initial_open_seen = True
            self._auto_complete_open_count = 0
            self.logger.info(
                "Automatic completion armed after initial gripper opening at %.1f",
                gripper_position,
            )
        return performed_action

    if not self._auto_complete_grasp_seen:
        if gripper_position <= self.auto_complete_close_threshold:
            self._auto_complete_close_count += 1
        else:
            self._auto_complete_close_count = 0
        if self._auto_complete_close_count >= self.auto_complete_close_frames:
            self._auto_complete_grasp_seen = True
            record("grasp_detected")
            self.logger.info(
                "Automatic completion: sustained gripper close detected at %.1f",
                gripper_position,
            )
        return performed_action

    runtime_s = now - self._auto_complete_started_s
    if (
        runtime_s >= self.auto_complete_min_runtime_s
        and gripper_position >= self.auto_complete_open_threshold
    ):
        self._auto_complete_open_count += 1
    else:
        self._auto_complete_open_count = 0

    if self._auto_complete_open_count >= self.auto_complete_open_frames:
        self._auto_complete_triggered = True
        record("cycle_completed")
        if self.auto_repeat_enabled:
            self.logger.info(
                "Automatic completion: basket-drop release detected at gripper %.1f "
                "after %.1fs; returning to rest for the next cycle",
                gripper_position,
                runtime_s,
            )

            with self.action_queue_lock:
                self.action_queue.queue.clear()
            reset_succeeded = _return_to_reset_position(
                self,
                stop_inference=False,
                force=True,
            )
            if not reset_succeeded:
                self.logger.error(
                    "Automatic repeat stopped because the robot did not reach rest"
                )
                self.shutdown_event.set()
                return performed_action
            self.logger.info(
                "At rest; waiting %.1fs for the marker/basket scene to be reset",
                self.auto_repeat_pause_s,
            )
            time.sleep(self.auto_repeat_pause_s)

            with self.action_queue_lock:
                self.action_queue.queue.clear()
            self.must_go.set()
            _begin_task_cycle(self)
            self.logger.info("Starting the next automatic task cycle")
        else:
            self.logger.info(
                "Automatic completion: basket-drop release detected at gripper %.1f "
                "after %.1fs; stopping inference and returning to rest",
                gripper_position,
                runtime_s,
            )
            self.shutdown_event.set()

    return performed_action


def _control_loop_observation_with_web_view(
    self: RobotClient,
    task: str,
    verbose: bool = False,
):
    global _last_web_encode_s

    if not self.running:
        return None
    observation = _original_control_loop_observation(self, task, verbose)
    if observation is None:
        return None
    now = time.monotonic()
    if now - _last_web_encode_s < 0.1:
        return observation
    _last_web_encode_s = now

    encoded_frames: dict[str, bytes] = {}
    for camera_id, _ in CAMERAS:
        frame = observation.get(camera_id)
        if frame is None:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (480, 360))
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if ok:
            encoded_frames[camera_id] = encoded.tobytes()

    if encoded_frames:
        with _frame_condition:
            _frames.update(encoded_frames)
            _frame_condition.notify_all()
    return observation


def _stream(camera_id: str, stopped: threading.Event) -> Iterator[bytes]:
    last_frame: bytes | None = None
    while not stopped.is_set():
        with _frame_condition:
            _frame_condition.wait_for(
                lambda: _frames.get(camera_id) is not last_frame or stopped.is_set(),  # noqa: B023 - evaluated while holding the condition
                timeout=2,
            )
            frame = _frames.get(camera_id)
        if frame is not None and frame is not last_frame:
            last_frame = frame
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(frame)).encode()
                + b"\r\n\r\n"
                + frame
                + b"\r\n"
            )


def _display_host(host: str) -> str:
    if host not in {"0.0.0.0", "::"}:
        return host
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return str(probe.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


def start_live_server() -> tuple[object, threading.Thread, threading.Event]:
    host = os.environ.get("ACT_LIVE_HOST", "0.0.0.0")
    port = int(os.environ.get("ACT_LIVE_PORT", "5001"))
    token = secrets.token_urlsafe(18)
    stopped = threading.Event()
    app = Flask(__name__)

    def require_token(candidate: str) -> None:
        if not secrets.compare_digest(candidate, token):
            abort(404)

    @app.get("/")
    def root() -> Response:
        return redirect(f"/{token}/")

    @app.get("/favicon.ico")
    def favicon() -> Response:
        return Response(status=204)

    @app.get("/<token>/")
    def index(token: str) -> str:
        require_token(token)
        return render_template_string(PAGE, token=token, cameras=CAMERAS)

    @app.get("/<token>/stream/<camera_id>")
    def stream(token: str, camera_id: str) -> Response:
        require_token(token)
        if camera_id not in dict(CAMERAS):
            abort(404)
        return Response(
            _stream(camera_id, stopped),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    server = make_server(host, port, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(
        f"Live camera server: http://{_display_host(host)}:{port}/{token}/",
        flush=True,
    )
    return server, thread, stopped


RobotClient.__init__ = _som_init if _direct_som else _init_with_reset_position
RobotClient.start = _som_start if _direct_som else _start_after_reset
RobotClient.stop = _stop_after_reset
RobotClient.control_loop_action = _control_loop_action_with_completion
RobotClient.control_loop_observation = _original_control_loop_observation if _direct_som else _control_loop_observation_with_web_view
RobotClient._aggregate_action_queues = _aggregate_action_queues_with_latency
RobotClient._action_tensor_to_action_dict = _action_tensor_to_dict_with_rest_snap
if _direct_som:
    RobotClient.send_observation = _som_send_observation
    RobotClient.receive_actions = _som_receive_actions
OpenCVCamera._configure_capture_settings = _configure_camera_with_perspective_autofocus


if __name__ == "__main__":
    register_third_party_plugins()
    signal.signal(signal.SIGINT, _request_graceful_shutdown)
    signal.signal(signal.SIGTERM, _request_graceful_shutdown)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _request_graceful_shutdown)
    live_server = live_thread = live_stopped = None
    if not _direct_som:
        live_server, live_thread, live_stopped = start_live_server()
    else:
        print("Live camera preview disabled for direct Modalix inference.", flush=True)
    try:
        async_client()
    except KeyboardInterrupt:
        print(
            "Robot client interrupted; reset and shutdown complete.",
            flush=True,
        )
    finally:
        if live_stopped is not None:
            live_stopped.set()
            with _frame_condition:
                _frame_condition.notify_all()
            live_server.shutdown()
            live_thread.join(timeout=2)
