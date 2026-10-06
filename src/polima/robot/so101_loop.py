"""Drive an SO-101 follower arm with a deployed PoLiMa bundle (ACT or GR00T N1.6).

LeRobot's async client runs the arm; PoLiMa replaces its gRPC policy with the
board's polima_server and adds the cycle the SO-101 bundles were proven with:
start at the dataset rest pose, snap back to it when the policy returns, detect
the grasp-and-release that ends a pick, and return to rest on every exit.

Runs on the board, in the LeRobot environment, via a bundle's
robot_client/start.sh: `python -m polima.robot.so101` hands ACT bundles to this
module. GR00T and SmolVLA bundles take polima.robot.so101's own path (the
established ACT control loop with PoLiMa as its transport), which is the one
verified on the robot with them; this loop still accepts a GR00T bundle when
run directly (`python -m polima.robot.so101_loop`). Studio's toggles arrive as
POLIMA_AUTO_COMPLETE, POLIMA_AUTO_REPEAT and POLIMA_TASK. An emergency halt
(SIGUSR1) is not caught: it ends the process at once, without the
return-to-rest motion.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import sys
import time
from pathlib import Path
from queue import Empty, Full, Queue

log = logging.getLogger("polima.so101")

REST_TOLERANCE = 5.0          # degrees: the policy is "back at rest"
OPEN, CLOSED = 20.0, 10.0     # gripper positions for release and grasp
OPEN_FRAMES, CLOSED_FRAMES = 8, 5
MIN_CYCLE_S = 2.0


def load_rest_pose(path: Path) -> dict[str, float]:
    positions = json.loads(Path(path).read_text())["positions"]
    if not positions or any(not key.endswith(".pos") for key in positions):
        raise ValueError(f"invalid rest pose: {path}")
    return {key: float(value) for key, value in positions.items()}


class RestSnap:
    """Hold the exact rest pose once the policy brings the arm back near it.

    It arms only after the arm has moved well away, so the first observations at
    rest cannot latch it. With repeat, the hold ends after `pause_s` and
    `apply` reports that the next cycle may start.
    """

    def __init__(self, rest: dict[str, float], repeat: bool, pause_s: float) -> None:
        self.rest, self.repeat, self.pause_s = rest, repeat, pause_s
        self.arm_distance = max(15.0, REST_TOLERANCE * 3.0)
        self.reset()

    def reset(self) -> None:
        self.armed, self.latched_at = False, None

    def apply(self, action: dict[str, float], now: float) -> tuple[dict[str, float], bool]:
        joints = [key for key in self.rest if key != "gripper.pos" and key in action]
        if not joints:
            return action, False
        if self.latched_at is not None:
            restart = self.repeat and now - self.latched_at >= self.pause_s
            if restart:
                self.reset()
            action.update(self.rest)
            return action, restart
        error = max(abs(float(action[key]) - self.rest[key]) for key in joints)
        if not self.armed:
            self.armed = error >= self.arm_distance
            if self.armed:
                log.info("Rest snap armed after moving %.1f degrees from rest", error)
        elif error <= REST_TOLERANCE:
            self.latched_at = now
            log.info("Back within %.1f degrees of rest; holding the rest pose", error)
            action.update(self.rest)
        return action, False


class GraspCycle:
    """Detect one pick: open, close on the object, then open again to release.

    The rest pose has the gripper closed, so closing only counts after a first
    opening. Each transition must hold for several frames.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.started = None
        self.opened = self.grasped = self.released = False
        self.open_frames = self.closed_frames = 0

    def update(self, gripper: float, now: float) -> bool:
        """True on the frame the release is confirmed."""
        if self.released:
            return False
        self.started = self.started if self.started is not None else now
        if not self.opened:
            self.open_frames = self.open_frames + 1 if gripper >= OPEN else 0
            if self.open_frames >= OPEN_FRAMES:
                self.opened, self.open_frames = True, 0
                log.info("Grasp cycle armed: gripper opened to %.1f", gripper)
            return False
        if not self.grasped:
            self.closed_frames = self.closed_frames + 1 if gripper <= CLOSED else 0
            self.grasped = self.closed_frames >= CLOSED_FRAMES
            if self.grasped:
                log.info("Grasp detected: gripper closed to %.1f", gripper)
            return False
        opening = gripper >= OPEN and now - self.started >= MIN_CYCLE_S
        self.open_frames = self.open_frames + 1 if opening else 0
        self.released = self.open_frames >= OPEN_FRAMES
        return self.released


def return_to_rest(robot, rest: dict[str, float], *, fps: int = 50, lift_s: float = 1.5,
                   move_s: float = 5.0, lift_degrees: float = 20.0,
                   tolerance: float = 2.0, settle_s: float = 5.0) -> None:
    """Lift clear, move to the rest pose, and hold it until every joint settles.

    The lift changes only shoulder_lift and wrist_flex, so the arm rises out of
    the basket before the larger move instead of sweeping through it.
    """
    start = {key: float(value) for key, value in robot.get_observation().items() if key in rest}
    lifted = dict(start)
    for key in ("shoulder_lift.pos", "wrist_flex.pos"):
        if key in lifted:
            lifted[key] += max(-lift_degrees, min(lift_degrees, rest[key] - lifted[key]))
    for source, target, seconds in ((start, lifted, lift_s), (lifted, rest, move_s)):
        steps = max(int(seconds * fps), 1)
        for step in range(1, steps + 1):
            ratio = step / steps
            robot.send_action({key: source[key] * (1 - ratio) + target[key] * ratio
                               for key in source})
            time.sleep(1 / fps)
    deadline = time.monotonic() + settle_s
    while True:
        observation = robot.get_observation()
        error = max(abs(float(observation[key]) - target) for key, target in rest.items())
        if error <= tolerance:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"rest pose not reached; {error:.2f} degrees remain")
        robot.send_action(dict(rest))
        time.sleep(1 / fps)
    robot.send_action(dict(rest))


class SO101Client:
    """Mixed in ahead of LeRobot's RobotClient (see `client_class`)."""

    #: Set by `main` before LeRobot constructs the client.
    model = rest = recorder = None
    auto_complete = auto_repeat = True
    repeat_pause_s = 2.0
    active: SO101Client | None = None
    stop_requested = False
    failure: str | None = None

    def __init__(self, config) -> None:
        super().__init__(config)
        self.at_rest = False
        self.snap = RestSnap(self.rest, self.auto_repeat, self.repeat_pause_s)
        self.cycle = GraspCycle()
        self.observations: Queue = Queue(maxsize=1)
        self.last_latency_print = 0.0
        self.cycles = 0
        SO101Client.active = self

    def record(self, kind: str, **fields) -> None:
        if self.recorder is not None:
            self.recorder.event(kind, **fields)

    def fail(self, message: str) -> None:
        SO101Client.failure = SO101Client.failure or message
        self.record("error", message=message)

    # -- rest pose -----------------------------------------------------------

    def go_to_rest(self, stop_inference: bool = True) -> bool:
        if stop_inference:
            self.shutdown_event.set()
        self.logger.info("Returning to the rest pose")
        try:
            return_to_rest(self.robot, self.rest)
        except Exception as error:  # noqa: BLE001 - the caller decides what to stop
            self.logger.warning("Could not reach the rest pose: %s", error)
            self.record("warning", message=f"could not reach the rest pose: {error}")
            return False
        self.at_rest = True
        return True

    def start(self) -> bool:
        # Motors and cameras are connected, but no observation has been sent:
        # establish the known start pose before the policy sees anything.
        if not self.go_to_rest(stop_inference=False):
            self.logger.error("Startup aborted: the arm did not reach the rest pose")
            self.fail("startup aborted: the arm did not reach the rest pose")
            super().stop()
            return False
        self.at_rest = False
        self.shutdown_event.clear()
        server = self.model.client
        try:
            socket.create_connection((server.host, server.port), timeout=2.0).close()
        except OSError as error:
            self.logger.error("Cannot reach the policy server at %s: %s", server.address, error)
            self.fail(f"cannot reach the policy server at {server.address}: {error}")
            return False
        self.logger.info("Connected to the policy server at %s", server.address)
        self.record("cycle_started", cycle=1)
        return True

    def stop(self) -> None:
        if not self.at_rest:
            self.go_to_rest()
        super().stop()

    # -- policy transport ----------------------------------------------------

    def send_observation(self, observation) -> bool:
        if not self.running:
            raise RuntimeError("client is not running")
        try:
            self.observations.put_nowait(observation)
        except Full:
            try:
                self.observations.get_nowait()
            except Empty:
                pass
            self.observations.put_nowait(observation)
        return True

    def receive_actions(self, verbose: bool = False) -> None:
        import torch
        from lerobot.async_inference.helpers import TimedAction

        self.start_barrier.wait()
        joints = tuple(self.robot.action_features)
        if joints != tuple(self.model.joint_names):
            self.logger.error("Joint order must be %s; the arm reports %s",
                              self.model.joint_names, joints)
            self.fail(f"joint order mismatch: the arm reports {list(joints)}")
            self.shutdown_event.set()
            return
        while self.running:
            try:
                observation = self.observations.get(timeout=0.2)
            except Empty:
                continue
            raw = observation.get_observation()
            try:
                actions, model_ms = self.model.predict(raw)
                count = min(int(self.config.actions_per_chunk), actions.shape[0])
                first, stamp = max(observation.get_timestep(), 0), observation.get_timestamp()
                chunk = [TimedAction(timestamp=stamp, timestep=first + index,
                                     action=torch.from_numpy(actions[index].copy()))
                         for index in range(count)]
                self.action_chunk_size = max(self.action_chunk_size, count)
                self._aggregate_action_queues(chunk, self.config.aggregate_fn)
                self.must_go.set()
                queue_ms = (time.time() - stamp) * 1000.0
                self.logger.info("%d actions: model %.1f ms, observation to queue %.1f ms",
                                 count, model_ms, queue_ms)
                self.record("inference_completed", step=first, actions=count,
                            model_ms=round(float(model_ms), 2), queue_ms=round(queue_ms, 2))
            except Exception as error:  # stop the arm on any failure
                self.logger.exception("Policy request failed")
                self.fail(f"policy request failed: {error}")
                self.shutdown_event.set()
                return

    def _aggregate_action_queues(self, incoming, aggregate_fn=None):
        now = time.monotonic()
        if incoming and now - self.last_latency_print >= 1.0:
            latency = max(0.0, (time.time() - incoming[0].get_timestamp()) * 1000)
            print(f"[latency] observation-to-actions={latency:.1f} ms, "
                  f"queue={self.action_queue.qsize()}", flush=True)
            self.last_latency_print = now
        return super()._aggregate_action_queues(incoming, aggregate_fn)

    # -- task cycle ----------------------------------------------------------

    def clear_actions(self, observe: bool = True) -> None:
        with self.action_queue_lock:
            self.action_queue.queue.clear()
        if observe:
            self.must_go.set()

    def _action_tensor_to_action_dict(self, action_tensor):
        action, restart = self.snap.apply(super()._action_tensor_to_action_dict(action_tensor),
                                          time.monotonic())
        if restart:
            # Predictions made before the hold belong to the finished cycle, and
            # so does its grasp: a miss must not make the retry's opening a release.
            self.clear_actions()
            self.cycle.reset()
            self.at_rest = False
            self.logger.info("Rest hold complete; starting the next cycle")
            self.record("cycle_restarted", reason="back at rest without a release")
        return action

    def control_loop_action(self, verbose: bool = False):
        performed = super().control_loop_action(verbose)
        self.record("action_sent", step=getattr(self, "latest_action", None),
                    action={key: round(float(value), 2) for key, value in performed.items()})
        gripper = performed.get("gripper.pos")
        if not self.auto_complete or gripper is None:
            return performed
        opened, grasped = self.cycle.opened, self.cycle.grasped
        released = self.cycle.update(gripper, time.monotonic())
        if self.cycle.opened and not opened:
            self.record("gripper_opened", gripper=round(float(gripper), 2))
        if self.cycle.grasped and not grasped:
            self.record("grasp_detected", gripper=round(float(gripper), 2))
        if not released:
            return performed
        self.cycles += 1
        self.record("release_detected", gripper=round(float(gripper), 2))
        if not self.auto_repeat:
            self.logger.info("Release detected; stopping and returning to rest")
            self.record("cycle_completed", cycle=self.cycles)
            self.shutdown_event.set()
            return performed
        self.logger.info("Release detected; returning to rest for the next cycle")
        self.clear_actions(observe=False)  # nothing planned past the release runs
        if not self.go_to_rest(stop_inference=False):
            self.logger.error("Repeat stopped: the arm did not reach the rest pose")
            self.record("warning", message="repeat stopped: the arm did not reach the rest pose")
            self.shutdown_event.set()
            return performed
        self.logger.info("At rest; waiting %.1fs for the scene to be reset", self.repeat_pause_s)
        time.sleep(self.repeat_pause_s)
        self.clear_actions()
        self.at_rest = False
        self.cycle.reset()
        self.snap.reset()
        self.record("cycle_completed", cycle=self.cycles)
        self.record("cycle_started", cycle=self.cycles + 1)
        return performed


def client_class():
    from lerobot.async_inference.robot_client import RobotClient

    return type("SO101RobotClient", (SO101Client, RobotClient), {})


def apply_camera_focus(focus_config: Path, roles: dict[str, str]) -> None:
    """Re-apply the focus chosen in the camera preview whenever LeRobot opens a camera."""
    import cv2
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

    configure = OpenCVCamera._configure_capture_settings

    def configured(camera) -> None:
        configure(camera)
        role = roles.get(str(camera.index_or_path))
        if role is None or camera.videocapture is None:
            return
        try:
            setting = json.loads(focus_config.read_text()).get(role, {})
        except (OSError, ValueError, AttributeError):
            setting = {}
        autofocus = bool(setting.get("autofocus", True))
        capture = camera.videocapture
        if not (capture.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
                and (autofocus or capture.set(cv2.CAP_PROP_FOCUS, int(setting.get("focus", 0))))):
            log.warning("%s camera did not accept its focus setting", role)

    OpenCVCamera._configure_capture_settings = configured


def lerobot_arguments(args, spec, bundle: Path) -> list[str]:
    robot = spec.robot
    cameras = robot.camera_config(
        {"overhead": args.overhead_camera, "wrist": args.wrist_camera}, args.fps)
    return [
        f"--server_address={args.server_address}", "--robot.type=so101_follower",
        f"--robot.port={args.robot_port}", f"--robot.id={robot.calibration_id}",
        f"--robot.calibration_dir={bundle / 'robot_client/calibration'}",
        f"--robot.max_relative_target={args.max_relative_target}",
        f"--robot.cameras={cameras}", f"--task={args.task}", f"--policy_type={spec.name}",
        "--pretrained_name_or_path=polima-bundle", "--policy_device=cpu", "--client_device=cpu",
        f"--actions_per_chunk={args.actions_per_chunk}",
        f"--chunk_size_threshold={robot.chunk_threshold}",
        f"--aggregate_fn_name={robot.aggregate_fn}", f"--fps={args.fps}",
    ]


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(f"POLIMA_{name}", default) != "0"


def main(argv: list[str] | None = None) -> int:
    from polima.robot.policy import BundlePolicy

    parser = argparse.ArgumentParser(prog="polima.robot.so101_loop", description=__doc__.split("\n")[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--robot-port", required=True)
    parser.add_argument("--overhead-camera", "--perspective-camera", required=True)
    parser.add_argument("--wrist-camera", required=True)
    parser.add_argument("--server-address")
    parser.add_argument("--actions-per-chunk", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--max-relative-target", type=int)
    parser.add_argument("--task", default=os.environ.get("POLIMA_TASK", ""))
    args = parser.parse_args(argv)

    root = args.bundle.resolve()
    policy = BundlePolicy(root, args.server_address or "127.0.0.1", args.task)
    spec = policy.spec
    if spec.name not in ("act", "groot"):
        print(f"the SO-101 client does not run {spec.display_name} bundles yet", file=sys.stderr)
        return 1
    robot = spec.robot
    args.server_address = policy.client.address
    args.task = policy.task
    args.actions_per_chunk = args.actions_per_chunk or robot.actions_per_chunk
    args.fps = args.fps or robot.default_fps
    args.max_relative_target = args.max_relative_target or robot.max_relative_target

    client = root / "robot_client"
    calibration = client / f"calibration/{robot.calibration_id}.json"
    problems = [f"{label} not found: {path}" for label, path in (
        ("follower arm", args.robot_port), ("overhead camera", args.overhead_camera),
        ("wrist camera", args.wrist_camera), ("rest pose", client / "robot_rest_position.json"),
    ) if not Path(path).exists()]
    if not calibration.is_file():
        problems.append("the arm is not calibrated for this bundle; run `polima robot "
                        "calibrate` or calibrate in Studio first")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1

    SO101Client.rest = load_rest_pose(client / "robot_rest_position.json")
    SO101Client.model = policy
    SO101Client.auto_complete = _flag("AUTO_COMPLETE")
    SO101Client.auto_repeat = _flag("AUTO_REPEAT")
    SO101Client.repeat_pause_s = float(os.environ.get("POLIMA_AUTO_REPEAT_PAUSE", "2"))
    if not policy.client.ping():
        print(f"policy server is not listening at {args.server_address}", file=sys.stderr)
        return 1
    SO101Client.recorder = start_record(root, spec, policy, args, calibration)

    import lerobot.async_inference.robot_client as lerobot_client
    import lerobot.robots.so_follower  # noqa: F401 - registers so101_follower
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    lerobot_client.RobotClient = client_class()
    apply_camera_focus(client / "camera_focus_config.json",
                       {args.overhead_camera: "perspective", args.wrist_camera: "wrist"})

    def graceful(signum, _frame) -> None:
        if SO101Client.active is None:
            raise KeyboardInterrupt
        SO101Client.active.logger.info("Signal %s: returning to rest and stopping", signum)
        SO101Client.stop_requested = True
        SO101Client.active.shutdown_event.set()
        if signum == signal.SIGINT:
            raise KeyboardInterrupt  # unwinds into stop(), which returns to rest first

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, graceful)

    print(f"SO-101 robot client: {spec.display_name} bundle {root.name}\n"
          f"  task {policy.task_id}: {policy.task or '(none)'}\n"
          f"  server {args.server_address} · arm {args.robot_port}\n"
          f"  auto-complete {'on' if SO101Client.auto_complete else 'off'} · "
          f"repeat {'on' if SO101Client.auto_repeat else 'off'}", flush=True)
    sys.argv = [sys.argv[0], *lerobot_arguments(args, spec, root)]
    status, error = "completed", None
    try:
        lerobot_client.async_client()
    except KeyboardInterrupt:
        status = "interrupted"
        print("Robot client stopped; the arm returned to rest.", flush=True)
    except Exception as exc:
        status, error = "failed", str(exc)
        raise
    finally:
        finish_record(SO101Client.recorder, status, error)
    return 0


def start_record(root: Path, spec, policy, args, calibration: Path):
    """Open this run's evidence record; a recording problem never stops the arm."""
    from polima.robot import evidence
    from polima.util.hashing import sha256_file

    try:
        rest = json.loads((root / "robot_client/robot_rest_position.json").read_text())
        recorder = evidence.RunRecorder.create(
            bundle_id=root.name, policy=spec.display_name,
            robot={"type": "SO-101 follower", "calibration_id": spec.robot.calibration_id,
                   "port": args.robot_port},
            task={"id": policy.task_id, "instruction": policy.task},
            cameras={"overhead": args.overhead_camera, "wrist": args.wrist_camera,
                     "fps": args.fps, "fourcc": spec.robot.camera_fourcc},
            calibration={"file": calibration.name, "sha256": sha256_file(calibration),
                         "modified": calibration.stat().st_mtime},
            rest_pose={key: rest.get(key) for key in ("source_dataset", "method")},
            settings={"fps": args.fps, "actions_per_chunk": args.actions_per_chunk,
                      "max_relative_target": args.max_relative_target,
                      "chunk_threshold": spec.robot.chunk_threshold, "server": args.server_address,
                      "auto_complete": SO101Client.auto_complete,
                      "auto_repeat": SO101Client.auto_repeat})
    except OSError as error:
        print(f"Run evidence is not being recorded: {error}", file=sys.stderr)
        return None
    print(f"Run evidence: {recorder.folder}", flush=True)
    return recorder


def finish_record(recorder, status: str, error: str | None) -> None:
    if recorder is None:
        return
    if SO101Client.failure:
        status, error = "failed", SO101Client.failure
    elif SO101Client.stop_requested and status == "completed":
        status = "interrupted"  # stopped by the operator, not by finishing the task
    recorder.finish(status, error)


if __name__ == "__main__":
    raise SystemExit(main())
