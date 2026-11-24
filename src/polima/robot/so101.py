"""Connect ACT and GR00T bundles to the SO-101 control client."""

from __future__ import annotations

import argparse
import json
import logging
import os
import runpy
import sys
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np

from polima.bundle.contract import spec_from_bundle
from polima.bundle.layout import Bundle
from polima.policies.registry import get_policy
from polima.robot import crossfade, reset_ease, smoothing, substeps
from polima.robot.policy import CompiledPolicyAdapter
from polima.util.jsonio import read_json

AGGREGATES = ("weighted_average", "latest_only", "average", "conservative", crossfade.NAME)


SUPPORTED_POLICIES = ("groot", "act", "pi05")


ACT_INSTRUCTION = "ACT (no language input)"


RECIPES = {
    "groot": {"fps": 15, "POLIMA_SMOOTH_MOVES": "5", "POLIMA_SUBSTEPS": "2",
              "POLIMA_CHUNK_THRESHOLD": "0.5", "POLIMA_GROOT_NOISE": "zero",
              "POLIMA_AUTO_COMPLETE": "0"},
    "act": {"fps": 30, "POLIMA_SMOOTH_MOVES": "10", "POLIMA_SUBSTEPS": "1",
            "POLIMA_CHUNK_THRESHOLD": "0.5"},
    # Pi0.5: overlap plans and smooth arm targets within each plan.
    "pi05": {"fps": 30, "POLIMA_SMOOTH_MOVES": "0", "POLIMA_SUBSTEPS": "1",
             "POLIMA_CHUNK_THRESHOLD": "0.5", "POLIMA_GROOT_NOISE": "fixed",
             "POLIMA_CHUNK_FILTER": "fir3", "POLIMA_AGGREGATE": "crossfade",
             "POLIMA_AUTO_COMPLETE": "0"},
}

COMMON_RECIPE = {
    smoothing.SHAPE_ENV: "smoothstep",
    reset_ease.ENV: "smoothstep",
}


#: Upper bounds a run accepts: above these the per-move safety clamp stops protecting the arm.
MAX_RELATIVE_TARGET = 30.0
MAX_FPS = 60

FAST_ENV = "POLIMA_FAST"
FAST_RECIPE = {"fps": 30, "POLIMA_SMOOTH_MOVES": "10", "POLIMA_SUBSTEPS": "1",
               "POLIMA_CHUNK_THRESHOLD": "0.6"}


def fast_from_env(environ=os.environ) -> bool:
    value = environ.get(FAST_ENV, "1").strip().lower()
    if value in ("0", "false", "no"):
        return False
    if value in ("", "1", "true", "yes"):
        return True
    raise ValueError(f"{FAST_ENV} must be 1 or 0, not {value!r}")


def policy_recipe(spec, environ=os.environ) -> dict:
    """The policy's recipe, with the fast settings on top when --fast is on."""
    return {**RECIPES[spec.name], **(FAST_RECIPE if spec.name == "groot" and fast_from_env(environ) else {})}


JOINTS = tuple(
    f"{name}.pos"
    for name in (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )
)


def load_contract(root: Path, task: str | int = 0):
    bundle = Bundle.from_dict(read_json(root / "bundle.json"), root)
    spec = spec_from_bundle(get_policy(bundle.policy), bundle)
    if (
        spec.name not in SUPPORTED_POLICIES
        or spec.robot.calibration_id != "so-arm101"
        or tuple(spec.robot.joint_names) != JOINTS
        or spec.dataset.state_dim != 6
        or spec.dataset.action_dim != 6
        or len(spec.robot.camera_keys) != 2
        or set(spec.robot.camera_keys) not in ({"front", "wrist"}, {"overhead", "wrist"})
    ):
        raise ValueError("This launcher requires a two-camera SO101 GR00T, ACT or Pi0.5 bundle")
    if spec.name == "act" and not bundle.tasks:

        if str(task) != "0":
            raise ValueError(f"ACT has no task catalog; task must be 0, not {task!r}")
        return bundle, spec, 0
    if not bundle.tasks:
        raise ValueError(
            f"{spec.name} bundle names no task; repack it so the robot client can "
            "report which instruction its compiled constants encode"
        )
    try:
        task_id = int(task)
    except ValueError:
        if str(task) not in bundle.tasks:
            raise ValueError(f"Unknown task {task!r}; available: {bundle.tasks}") from None
        task_id = bundle.tasks.index(str(task))
    if not 0 <= task_id < len(bundle.tasks):
        raise ValueError(f"Task {task_id} is outside the compiled task catalog")
    return bundle, spec, task_id


def recipe_environment(spec, environ=os.environ) -> dict[str, str]:
    """The recipe's environment defaults for this policy (no control rate)."""
    recipe = {**COMMON_RECIPE, **policy_recipe(spec, environ)}
    recipe.pop("fps")
    return recipe


def apply_recipe(spec, environ=os.environ) -> dict[str, str]:
    """Fill in every recipe setting the caller did not set; return what was filled.

    A variable set to an empty string counts as not set (the policy default applies).
    """
    applied = {
        name: value for name, value in recipe_environment(spec, environ).items()
        if not environ.get(name, "").strip()
    }
    environ.update(applied)
    return applied


def control_fps(spec, args, environ=os.environ) -> int:
    """Commands per second: --fps when given, otherwise the recipe rate (--fast: 30)."""
    return policy_recipe(spec, environ)["fps"] if args.fps is None else args.fps


def instruction(bundle, task_id: int) -> str:
    """The instruction the bundle's compiled constants encode (a label for ACT)."""
    return bundle.tasks[task_id] if bundle.tasks else ACT_INSTRUCTION


class SO101PolicyClient:
    """Adapt the control client's camera and joint observations to the bundle."""

    def __init__(self, address: str, stats=None, timeout: float = 30.0):

        root = Path(os.environ["POLIMA_SO101_BUNDLE"])
        bundle, spec, task_id = load_contract(root, os.environ.get("POLIMA_TASK_ID", "0"))
        self.adapter = CompiledPolicyAdapter(
            spec, bundle, address, task_id=task_id, timeout=timeout,
            noise=(os.environ.get("POLIMA_GROOT_NOISE", "").strip() or "random"),
        )
        self.chunk_filter = smoothing.chunk_filter_from_env()
        if spec.name == "pi05":
            # Pi0.5 reads its instruction as text, so any instruction works; the
            # bundle's compiled-in task is only the default.
            self.adapter.task = (os.environ.get("POLIMA_PI05_TASK", "").strip()
                                 or instruction(bundle, task_id))
        self.host = self.adapter.client.host
        self.port = self.adapter.client.port

    def predict(self, overhead, wrist, state):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (6,) or not np.isfinite(state).all():
            raise ValueError("Expected six finite SO101 joint positions")
        observation = dict(zip(JOINTS, state))

        views = {"front": overhead, "overhead": overhead, "wrist": wrist}
        observation.update((role, views[role]) for role in self.adapter.spec.robot.camera_keys)
        actions, latency = self.adapter.predict(observation)
        expected = tuple(self.adapter.spec.wire.response_shape)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite actions shaped {expected}, got {actions.shape}")
        return smoothing.filter_chunk(actions, self.chunk_filter), latency


DEVKIT_CALIBRATION = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower"

def arm_calibration_dir(robot_dir: Path, devkit: Path | None = None) -> Path:
    """The bundle's own arm calibration (set in Studio), else the DevKit's; never none."""
    for folder in (robot_dir / "calibration", devkit or DEVKIT_CALIBRATION):
        if (folder / "so-arm101.json").is_file():
            return folder
    raise FileNotFoundError("No SO-101 arm calibration on this DevKit: run `polima robot calibrate` "
                            "(or calibrate in Studio) before driving the arm")

def client_configuration(root, spec, task_id, instruction, args):
    """Build the LeRobot launch configuration from the bundle contract."""
    robot_dir = root / "robot_client"
    rest = read_json(robot_dir / "robot_rest_position.json")
    if set(rest.get("positions", {})) != set(JOINTS):
        raise ValueError("Bundle rest pose must contain all six SO101 joints")
    if not all(np.isfinite(float(v)) for v in rest["positions"].values()):
        raise ValueError("Bundle rest pose contains nonfinite positions")
    calibration = arm_calibration_dir(robot_dir)
    actions = (
        spec.robot.actions_per_chunk if args.actions_per_chunk is None else args.actions_per_chunk
    )
    fps = control_fps(spec, args)
    if not 1 <= actions <= spec.robot.actions_per_chunk:
        raise ValueError(
            f"--actions-per-chunk must be between 1 and {spec.robot.actions_per_chunk}")
    if not 1 <= fps <= MAX_FPS:
        raise ValueError(f"--fps must be between 1 and {MAX_FPS}")
    if not (np.isfinite(args.max_relative_target)
            and 0 < args.max_relative_target <= MAX_RELATIVE_TARGET):
        raise ValueError(f"--max-relative-target must be greater than 0 and at most "
                         f"{MAX_RELATIVE_TARGET} degrees")
    cameras = {
        role: {
            "type": "opencv",
            "index_or_path": path,
            "width": 640,
            "height": 480,

            "fps": spec.robot.camera_capture_fps.get(
                role, spec.robot.camera_capture_fps.get(
                    "front" if role == "overhead" else role, spec.robot.default_fps
                )
            ),
            "fourcc": spec.robot.camera_fourcc,
        }
        for role, path in [("overhead", args.front_camera), ("wrist", args.wrist_camera)]
    }
    env = {
        "POLIMA_SO101_BUNDLE": str(root),
        "POLIMA_TASK_ID": str(task_id),

        "ACT_SOM_DIRECT": "1",
        "ACT_SOM_STATS": str(root / "fixtures/normalization_stats.npz"),
        "ACT_REST_POSITION": str(robot_dir / "robot_rest_position.json"),
        "ACT_CAMERA_FOCUS_CONFIG": str(robot_dir / "camera_focus_config.json"),
        "ACT_PERSPECTIVE_CAMERA": args.front_camera,
        "ACT_WRIST_CAMERA": args.wrist_camera,
    }
    threshold = float(os.environ.get("POLIMA_CHUNK_THRESHOLD", "").strip() or "0.5")
    if not 0 <= threshold <= 1:
        raise ValueError("POLIMA_CHUNK_THRESHOLD must be between 0 and 1")
    smooth_moves = smoothing.moves_from_env()

    smoothing.shape_from_env()
    smoothing.chunk_filter_from_env()
    substeps.substeps_from_env()
    reset_ease.ease_from_env()
    aggregate = (os.environ.get("POLIMA_AGGREGATE", "").strip()
                 or ("latest_only" if smooth_moves else "weighted_average"))
    if aggregate not in AGGREGATES:
        raise ValueError(f"POLIMA_AGGREGATE must be one of {sorted(AGGREGATES)}")
    if smooth_moves and aggregate != "latest_only":

        raise ValueError(f"{smoothing.ENV} replaces the overlap blend; leave POLIMA_AGGREGATE "
                         "unset or set it to latest_only")
    argv = [
        str(Path(__file__).parent / "assets/so101/run_robot_client_with_live_view.py"),
        f"--server_address={args.server_address}",
        "--robot.type=so101_follower",
        f"--robot.port={args.robot_port}",
        "--robot.id=so-arm101",
        f"--robot.calibration_dir={calibration}",
        "--robot.use_degrees=true",
        f"--robot.max_relative_target={args.max_relative_target}",
        f"--robot.cameras={json.dumps(cameras)}",
        f"--task={instruction}",

        "--policy_type=act",
        "--pretrained_name_or_path=modalix-direct",
        "--policy_device=cpu",
        "--client_device=cpu",
        f"--actions_per_chunk={actions}",
        f"--chunk_size_threshold={threshold}",
        f"--aggregate_fn_name={aggregate}",
        f"--fps={fps}",
    ]
    return env, argv


_RECORDER = None
_RUN_STATUS = "completed"


def record(kind: str, **fields) -> None:
    """Record telemetry without changing the control loop on storage errors."""
    global _RUN_STATUS
    if kind == "action_sent" and not fields.get("positions"):
        return
    if kind in ("error", "camera_fault"):
        _RUN_STATUS = "failed"
    elif kind == "operator_stop" and _RUN_STATUS != "failed":
        _RUN_STATUS = "interrupted"
    if _RECORDER is not None:
        try:
            _RECORDER.event(kind, **fields)
        except OSError:
            pass


def bound_queue_history(client) -> None:
    """Keep only the recent queue-size samples used by LeRobot's optional plot."""
    client.action_queue_size = deque(maxlen=900)


def configure_run_logging(logger=None) -> None:
    """Replace LeRobot's unlimited debug file with a bounded warning/error log."""
    logger = logging.getLogger() if logger is None else logger
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()
    folder = Path(os.environ.get("POLIMA_ROOT", "/media/nvme/polima")) / "var/log"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(folder / "robot-errors.log", maxBytes=2 * 1024 * 1024,
                                      backupCount=2, encoding="utf-8")
    except OSError as error:
        print(f"Error log unavailable: {error}", file=sys.stderr)
        return
    handler.setLevel(logging.WARNING)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)


def start_record(root, bundle, spec, task_id, args):
    if os.environ.get("POLIMA_RECORD_RUN", "0") != "1":
        return None
    from polima.robot.evidence import RunRecorder
    from polima.util.hashing import sha256_file

    calibration = arm_calibration_dir(root / "robot_client") / "so-arm101.json"
    try:
        return RunRecorder.create(
            bundle_id=bundle.bundle_id, policy=spec.display_name,
            robot={"type": "SO-101 follower", "calibration_id": spec.robot.calibration_id,
                   "port": args.robot_port},
            task={"id": task_id, "instruction": instruction(bundle, task_id)},
            cameras={"overhead": args.front_camera, "wrist": args.wrist_camera,
                     "capture_fps": dict(spec.robot.camera_capture_fps)},
            calibration={"file": calibration.name, "sha256": sha256_file(calibration)},
            settings={"fps": control_fps(spec, args),
                      "actions_per_chunk": args.actions_per_chunk or spec.robot.actions_per_chunk,
                      "max_relative_target": args.max_relative_target,
                      "auto_complete": os.environ.get("ACT_AUTO_COMPLETE", "1") != "0",
                      "auto_repeat": os.environ.get("ACT_AUTO_REPEAT", "1") != "0",
                      **recipe_environment(spec)})
    except OSError as error:
        print(f"Run recording unavailable: {error}", file=sys.stderr)
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--robot-port", required=True)
    parser.add_argument(
        "--front-camera", "--perspective-camera", "--overhead-camera", required=True
    )
    parser.add_argument("--wrist-camera", required=True)
    parser.add_argument("--server-address", required=True)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--actions-per-chunk", type=int)
    parser.add_argument("--max-relative-target", type=float, default=12)
    speed = parser.add_mutually_exclusive_group()
    speed.add_argument("--fast", action="store_true", help="30 commands/s (default)")
    speed.add_argument("--slow", action="store_true", help="15 commands/s")
    args = parser.parse_args(argv)
    if args.fast:
        os.environ[FAST_ENV] = "1"
    if args.slow or args.fps == 15:
        os.environ[FAST_ENV] = "0"
    fast_from_env()
    root = args.bundle.resolve()
    bundle, spec, task_id = load_contract(root, os.environ.get("POLIMA_TASK_ID", "0"))
    applied = apply_recipe(spec)
    env, client_args = client_configuration(root, spec, task_id, instruction(bundle, task_id), args)
    for name, value in env.items():
        os.environ[name] = value
    for name in ("AUTO_COMPLETE", "AUTO_REPEAT", "AUTO_REPEAT_PAUSE"):
        if f"POLIMA_{name}" in os.environ:
            os.environ[f"ACT_{name}"] = os.environ[f"POLIMA_{name}"]
    if os.environ.get("POLIMA_NO_AUTO_COMPLETE"):
        os.environ["ACT_AUTO_COMPLETE"] = "0"
    aggregate = next(
        (
            value.split("=", 1)[1]
            for value in client_args
            if value.startswith("--aggregate_fn_name=")
        ),
        os.environ.get("POLIMA_AGGREGATE", "weighted_average"),
    )
    smooth_moves = smoothing.moves_from_env()
    substep_count = substeps.substeps_from_env()
    ease = reset_ease.ease_from_env()
    if aggregate == crossfade.NAME or smooth_moves or substep_count > 1:
        from lerobot.async_inference.robot_client import RobotClient
        if aggregate == crossfade.NAME:
            import lerobot.async_inference.configs as lerobot_configs
            crossfade.install(lerobot_configs, RobotClient)
        if smooth_moves:
            smoothing.install(RobotClient, smooth_moves, shape=smoothing.shape_from_env())
        if substep_count > 1:
            substeps.install(RobotClient, substep_count)

    sys.path.insert(0, str(root / "robot_client"))
    sys.argv = client_args
    print(f"{spec.name} SO101: {bundle.bundle_id}; task {task_id}: {instruction(bundle, task_id)}; "
          f"cameras {list(spec.robot.camera_keys)}", flush=True)
    print(f"{control_fps(spec, args)} commands/s; "
          + ", ".join(f"{name}={os.environ[name]}" for name in recipe_environment(spec))
          + (f" (recipe defaults: {', '.join(sorted(applied))})" if applied else ""), flush=True)
    def run():
        restore = None
        if ease != "linear":

            import lerobot.async_inference.robot_client as robot_client_module
            restore = reset_ease.install(robot_client_module, ease)
        try:
            return runpy.run_path(client_args[0], run_name="__main__")
        finally:
            if restore is not None:
                restore()
    import polima.robot.so101 as transport
    transport._RECORDER = start_record(root, bundle, spec, task_id, args)
    transport._RUN_STATUS = "completed"
    try:
        run()
    except BaseException as error:
        transport.record("operator_stop" if isinstance(error, KeyboardInterrupt) else "error",
                         message=str(error))
        raise
    finally:
        if transport._RECORDER is not None:
            try:
                transport._RECORDER.finish(transport._RUN_STATUS)
            except OSError as error:
                print(f"Run summary unavailable: {error}", file=sys.stderr)
            transport._RECORDER = None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
