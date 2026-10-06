"""Connect a compiled GR00T or SmolVLA bundle to the established ACT SO101 control loop.

This module owns model selection and transport only. Motor initialization,
camera acquisition, timed action queues, rest handling and shutdown remain in
the existing packaged run_robot_client_with_live_view.py.

`python -m polima.robot.so101` is every SO-101 bundle's launcher; an ACT bundle
is handed to PoLiMa's own loop, polima.robot.so101_loop.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import time
from pathlib import Path

import numpy as np

from polima.bundle.contract import spec_from_bundle
from polima.bundle.layout import Bundle
from polima.policies.registry import get_policy
from polima.robot import arm_gain, crossfade, reset_ease, smoothing, substeps
from polima.robot.policy import CompiledPolicyAdapter
from polima.util.jsonio import read_json

_TRIAL_METRICS = None
# LeRobot's registered names plus Polima's position-aware hand-off.
AGGREGATES = ("weighted_average", "latest_only", "average", "conservative", crossfade.NAME)

#: Policies whose compiled bundles this transport can drive.
SUPPORTED_POLICIES = ("groot", "smolvla")

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
        raise ValueError("This launcher requires a two-camera SO101 GR00T or SmolVLA bundle")
    if spec.name == "smolvla" and tuple(spec.robot.camera_keys) != ("overhead", "wrist"):
        # SmolVLA has no per-camera token windows: wire slot image0 is simply
        # the checkpoint's camera1, which every SO101 SmolVLA checkpoint maps
        # from the overhead view. A wrist-first bundle runs and reaches for the
        # wrong place, so refuse it rather than drive the arm.
        raise ValueError(
            "SO101 SmolVLA bundle must send overhead as image0 and wrist as image1; "
            f"this bundle orders cameras {tuple(spec.robot.camera_keys)}. Repack it."
        )
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


class SO101PolicyClient:
    """The existing ACT client's predict(overhead, wrist, state) interface."""

    def __init__(self, address: str, stats=None, timeout: float = 30.0):
        # stats is accepted for compatibility with the established loop. GR00T
        # normalization/action decoding already runs in the compiled plan.
        root = Path(os.environ["POLIMA_SO101_BUNDLE"])
        bundle, spec, task_id = load_contract(root, os.environ.get("POLIMA_TASK_ID", "0"))
        self.adapter = CompiledPolicyAdapter(
            spec, bundle, address, task_id=task_id, timeout=timeout,
            noise=os.environ.get("POLIMA_GROOT_NOISE", "random"),
        )
        if _TRIAL_METRICS is not None:
            self.adapter.client.trace_event = _TRIAL_METRICS.record
        self.host = self.adapter.client.host
        self.port = self.adapter.client.port

    def predict(self, overhead, wrist, state):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (6,) or not np.isfinite(state).all():
            raise ValueError("Expected six finite SO101 joint positions")
        observation = dict(zip(JOINTS, state))
        # The control loop supplies named views; the bundle determines wire order.
        views = {"front": overhead, "overhead": overhead, "wrist": wrist}
        observation.update((role, views[role]) for role in self.adapter.spec.robot.camera_keys)
        started = time.perf_counter()
        actions, latency = self.adapter.predict(observation)
        if _TRIAL_METRICS is not None:
            _TRIAL_METRICS.record("prediction", request_id=self.adapter.client.request_id, server_compute_ms=float(latency),
                                  prediction_ms=(time.perf_counter() - started) * 1000)
        expected = tuple(self.adapter.spec.wire.response_shape)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite actions shaped {expected}, got {actions.shape}")
        return actions, latency


#: Where `polima robot calibrate` writes this DevKit's SO-101 calibration.
DEVKIT_CALIBRATION = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower"


def arm_calibration_dir(robot_dir: Path, devkit: Path = DEVKIT_CALIBRATION) -> Path:
    """The bundle's own arm calibration (set in Studio), else the DevKit's; never none."""
    for folder in (robot_dir / "calibration", devkit):
        if (folder / "so-arm101.json").is_file():
            return folder
    raise FileNotFoundError("No SO-101 arm calibration on this DevKit: run `polima robot calibrate` "
                            "(or calibrate in Studio) before driving the arm")


def client_configuration(root, spec, task_id, instruction, args):
    """Build the same LeRobot launch contract used by the ACT SOM launcher."""
    robot_dir = root / "robot_client"
    rest = read_json(robot_dir / "robot_rest_position.json")
    if set(rest.get("positions", {})) != set(JOINTS):
        raise ValueError("Bundle rest pose must contain all six SO101 joints")
    if not all(np.isfinite(float(v)) for v in rest["positions"].values()):
        raise ValueError("Bundle rest pose contains nonfinite positions")
    calibration_dir = arm_calibration_dir(robot_dir)
    actions = (
        spec.robot.actions_per_chunk if args.actions_per_chunk is None else args.actions_per_chunk
    )
    fps = spec.robot.default_fps if args.fps is None else args.fps
    if not 1 <= actions <= spec.robot.actions_per_chunk or fps <= 0:
        raise ValueError("Invalid action horizon or control rate")
    cameras = {
        role: {
            "type": "opencv",
            "index_or_path": path,
            "width": 640,
            "height": 480,
            # Control frequency must not change camera capture modes.
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
        # Compatibility settings consumed by the unmodified ACT control loop.
        "ACT_SOM_DIRECT": "1",
        "ACT_SOM_STATS": str(root / "fixtures/normalization_stats.npz"),
        "ACT_REST_POSITION": str(robot_dir / "robot_rest_position.json"),
        "ACT_CAMERA_FOCUS_CONFIG": str(robot_dir / "camera_focus_config.json"),
        "ACT_PERSPECTIVE_CAMERA": args.front_camera,
        "ACT_WRIST_CAMERA": args.wrist_camera,
    }
    threshold = float(os.environ.get("POLIMA_CHUNK_THRESHOLD", "0.5"))
    if not 0 <= threshold <= 1:
        raise ValueError("POLIMA_CHUNK_THRESHOLD must be between 0 and 1")
    smooth_moves = smoothing.moves_from_env()
    # Validated here, before anything touches the robot.
    smoothing.shape_from_env()
    substeps.substeps_from_env()
    reset_ease.ease_from_env()
    arm_gain.gain_from_env()
    aggregate = os.environ.get("POLIMA_AGGREGATE", "latest_only" if smooth_moves else "weighted_average")
    if aggregate not in AGGREGATES:
        raise ValueError(f"POLIMA_AGGREGATE must be one of {sorted(AGGREGATES)}")
    if smooth_moves and aggregate != "latest_only":
        # An overlap blend splits the switch; its remainder would land after the smoothing.
        raise ValueError(f"{smoothing.ENV} replaces the overlap blend; leave POLIMA_AGGREGATE "
                         "unset or set it to latest_only")
    argv = [
        str(robot_dir / "run_robot_client_with_live_view.py"),
        f"--server_address={args.server_address}",
        "--robot.type=so101_follower",
        f"--robot.port={args.robot_port}",
        "--robot.id=so-arm101",
        f"--robot.calibration_dir={calibration_dir}",
        "--robot.use_degrees=true",
        f"--robot.max_relative_target={args.max_relative_target}",
        f"--robot.cameras={json.dumps(cameras)}",
        f"--task={instruction}",
        # Direct-SOM overrides the model client; preserve the known LeRobot
        # config discriminator without requesting its GR00T model loader.
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


def _bundle_policy(argv) -> str | None:
    """The policy of the bundle named by --bundle, read before any other option."""
    for index, item in enumerate(argv):
        if item.startswith("--bundle="):
            value = item.split("=", 1)[1]
        elif item == "--bundle" and index + 1 < len(argv):
            value = argv[index + 1]
        else:
            continue
        try:
            return read_json(Path(value) / "bundle.json").get("policy")
        except (OSError, ValueError):
            return None
    return None


def main(argv=None):
    if _bundle_policy(sys.argv[1:] if argv is None else argv) == "act":
        # ACT bundles run PoLiMa's own SO-101 loop; the path below is
        # GR00T/SmolVLA through the established ACT control loop.
        from polima.robot.so101_loop import main as act_loop

        return act_loop(argv)
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
    args = parser.parse_args(argv)
    root = args.bundle.resolve()
    bundle, spec, task_id = load_contract(root, os.environ.get("POLIMA_TASK_ID", "0"))
    env, client_args = client_configuration(root, spec, task_id, bundle.tasks[task_id], args)
    for name, value in env.items():
        os.environ[name] = value
    if os.environ.get("POLIMA_NO_AUTO_COMPLETE"):
        os.environ["ACT_AUTO_COMPLETE"] = "0"
    aggregate = next(a.split("=", 1)[1] for a in client_args if a.startswith("--aggregate_fn_name="))
    smooth_moves = smoothing.moves_from_env()
    substep_count = substeps.substeps_from_env()
    ease = reset_ease.ease_from_env()
    gain = arm_gain.gain_from_env()
    substep_state = None
    if aggregate == crossfade.NAME or smooth_moves or substep_count > 1:
        # Before the bundle script runs: it captures and wraps the merge, the action
        # conversion, control_loop_action and __init__ at import.
        from lerobot.async_inference.robot_client import RobotClient
        if aggregate == crossfade.NAME:
            import lerobot.async_inference.configs as lerobot_configs
            crossfade.install(lerobot_configs, RobotClient)
        if smooth_moves:
            smoothing.install(RobotClient, smooth_moves, shape=smoothing.shape_from_env())
        if substep_count > 1:
            substep_state = substeps.install(RobotClient, substep_count)
    if gain is not None:
        # The follower connects (and configures) inside the bundle script's client.
        from lerobot.robots.so_follower import SOFollower
        arm_gain.install(SOFollower, gain)
    # Identical robot lifecycle; swap only the transport imported by that loop.
    sys.path.insert(0, str(root / "robot_client"))
    sys.argv = client_args
    print(f"{spec.name} SO101: {bundle.bundle_id}; task {task_id}: {bundle.tasks[task_id]}; "
          f"cameras {list(spec.robot.camera_keys)}", flush=True)
    def run():
        restore = None
        if ease != "linear":
            # After run_instrumented set its entry point (so the action trace's
            # return-to-rest hook wraps the eased function); before the script imports it.
            import lerobot.async_inference.robot_client as robot_client_module
            restore = reset_ease.install(robot_client_module, ease)
        try:
            return runpy.run_path(client_args[0], run_name="__main__")
        finally:
            if restore is not None:
                restore()
    metrics_dir = os.environ.get("POLIMA_METRICS_DIR")
    if metrics_dir:
        # The transport shim imports the canonical module even when this file
        # is launched with -m and therefore executes as __main__.
        import polima.robot.so101 as transport
        from polima.robot.metrics import TrialMetrics, run_instrumented
        metrics = TrialMetrics(metrics_dir, {
            "bundle": bundle.bundle_id, "mode": "async",
            "fps": spec.robot.default_fps if args.fps is None else args.fps,
            "actions_per_chunk": spec.robot.actions_per_chunk if args.actions_per_chunk is None else args.actions_per_chunk,
            "chunk_threshold": float(os.environ.get("POLIMA_CHUNK_THRESHOLD", "0.5")),
            "max_relative_target": args.max_relative_target,
            "aggregate_fn_name": aggregate,
            "smooth_moves": smooth_moves,
            "smooth_shape": smoothing.shape_from_env(),
            "substeps": substep_count,
            "reset_ease": ease,
            "arm_p": gain,
            "groot_noise": os.environ.get("POLIMA_GROOT_NOISE", "random"),
            "task_id": task_id, "camera_order": list(spec.robot.camera_keys),
            "server_address": args.server_address,
            "camera_capture_fps": {
                role: config["fps"] for role, config in json.loads(next(
                    arg.split("=", 1)[1] for arg in client_args
                    if arg.startswith("--robot.cameras=")
                )).items()
            },
        })
        transport._TRIAL_METRICS = metrics
        if substep_state is not None:
            substep_state.record = metrics.record
        try:
            run_instrumented(run, metrics)
        finally:
            transport._TRIAL_METRICS = None
            if substep_state is not None:
                substep_state.record = None
    else:
        run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
