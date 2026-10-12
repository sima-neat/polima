"""`polima robot` -- drive a supported arm from a deployed policy.

The SO-101 runs on the Modalix DevKit, next to the policy server.
The arm and cameras are attached to that DevKit; the host only compiles and
deploys bundles.

It runs in the DevKit's `/media/nvme/lerobot` venv, not the polima install:
lerobot pulls torch and opencv, which the DevKit's polima deliberately does not.
`$POLIMA_ROBOT_PYTHON` overrides that, which is how you drive an arm attached to
a host during bring-up.

Subcommands, cheapest first:

    ports       what arm and cameras are attached, and which role each fills
    doctor      the same, plus the policy server and calibration
    preview     show the cameras (on the DevKit)
    calibrate   back up the existing calibration, then lerobot-calibrate
    run         drive the arm (on the DevKit)

`ports` and `doctor` find the arm and cameras for you, so you rarely need to
pass device paths by hand.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from polima.config.loader import load
from polima.policies.registry import get_policy
from polima.robot import devices
from polima.util import table


def needs_capability(argv: list[str]) -> str | None:
    """Gate only commands that unconditionally need a robot stack.

    A bundle run checks its own dependencies when it starts. Help, ports, doctor
    and export-improvement-manifest only read files.
    """
    if "--help" in argv or "-h" in argv:
        return None
    if argv and argv[0] in ("ports", "doctor", "export-improvement-manifest"):
        return None
    return "robot"


def run(argv: list[str], parent: argparse.Namespace | None = None) -> int:
    parser = argparse.ArgumentParser(prog="polima robot", description=__doc__)
    parser.add_argument("--policy", default="act")
    sub = parser.add_subparsers(dest="command", required=True)

    installer = sub.add_parser(
        "install", help="set up this DevKit: LeRobot environment, PoLiMa, runtime")
    installer.add_argument("--script", default=None,
                           help="override scripts/install_polima_modalix.sh")
    installer.add_argument("--json", action="store_true")

    for name, help_text in (
        ("ports", "list the attached arm and cameras"),
        ("doctor", "check everything the control loop needs"),
        ("preview", "show the cameras (on the DevKit)"),
        ("calibrate", "back up the calibration, then run lerobot-calibrate"),
        ("run", "drive the arm (on the DevKit)"),
    ):
        child = sub.add_parser(name, help=help_text)
        child.add_argument("--robot-port", default=None, help="/dev/ttyACM0")
        for role in ("front", "overhead", "wrist"):
            child.add_argument(f"--{role}-camera", default=None,
                               help=f"/dev/v4l/by-id/... for the {role} camera")
        child.add_argument("--json", action="store_true")
        if name in ("run", "doctor"):
            child.add_argument("--server", default=None, help="host:port of the policy server")
        if name == "calibrate":
            child.add_argument("--calibration-dir", default=None,
                               help="directory containing this arm's calibration file")
            child.add_argument("--yes", action="store_true",
                               help="skip the outer recalibration confirmation")

    manifest = sub.add_parser(
        "export-improvement-manifest",
        help="reviewed failed and partial robot runs as one JSON manifest")
    manifest.add_argument("--output", default="-", help="file to write; - for stdout (default)")
    manifest.add_argument("--all", action="store_true", help="include reviewed successes")
    manifest.add_argument("--root", default=None, help="PoLiMa root (default $POLIMA_ROOT)")

    args = parser.parse_args(argv)
    if args.command == "install":
        return _install(args)
    if args.command == "export-improvement-manifest":
        return _export_manifest(args)

    spec = get_policy(args.policy)
    config = load(config_file=getattr(parent, "config", None))

    if args.command == "ports":
        return _ports(spec, args)
    if args.command == "doctor":
        return _doctor(spec, args, config)
    if args.command == "calibrate":
        return _calibrate(spec, args)
    if args.command in ("preview", "run"):
        print(f"polima robot {args.command} runs on the DevKit, where the arm and cameras are "
              f"plugged in.\n  Log in to the DevKit (ssh sima@<devkit-address>) and run it there.",
              file=sys.stderr)
        return 2
    return 2


def _export_manifest(args) -> int:
    import json
    import socket

    from polima.robot.evidence import improvement_manifest

    manifest = improvement_manifest(args.root, args.all, socket.gethostname())
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output == "-":
        sys.stdout.write(text)
    else:
        Path(args.output).write_text(text)
        counts = manifest["counts"]
        print(f"{counts['exported_runs']} of {counts['total_runs_considered']} runs -> {args.output}")
    return 0


# ------------------------------------------------------------------- install


def _install(args) -> int:
    """Provision this DevKit by delegating to scripts/install_polima_modalix.sh.

    Delegates rather than reimplements. That script already installs the
    LeRobot environment (via examples/so101/setup/), the DevKit package and the
    two native binaries; duplicating any of it here would fork that logic. All
    this adds is a
    discoverable front door, because the path to a shell script is not
    something the CLI should expect you to know.
    """
    relative = Path("scripts") / "install_polima_modalix.sh"
    # The DevKit runs polima from a pip install under the LeRobot venv, where
    # __file__ is site-packages/polima/cli/robot.py and walking up finds no
    # repository at all. So try the source checkout AND the deployed DevKit
    # layout, and if neither has it, say where we looked rather than guessing.
    candidates = (
        [Path(args.script)] if args.script else
        [Path(p) for p in [os.environ.get("POLIMA_INSTALLER")] if p] +
        [Path(__file__).resolve().parents[3] / relative,
         Path(os.environ.get("POLIMA_ROOT", "/media/nvme/polima")) / "src" / relative]
    )
    script = next((c for c in candidates if c.is_file()), None)
    if script is None:
        print("polima robot install: no installer found. Looked in:", file=sys.stderr)
        for candidate in candidates:
            print(f"  {candidate}", file=sys.stderr)
        print("  Pass --script, or set POLIMA_INSTALLER.", file=sys.stderr)
        return 2
    if not os.access(script, os.X_OK):
        print(f"polima robot install: {script} is not executable", file=sys.stderr)
        return 2
    # The installer refuses to run anywhere else, but saying so here costs one
    # line and beats a failure 300 lines into a DevKit provision.
    if platform.machine() != "aarch64":
        print(f"polima robot install: this sets up a Modalix DevKit and must run "
              f"on it; this host is {platform.machine()}.\n"
              f"  Deploy from the host instead: polima deploy --bundle <id> --start",
              file=sys.stderr)
        return 2

    print(f"running {script}")
    result = subprocess.run([str(script)], check=False)
    if args.json:
        from polima.util.jsonio import dumps

        print(dumps({"script": str(script), "returncode": result.returncode}), end="")
    return result.returncode


# --------------------------------------------------------------------- ports


def _discover(spec, args):
    cameras = devices.list_cameras()
    ports = devices.list_serial_ports()
    overrides = {
        role: getattr(args, f"{role}_camera", None)
        for role, _ in spec.robot.camera_roles
        if getattr(args, f"{role}_camera", None)
    }
    assigned, problems = devices.match_cameras(
        spec.robot.camera_roles, cameras, spec.robot.camera_hints, overrides
    )
    arm, arm_problem = devices.pick_arm_port(ports, args.robot_port)
    if arm_problem:
        problems.insert(0, arm_problem)
    return cameras, ports, assigned, arm, problems


def _ports(spec, args) -> int:
    cameras, ports, assigned, arm, problems = _discover(spec, args)

    if args.json:
        from polima.util.jsonio import dumps

        print(dumps({
            "arm": arm,
            "cameras": {role: assigned.get(role, "") for role, _ in spec.robot.camera_roles},
            "detected_cameras": [c.name for c in cameras],
            "detected_ports": [p.name for p in ports],
            "problems": problems,
        }), end="")
        return 0 if not problems else 1

    print(table.section("arm"))
    print(table.render([[p.path, "<- selected" if p.path == arm else ""] for p in ports])
          or "  none found (expected /dev/ttyACM*)")

    print(table.section("cameras"))
    by_path = {camera.path: camera for camera in cameras}
    rows = []
    for role, label in spec.robot.camera_roles:
        path = assigned.get(role, "")
        camera = by_path.get(path)
        rows.append([role, label, camera.vendor_model if camera else "-",
                     camera.node if camera else "", path or "unassigned"])
    print(table.render(rows, headers=["role", "label", "device", "node", "by-id path"]))

    unassigned = [c for c in cameras if c.path not in assigned.values()]
    if unassigned:
        print("\n  also present, unused:")
        for camera in unassigned:
            print(f"    {camera.vendor_model}  {camera.path}")

    if problems:
        print()
        for problem in problems:
            print(f"  ! {problem}", file=sys.stderr)
        return 1
    print("\n  cameras resolve by-id, never /dev/videoN -- enumeration order is "
          "not stable across reboots.")
    return 0


# -------------------------------------------------------------------- doctor


def _doctor(spec, args, config) -> int:
    from polima.util import table as ui

    _cameras, _ports, assigned, arm, problems = _discover(spec, args)
    failures = 0

    print(ui.section(f"robot ({spec.name})"))
    print(ui.status(ui.OK if arm else ui.FAIL, "arm", arm or "not found"))
    failures += 0 if arm else 1
    for role, _label in spec.robot.camera_roles:
        path = assigned.get(role, "")
        print(ui.status(ui.OK if path else ui.FAIL, f"camera {role}", path or "not found"))
        failures += 0 if path else 1

    print(ui.section("robot environment"))
    from polima.robot import env as robot_env

    found = robot_env.discover()
    if found is None:
        print(ui.status(ui.FAIL, "interpreter", "none found with lerobot; tried "
                        + ", ".join(origin for _, origin in robot_env.candidates())))
        failures += 1
    else:
        print(ui.status(ui.OK if found.usable else ui.FAIL, "interpreter",
                        f"{found.origin} (py{found.version})"))
        print(ui.status(ui.OK if found.lerobot else ui.FAIL, "  lerobot",
                        found.lerobot or "missing -- the robot stack"))
        print(ui.status(ui.OK if found.cv2 else ui.FAIL, "  opencv",
                        found.cv2 or "missing -- cameras"))
        # flask only backs the browser view; the control loop runs without it.
        print(ui.status(ui.OK if found.flask else ui.WARN, "  flask",
                        found.flask or "absent -- no live view"))
        if not found.usable:
            failures += 1

    print(ui.section("calibration"))
    calibration = _calibration_file(spec)
    print(ui.status(ui.OK if calibration.is_file() else ui.WARN, "calibration",
                    str(calibration) if calibration.is_file()
                    else "absent -- run `polima robot calibrate`"))

    print(ui.section("policy server"))
    from polima.wire.client import wait_for_port

    host, _, port_text = (args.server or "").partition(":")
    host = host or "127.0.0.1"
    port = int(port_text) if port_text else spec.wire.default_port
    reachable = wait_for_port(host, port, timeout=2.0)
    print(ui.status(ui.OK if reachable else ui.FAIL, "server", f"{host}:{port}"
                    + ("" if reachable else " -- not accepting connections")))
    failures += 0 if reachable else 1

    print()
    for problem in problems:
        print(f"  ! {problem}")
    print(f"\n{'ready' if failures == 0 else f'{failures} blocking problem(s)'}")
    return 0 if failures == 0 else 1


def robot_python() -> str | None:
    from polima.robot import env as robot_env

    found = robot_env.discover()
    return found.python if found and found.usable else None


def _calibration_file(spec, calibration_dir: str | None = None) -> Path:
    if calibration_dir:
        return Path(calibration_dir).expanduser() / f"{spec.robot.calibration_id}.json"
    return (Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"
            / "robots" / "so_follower" / f"{spec.robot.calibration_id}.json")


# ----------------------------------------------------------------- calibrate


def _calibrate(spec, args) -> int:
    """Back up first. A bad calibration run otherwise overwrites a working one
    with no way back, and the arm then moves incorrectly."""
    if not spec.robot.supports_calibrate:
        print(f"{spec.name} does not support calibration", file=sys.stderr)
        return 2

    import fcntl

    lock = Path(os.environ.get("POLIMA_ROOT", "/media/nvme/polima")) / "var/run/control.lock"
    try:
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open("a+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return _calibrate_arm(spec, args, stream.fileno())
    except BlockingIOError:
        print("  another PoLiMa controller is active; stop it before calibrating", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"  cannot calibrate: {exc}", file=sys.stderr)
        return 1


def _calibrate_arm(spec, args, lock_fd: int) -> int:
    _, _, _, arm, problems = _discover(spec, args)
    if not arm:
        for problem in problems:
            print(f"  ! {problem}", file=sys.stderr)
        return 1

    calibration = _calibration_file(spec, args.calibration_dir)
    try:
        calibration.parent.mkdir(parents=True, exist_ok=True)
        if calibration.is_file():
            stamp = f"{time.strftime('%Y%m%d_%H%M%S')}.{time.time_ns()}"
            backup = calibration.with_suffix(f".json.bak.{stamp}")
            shutil.copy2(calibration, backup)
            print(f"  backed up {calibration.name} -> {backup.name}")
        else:
            print(f"  no existing calibration at {calibration}")
    except OSError as exc:
        print(f"  cannot prepare calibration: {exc}", file=sys.stderr)
        return 1

    if not args.yes:
        answer = input(f"  recalibrate {spec.robot.calibration_id} on {arm}? [y/N] ").strip().lower()
        if not answer.startswith("y"):
            print("  cancelled")
            return 0

    python = robot_python()
    if python is None:
        print("  no interpreter with lerobot; run `polima robot doctor`",
              file=sys.stderr)
        return 1
    binary = Path(python).with_name("lerobot-calibrate")
    if not binary.exists():
        print(f"  {binary} not found", file=sys.stderr)
        return 1
    env = dict(os.environ)
    env.pop("LEROBOT_FORCE_PROVIDED_CALIBRATION", None)
    return subprocess.run([
        str(binary),
        "--robot.type=so101_follower",
        f"--robot.port={arm}",
        f"--robot.id={spec.robot.calibration_id}",
        f"--robot.calibration_dir={calibration.parent}",
    ], check=False, env=env, pass_fds=(lock_fd,)).returncode
