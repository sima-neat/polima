"""Manage or launch the SOM-resident PoLiMa Studio web cockpit."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

from polima.util.net import lan_address

SERVICE = "polima-studio.service"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polima studio",
        description="manage the PoLiMa Studio web service",
    )
    sub = parser.add_subparsers(dest="action")
    for action in ("status", "start", "stop", "restart", "enable", "disable", "open"):
        sub.add_parser(action)
    logs = sub.add_parser("logs")
    logs.add_argument("-f", "--follow", action="store_true")
    serve = sub.add_parser("serve", help="run Studio in the foreground")
    serve.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def _service_url() -> str:
    host = os.environ.get("POLIMA_STUDIO_HOST", "0.0.0.0")
    port = os.environ.get("POLIMA_STUDIO_PORT", "8080")
    public = os.environ.get("POLIMA_STUDIO_PUBLIC_URL")
    if public:
        return public.rstrip("/")
    host = lan_address(host)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"


def _health_url() -> str:
    port = os.environ.get("POLIMA_STUDIO_PORT", "8080")
    return f"http://127.0.0.1:{port}/api/v1/health"


def _wait_until_ready(timeout_s: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(_health_url(), timeout=1.0) as response:
                if response.status == 200:
                    return True
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.25)
    return False


def _systemctl(args: list[str], *, mutate: bool, dry_run: bool = False) -> int:
    command = ["systemctl", *args, SERVICE]
    if mutate and os.geteuid() != 0:
        command = ["sudo", "-n", *command]
    if dry_run:
        print("+ " + " ".join(command))
        return 0
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        for output, stream in ((result.stdout, sys.stdout), (result.stderr, sys.stderr)):
            if isinstance(output, str) and output:
                print(output, end="", file=stream)
        if result.returncode and isinstance(result.stderr, str) and "not found" in result.stderr:
            print("Studio service is not installed. From your PoLiMa checkout on the DevKit, "
                  "run ./scripts/install_polima_modalix.sh, then polima studio start. "
                  "See docs/install.md for prerequisites. No installation was performed.", file=sys.stderr)
        return result.returncode
    except FileNotFoundError as exc:
        print(f"polima studio: {exc.filename} is not installed", file=sys.stderr)
        return 2


def _status() -> int:
    if shutil.which("systemctl") is None:
        print("polima studio: systemd is not available", file=sys.stderr)
        return 2
    active = subprocess.run(
        ["systemctl", "is-active", SERVICE], capture_output=True, text=True, check=False
    ).stdout.strip() or "unknown"
    enabled = subprocess.run(
        ["systemctl", "is-enabled", SERVICE], capture_output=True, text=True, check=False
    ).stdout.strip() or "unknown"
    print(f"service: {active}")
    print(f"boot:    {enabled}")
    print(f"url:     {_service_url()}")
    return 0


def _logs(follow: bool) -> int:
    command = ["journalctl", "-u", SERVICE, "--no-pager"]
    command += ["--follow"] if follow else ["-n", "100"]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        for output, stream in ((result.stdout, sys.stdout), (result.stderr, sys.stderr)):
            if isinstance(output, str) and output:
                print(output, end="", file=stream)
        if result.returncode and isinstance(result.stderr, str) and "not found" in result.stderr:
            print("Studio service is not installed. From your PoLiMa checkout on the DevKit, "
                  "run ./scripts/install_polima_modalix.sh, then polima studio start. "
                  "See docs/install.md for prerequisites. No installation was performed.", file=sys.stderr)
        return result.returncode
    except FileNotFoundError:
        print("polima studio: journalctl is not installed", file=sys.stderr)
        return 2


def run(argv: list[str], parent=None) -> int:
    # Backward compatibility for the former foreground-only command.
    if argv and argv[0].startswith("-") and argv[0] not in {"-h", "--help"}:
        argv = ["serve", *argv]
    if argv[:1] == ["serve"]:  # everything after it is Studio's own options (--port, --host, --root)
        from polima.studio.main import main

        return main(argv[1:])
    args = _parser().parse_args(argv)
    action = args.action or "status"
    dry_run = bool(getattr(parent, "dry_run", False))

    if action == "serve":
        from polima.studio.main import main

        return main(args.args)
    if action == "status":
        return _status()
    if action == "logs":
        return _logs(args.follow)
    if action == "open":
        url = _service_url()
        print(url)
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            webbrowser.open(url)
        return 0
    if action == "disable":
        return _systemctl(["disable", "--now"], mutate=True, dry_run=dry_run)
    result = _systemctl([action], mutate=True, dry_run=dry_run)
    if result or dry_run or action not in {"start", "restart"}:
        return result
    if not _wait_until_ready():
        print(
            "polima studio: service was started but did not become ready within 20 seconds; "
            "run 'polima studio logs' for details",
            file=sys.stderr,
        )
        return 1
    verb = "started" if action == "start" else "restarted"
    print(f"PoLiMa Studio {verb} successfully.")
    print(f"Open: {_service_url()}")
    return 0


def needs_capability(argv: list[str]) -> str | None:
    # Service management does not import Flask/OpenCV. Only foreground serving does.
    if (argv and argv[0] == "serve") or (argv and argv[0].startswith("-")):
        return "studio"
    return None
