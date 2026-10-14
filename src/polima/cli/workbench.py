"""Manage Workbench for model import, compilation and deployment (port 8686)."""

from __future__ import annotations

import argparse
import fcntl
import http.cookiejar
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

from polima import role
from polima.util.net import lan_address
from polima.util.paths import package_root
from polima.workbench.app import Workbench, folders, serve
from polima.workbench.inputs import workspace

BOARD_BUNDLES = Path("/media/nvme/polima/models")


def state() -> dict:
    """Return a managed instance only when its live identity matches the saved record."""
    try:
        value = json.loads((workspace() / "service.json").read_text())
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(value["url"] + "/api/v1/instance", timeout=1) as response:
            actual = json.load(response)
        return value if value.get("instance") and actual.get("instance") == value["instance"] else {}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}


def manage(action: str, argv: list[str]) -> int:
    workspace().mkdir(parents=True, exist_ok=True)
    with (workspace() / "service.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _manage(action, argv)


def _manage(action: str, argv: list[str]) -> int:
    running = state()
    if action == "status":
        print("Workbench: " + running.get("display_url", running["url"]) if running else "Workbench is stopped.")
        return 0
    if action == "stop":
        if not running:
            print("Workbench is stopped.")
            return 0
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
                                              urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        try:
            with opener.open(running["url"] + "/api/v1/session", timeout=2) as response:
                session = json.load(response)
            request = urllib.request.Request(running["url"] + "/api/v1/shutdown", data=b"{}",
                                             headers={"X-CSRF-Token": session["csrf_token"]})
            opener.open(request, timeout=3).close()
        except urllib.error.HTTPError as error:
            try:
                reason = json.load(error).get("error", str(error))
            except (ValueError, AttributeError):
                reason = str(error)
            print(reason, file=sys.stderr)
            return 1
        except (OSError, ValueError, KeyError, TypeError):
            print("Could not contact Workbench. Run `polima workbench status` and retry.", file=sys.stderr)
            return 1
        for _ in range(30):
            if not state():
                print("Workbench stopped.")
                return 0
            time.sleep(.1)
        print("Workbench has not stopped yet. Check its status before restarting.", file=sys.stderr)
        return 1
    if running:
        print("Workbench: " + running.get("display_url", running["url"]))
        return 0
    instance = secrets.token_hex(16)
    env = {**os.environ, "POLIMA_WORKBENCH_INSTANCE": instance,
           "PYTHONPATH": os.pathsep.join([str(Path(__file__).resolve().parents[2]), os.environ.get("PYTHONPATH", "")])}
    log = workspace() / "workbench.log"
    with log.open("a") as stream:
        process = subprocess.Popen([sys.executable, "-m", "polima", "workbench", "serve", *argv],
                                   env=env, stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(60):
        found = state()
        if found.get("instance") == instance:
            print("Workbench started. Open: " + found.get("display_url", found["url"]))
            return 0
        if process.poll() is not None:
            break
        time.sleep(.1)
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    print(f"Workbench did not become ready. See {log}", file=sys.stderr)
    return 1


def default_models(on_board: bool) -> list[Path]:
    """Discover checkpoints automatically only in a source checkout."""
    root = package_root()
    checkout = (root / "pyproject.toml").is_file() and (root / "src/polima/__init__.py").is_file()
    return [root / "models"] if checkout and not on_board else []


def needs_capability(argv: list[str]) -> None:
    return None


def run(argv: list[str], parent: argparse.Namespace | None = None) -> int:
    parser = argparse.ArgumentParser(prog="polima workbench", description=__doc__)
    parser.add_argument("action", nargs="?", choices=("serve", "start", "stop", "status"), default="serve")
    parser.add_argument("--models", action="append", metavar="DIR",
                        help="folder of trained checkpoints (repeatable; POLIMA_MODELS; default: checkout models/)")
    parser.add_argument("--bundles", action="append", metavar="DIR",
                        help="folder of compiled bundles (repeatable; POLIMA_BUNDLES)")
    parser.add_argument("--devkit", "--board", dest="board", action="append", metavar="ADDRESS",
                        help="a DevKit running PoLiMa Studio (repeatable; POLIMA_DEVKIT_URLS)")
    parser.add_argument("--port", type=int, default=8686)
    parser.add_argument("--host", default="127.0.0.1",
                        help="127.0.0.1 serves this computer (default); "
                             "0.0.0.0 allows network access")
    parser.add_argument("--open", action="store_true", help="open it in the default browser")
    args = parser.parse_args(argv)

    if args.action != "serve":
        options = list(argv)
        options.remove(args.action)
        return manage(args.action, options)

    on_board = role.is_board()
    bundles = [Path(os.environ.get("POLIMA_OUTPUTS", workspace() / "outputs")) / "bundles"] + (
        [BOARD_BUNDLES] if on_board and BOARD_BUNDLES.is_dir() else [])
    boards = args.board or [b for b in (os.environ.get("POLIMA_DEVKIT_URLS") or os.environ.get("POLIMA_BOARDS", "")).split(",") if b] or (
        ["localhost:8080"] if on_board else [])
    workbench = Workbench(
        models=folders(args.models, "POLIMA_MODELS", default_models(on_board)),
        bundles=folders(args.bundles, "POLIMA_BUNDLES", bundles),
        boards=boards,
    )
    server = serve(workbench, args.host, args.port)
    url = f"http://{lan_address(args.host)}:{server.server_address[1]}"
    access = "this computer" if args.host in {"127.0.0.1", "localhost", "::1"} else "this network"
    print(f"PoLiMa Workbench: {url}\n  Open it in a browser on {access}. Ctrl-C stops it.",
          flush=True)
    instance = os.environ.get("POLIMA_WORKBENCH_INSTANCE")
    if instance:
        workspace().mkdir(parents=True, exist_ok=True)
        control_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
        (workspace() / "service.json").write_text(json.dumps({"instance": instance, "display_url": url,
            "url": f"http://{control_host}:{server.server_address[1]}"}))
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
