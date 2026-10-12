"""`polima workbench`: the PoLiMa page for checkpoints, builds and boards (port 8686).

Runs on any machine -- a build computer, a Modalix board, a laptop -- and is
opened from any browser on the same network, like the board's Studio.
"""

from __future__ import annotations

import argparse
import os
import webbrowser
from pathlib import Path

from polima import role
from polima.util.net import lan_address
from polima.util.paths import outputs_root
from polima.workbench.app import Workbench, folders, serve

BOARD_BUNDLES = Path("/media/nvme/polima/models")


def needs_capability(argv: list[str]) -> None:
    return None  # works everywhere


def run(argv: list[str], parent: argparse.Namespace | None = None) -> int:
    parser = argparse.ArgumentParser(prog="polima workbench", description=__doc__)
    parser.add_argument("--models", action="append", metavar="DIR",
                        help="folder of trained checkpoints (repeatable; POLIMA_MODELS)")
    parser.add_argument("--bundles", action="append", metavar="DIR",
                        help="folder of compiled bundles (repeatable; POLIMA_BUNDLES)")
    parser.add_argument("--board", action="append", metavar="ADDRESS",
                        help="a Modalix board running PoLiMa Studio (repeatable; POLIMA_BOARDS)")
    parser.add_argument("--port", type=int, default=8686)
    parser.add_argument("--host", default="0.0.0.0",
                        help="0.0.0.0 serves every browser on the network (default); "
                             "127.0.0.1 keeps it to this machine")
    parser.add_argument("--open", action="store_true", help="open it in the default browser")
    args = parser.parse_args(argv)

    on_board = role.is_board()
    bundles = [outputs_root() / "bundles"] + ([BOARD_BUNDLES] if BOARD_BUNDLES.is_dir() else [])
    boards = args.board or [b for b in os.environ.get("POLIMA_BOARDS", "").split(",") if b] or (
        ["localhost:8080"] if on_board else [])
    workbench = Workbench(
        models=folders(args.models, "POLIMA_MODELS", [Path.cwd()]),
        bundles=folders(args.bundles, "POLIMA_BUNDLES", bundles),
        boards=boards,
    )
    server = serve(workbench, args.host, args.port)
    url = f"http://{lan_address(args.host)}:{server.server_address[1]}"
    print(f"PoLiMa Workbench: {url}\n  Open it from any browser on this network. Ctrl-C stops it.",
          flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
