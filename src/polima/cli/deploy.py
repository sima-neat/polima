"""`polima deploy` -- push a bundle to a Modalix DevKit, serving only on request."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from polima.bundle.contract import spec_from_bundle
from polima.bundle.layout import Bundle
from polima.config.base import NO_BOARD_MESSAGE
from polima.config.loader import load
from polima.deploy import board as board_ops
from polima.deploy import deploy as run_deploy
from polima.deploy import mla
from polima.deploy.service import logs as server_logs
from polima.deploy.service import status as service_status
from polima.deploy.service import stop as stop_server
from polima.deploy.ssh import BoardSession
from polima.policies.registry import get_policy
from polima.util import table
from polima.util.jsonio import dumps, read_json
from polima.util.paths import outputs_root


def run(argv: list[str], parent: argparse.Namespace | None = None) -> int:
    parser = argparse.ArgumentParser(prog="polima deploy")
    parser.add_argument("--bundle", help="bundle id or path")
    parser.add_argument("--devkit", "--board", dest="board", default=None, metavar="ADDRESS",
                        help="the DevKit: user@host or a name saved in the Workbench")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--bundles-root", default=None)
    parser.add_argument("--no-build", action="store_true", help="do not build the runtime on the DevKit")
    parser.add_argument("--start", action="store_true", help="start serving after deployment")
    parser.add_argument("--no-activate", action="store_true", help="do not move `current`")
    parser.add_argument("--force", action="store_true", help="re-transfer and rebuild")
    parser.add_argument("--verbose-server", action="store_true", help="per-stage server timings")
    parser.add_argument("--json", action="store_true")

    parser.add_argument("--list", action="store_true", help="list bundles on the DevKit")
    parser.add_argument("--status", action="store_true", help="show server status")
    parser.add_argument("--stop", action="store_true", help="stop the server")
    parser.add_argument("--logs", type=int, nargs="?", const=40, help="tail the server log")
    parser.add_argument("--activate", metavar="BUNDLE_ID", help="switch `current` and restart")
    parser.add_argument(
        "--reset-mla",
        action="store_true",
        help="reset the accelerator (SDK recovery script) and exit",
    )

    args = parser.parse_args(argv)
    config = load(
        config_file=getattr(parent, "config", None),
        overrides={"board.host": args.board, "board.port": args.port},
    )
    board = config.board
    dry_run = bool(getattr(parent, "dry_run", False))

    if (
        args.list
        or args.status
        or args.stop
        or args.logs is not None
        or args.activate
        or args.reset_mla
    ):
        if not board.host:
            print(f"polima deploy: {NO_BOARD_MESSAGE}", file=sys.stderr)
            return 2
        return _manage(args, board, dry_run)

    if not args.bundle:
        parser.error("--bundle is required (or use --list/--status/--stop/--logs)")

    bundle = _resolve_bundle(args.bundle, args.bundles_root)
    if not board.host:
        print(f"polima deploy: {NO_BOARD_MESSAGE}", file=sys.stderr)
        return 2
    spec = spec_from_bundle(get_policy(bundle.policy), bundle)
    # A bundle with an out-of-date robot client is upgraded in place: the id covers model arithmetic, while bundle.json's
    # sidecars make the deploy notice and transfer the added client files.
    from polima.bundle.importer import hydrate_robot_client, hydrate_runtime_metadata

    hydrate_runtime_metadata(bundle, spec)
    hydrate_robot_client(bundle, spec)
    port = args.port or board.port or spec.wire.default_port

    report = run_deploy(
        bundle,
        board,
        port=port,
        build=not args.no_build,
        start_service=args.start,
        activate=not args.no_activate,
        force=args.force,
        verbose_server=args.verbose_server,
        dry_run=dry_run,
    )

    if args.json:
        print(dumps(report.to_dict()), end="")
        return 0 if report.ok else 1

    print(table.section(f"deploy {report.bundle_id} -> {report.host}"))
    print(
        table.render(
            [[name, info["status"], _detail(name, info)] for name, info in report.stages.items()],
            headers=["stage", "status", "detail"],
        )
    )
    print(f"\n  {report.duration_s:.1f}s total")
    if report.service and report.service.running:
        print(f"  serving on {board.address}:{report.port} (pid {report.service.pid})")
        print(f"\nnext:  polima run --bundle {report.bundle_id} --fixture")
    return 0 if report.ok else 1


def _detail(stage: str, info: dict) -> str:
    if stage == "preflight":
        return (
            f"{info.get('machine', '?')}, {info.get('cores', '?')} cores, "
            f"{info.get('free_bytes', 0) / 1e9:.0f} GB free"
        )
    if stage == "build":
        return info.get("reason") or f"hash {info.get('source_hash', '?')}"
    if stage == "sync-bundle":
        size = info.get("bytes")
        return f"{size / 1048576:.1f} MiB" if size else info.get("reason", "")
    if stage == "service":
        return f"pid {info.get('pid')} port {info.get('port')}"
    if stage == "link-bins":
        aliases = info.get("aliases") or []
        return ", ".join(aliases) if aliases else info.get("reason", "")
    if info.get("problems"):
        return "; ".join(info["problems"])[:80]
    return info.get("reason", "")


def _manage(args, board, dry_run: bool) -> int:
    with BoardSession(board, dry_run=dry_run) as session:
        if args.list:
            current = board_ops.current_bundle(session, board)
            managed = board_ops.managed_bundles(session, board)
            rows = []
            for name in board_ops.list_bundles(session, board):
                is_managed = name in managed
                if is_managed:
                    policy, robot = board_ops.bundle_identity(session, board, name)
                else:
                    lowered = name.lower()
                    policy = next(
                        (
                            candidate
                            for candidate in ("act", "groot")
                            if lowered.startswith(candidate)
                        ),
                        "-",
                    )
                    robot = "-"
                rows.append(
                    [
                        name,
                        policy,
                        robot,
                        "bundle" if is_managed else "other",
                        "current" if name == current else "",
                    ]
                )
            print(
                table.render(
                    rows,
                    headers=["name", "policy", "robot", "format", "status"],
                )
                or "no bundles on the DevKit"
            )
            return 0

        if args.stop:
            stopped = stop_server(session, board)
            print("stopped" if stopped else "no server was running")
            return 0

        if args.reset_mla:
            # Stop first: the reset restarts the dispatcher underneath any
            # process still holding models, and a server that survives it is
            # left talking to a runtime that no longer knows about its ELFs.
            stop_server(session, board)
            report = mla.reset(session, board)
            if args.json:
                print(dumps(report.to_dict()), end="")
                return 0 if report.ok else 1
            print(f"MLA reset via {report.method}: {'ok' if report.ok else 'FAILED'}")
            for step in report.steps:
                print(f"  {step}")
            if report.reclaimed_kb is not None:
                print(
                    f"  CmaFree {report.cma_free_kb_before} -> "
                    f"{report.cma_free_kb_after} kB ({report.reclaimed_kb:+d})"
                )
            return 0 if report.ok else 1

        if args.logs is not None:
            print(server_logs(session, board, args.logs))
            return 0

        if args.activate:
            board_ops.activate(session, board, args.activate)
            print(f"current -> {args.activate}")
            return 0

        current = board_ops.current_bundle(session, board)
        # Without a resolved port there is nothing to probe, and `--status`
        # would report `listening False` for a server that is plainly serving.
        # The port comes from the active bundle's own policy.
        port = args.port or board.port or _default_port(current)
        status = service_status(session, board, port or None)
        rows = [
            ["current", current or "-"],
            ["running", str(status.running)],
            ["pid", status.pid or "-"],
            ["port", port or "-"],
            ["listening", str(status.listening)],
        ]
        if status.bound and status.pid not in status.bound:
            # A stale server answering while the recorded process is dead
            # makes redeploys appear to work while changing nothing.
            rows.append(["WARNING", f"port held by pid(s) {status.bound}, not {status.pid}"])
        print(table.render(rows))
        return 0


def _resolve_bundle(reference: str, bundles_root: str | None) -> Bundle:
    candidate = Path(reference)
    if not candidate.is_dir():
        root = Path(bundles_root or (outputs_root() / "bundles"))
        candidate = root / reference
    manifest = candidate / "bundle.json"
    if not manifest.is_file():
        print(
            f"polima deploy: no bundle at {candidate}\n"
            f"  build one with: polima compile --checkpoint <path> --build-dir <dir>",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return Bundle.from_dict(read_json(manifest), candidate)


def _default_port(bundle_id: str | None) -> int:
    """The wire port of the policy that owns `bundle_id`.

    Bundle ids lead with the policy name (`act-<dataset>-<steps>-<sha>`), which
    is enough to find the spec without fetching bundle.json off the DevKit.
    """
    from polima.policies.registry import available

    name = (bundle_id or "").split("-")[0]
    if name not in available():
        name = "act"
    try:
        return get_policy(name).wire.default_port
    except (KeyError, AttributeError):
        return 0
