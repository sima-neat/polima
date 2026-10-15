"""`polima` -- one CLI for compiling, deploying and running robot policies on
SiMa.ai Modalix.

    polima compile    --policy act --checkpoint <path> --build-dir <dir>
    polima deploy     --bundle <id> --devkit <address>
    polima run        --bundle <id> --fixture
    polima robot      install | ports | doctor | preview | calibrate | run | export-improvement-manifest
    polima studio     start | stop | status
    polima workbench  start | stop | status
    polima doctor

Run `polima <command> --help` for a command's options.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys

from polima import role
from polima.util.logging import setup

#: subcommand -> (module, help). Modules are imported only when invoked, so
#: `polima doctor` still works in an environment missing torch or afe -- which
#: is exactly the situation doctor exists to diagnose.
#: Where each stage runs: see polima.role.HOST_COMMANDS / BOARD_COMMANDS.
COMMANDS = {
    "compile": ("polima.cli.compile", "compile a checkpoint into a bundle for Modalix"),
    "deploy": ("polima.cli.deploy", "deploy a bundle to a Modalix DevKit"),
    "run": ("polima.cli.run", "measure numerical fidelity and inference latency"),
    "robot": ("polima.cli.robot", "check, calibrate and drive the arm"),
    "studio": ("polima.cli.studio", "start or stop PoLiMa Studio on the DevKit"),
    "workbench": ("polima.cli.workbench", "manage PoLiMa Workbench on this computer"),
    "doctor": ("polima.cli.doctor", "check this computer, the compiler and the DevKit"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polima",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="COMMANDS:\n"
        + "\n".join(f"  {name:9s} {help_text}" for name, (_, help_text) in COMMANDS.items()),
    )
    parser.add_argument("--config", metavar="FILE", help="extra config file, highest file layer")
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ERROR")
    parser.add_argument(
        "--dry-run", action="store_true", help="print what would run without doing it"
    )
    parser.add_argument("--version", action="store_true", help="print the PoLiMa version and exit")
    parser.add_argument("command", nargs="?", choices=list(COMMANDS), help=argparse.SUPPRESS)
    return parser


def split_argv(argv: list[str]) -> tuple[list[str], str | None, list[str]]:
    """Split at the first recognised command name.

    argparse's REMAINDER cannot do this: it stops capturing at a leading
    `--flag`, so `polima list --graphs` would try to parse --graphs globally.
    Splitting by hand lets every subcommand own its own flag namespace, which is
    what makes `polima data validate --json` and `polima doctor --json` independent.
    """
    for index, token in enumerate(argv):
        if token in COMMANDS:
            return argv[:index], token, argv[index + 1:]
    return argv, None, []


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    globals_argv, command, rest = split_argv(argv)

    parser = build_parser()
    args = parser.parse_args(globals_argv)
    args.command = command

    if args.version:
        print(f"polima {_version()}")
        return 0
    if not command:
        parser.print_help()
        return 1

    setup(args.log_level)

    module_name, _ = COMMANDS[command]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        # Show the real missing dependency rather than a generic message.
        print(f"polima {command}: cannot load {module_name}: {exc}\n"
              f"Install the missing package (see Install in README.md).", file=sys.stderr)
        return 2

    denial = role.check(command, rest, module)
    if denial:
        print(f"polima {command}: {denial}", file=sys.stderr)
        return 3

    try:
        return module.run(rest, parent=args)
    except KeyboardInterrupt:
        print(f"\npolima {command}: interrupted", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            raise
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        # Studio reports the last output line containing words like "error": this one, not the hint.
        print(f"polima {command}: error: {message}", file=sys.stderr)
        print("  (for more detail, set POLIMA_LOG_LEVEL=DEBUG and run the command again)",
              file=sys.stderr)
        return 1


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("polima")
    except Exception:  # noqa: BLE001
        return "0.3.0+src"


if __name__ == "__main__":
    raise SystemExit(main())
