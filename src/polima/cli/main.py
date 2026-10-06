"""`polima` -- one CLI for compiling, deploying and running robot policies on
SiMa.ai Modalix.

Mirrors LLiMa's shape (llima-compile / llima-deploy / llima-run) as subcommands:

    polima compile  --policy act --checkpoint <path>
    polima deploy   --bundle <id> --board sima@<board-address>
    polima run      --bundle <id> --fixture
    polima robot    install | ports | doctor | preview | calibrate | run | export-improvement-manifest
    polima doctor
    polima data     validate <roots...>
    polima list
    polima clean    <build-dir> [--level scratch|inputs|all] [--yes]

Subcommand modules are imported lazily so `polima doctor` still works in an
environment where, say, torch or afe is missing -- which is exactly the
situation doctor exists to diagnose.
"""

from __future__ import annotations

import argparse
import importlib
import sys

from polima import role
from polima.util.logging import setup

#: subcommand -> (module, help). Modules are imported only when invoked, so
#: `polima doctor` still works in an environment missing torch or afe -- which
#: is exactly the situation doctor exists to diagnose.
#: Where each stage runs: see polima.role.HOST_COMMANDS / BOARD_COMMANDS.
COMMANDS = {
    "help": ("polima.cli.help_cmd", "show help for PoLiMa or a specific command"),
    "doctor": ("polima.cli.doctor", "diagnose the environment (host, compiler venv, board)"),
    "data": ("polima.cli.data", "inspect, validate and combine datasets"),
    "list": ("polima.cli.list_cmd", "list registered policies"),
    "clean": ("polima.cli.clean", "reclaim space from build trees"),
    "compile": ("polima.cli.compile", "export, quantize and compile a policy to MLA ELFs"),
    "deploy": ("polima.cli.deploy", "push a bundle to a Modalix board and start it"),
    "run": ("polima.cli.run", "run inference against a deployed bundle"),
    "robot": ("polima.cli.robot", "robot control: teleop, diagnostics, live view"),
    "studio": ("polima.cli.studio", "manage the SOM web control cockpit"),
    "workbench": ("polima.cli.workbench", "open the build computer's page (localhost:8686)"),
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

    return module.run(rest, parent=args)


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("polima")
    except Exception:  # noqa: BLE001
        return "0.2.0+src"


if __name__ == "__main__":
    raise SystemExit(main())
