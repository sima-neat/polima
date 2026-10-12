"""`polima doctor` -- check this computer, its Python environments and the DevKit.

Run it after any install. On the build computer it checks the Python
environment, the SiMa model compiler, the policies, your datasets and the
DevKit (over ssh). On a DevKit it checks the DevKit itself. `--all` runs every
check on either machine.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

from polima.compile import toolchain
from polima.config.base import NO_BOARD_MESSAGE
from polima.config.loader import load
from polima.util import table

#: Modules that must import with stdlib + numpy only, in every interpreter.
CORE_MODULES = (
    "polima.util.jsonio",
    "polima.util.hashing",
    "polima.util.paths",
    "polima.config.base",
    "polima.config.loader",
    "polima.policies.base",
    "polima.policies.registry",
    "polima.data.episodes",
    "polima.data.contract",
    # The compile modules belong here too: `polima.compile.tensor` runs *inside*
    # the compiler venv, so it must import with numpy alone and reach afe only
    # from within the functions that use it.
    "polima.compile.calibration",
    "polima.compile.mpk",
    "polima.compile.toolchain",
    "polima.compile.driver",
    "polima.compile.tensor",
    # Likewise the export driver: it resolves torch-dependent code by dotted
    # string, so the driver itself must import without torch. Only
    # `polima.policies.<name>.graphs` may depend on it, and that is never listed
    # here.
    "polima.export.driver",
    "polima.export.samples",
    "polima.export.normalization",
)

COMPILER_BIN_CANDIDATES = toolchain.SEARCH_PATH


#: Sections about the build computer; a DevKit has no compiler or datasets.
HOST_ONLY_SECTIONS = ("model compiler", "datasets")


class Doctor:
    def __init__(self, *, json_output: bool = False) -> None:
        self.json_output = json_output
        self.results: list[dict] = []
        self.worst = table.OK

    def record(self, kind: str, label: str, detail: str = "", **extra) -> None:
        self.results.append({"status": kind, "check": label, "detail": detail, **extra})
        order = {table.OK: 0, table.SKIP: 0, table.WARN: 1, table.FAIL: 2}
        if order.get(kind, 0) > order.get(self.worst, 0):
            self.worst = kind
        if not self.json_output:
            print(table.status(kind, label, detail))

    def heading(self, title: str) -> None:
        if not self.json_output:
            print(table.section(title))


def run(argv: list[str], parent: argparse.Namespace | None = None) -> int:
    parser = argparse.ArgumentParser(prog="polima doctor", description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--board", action="store_true", help="only the board checks")
    parser.add_argument("--imports", action="store_true", help="only the import matrix")
    parser.add_argument("--no-board", action="store_true", help="skip board reachability")
    parser.add_argument("--all", action="store_true",
                        help="run every section, including ones this machine's role skips")
    args = parser.parse_args(argv)

    config = load(config_file=getattr(parent, "config", None))
    doc = Doctor(json_output=args.json)
    only_board, only_imports = args.board, args.imports
    everything = not (only_board or only_imports)

    from polima import role

    capabilities = role.detect()
    # A board runs the host-only sections only when explicitly asked. The role,
    # not the flag, is the default: `polima-doctor` should be clean on a healthy
    # SoM without anyone having to know which flags to pass.
    on_board = capabilities.role == role.BOARD
    host_side = args.all or not on_board

    if everything:
        check_role(doc)
        check_host(doc, host_side)
    if everything or only_imports:
        check_import_matrix(doc, config)
    if everything:
        if host_side:
            check_compiler(doc, config)
        check_policies(doc)
        if host_side:
            check_datasets(doc, config)
        else:
            skip_host_sections(doc, capabilities)
    # On the board, probe this machine directly; ssh'ing to the configured board
    # from the board itself would either loop back or fail on a key it need not
    # have. An explicit `--board` still means "do the remote probe".
    if only_board:
        check_board(doc, config)
    elif everything and not args.no_board:
        check_board(doc, config, local=on_board)

    if args.json:
        print(json.dumps({"status": doc.worst, "checks": doc.results}, indent=2))
    else:
        print()
        counts: dict[str, int] = {}
        for item in doc.results:
            counts[item["status"]] = counts.get(item["status"], 0) + 1
        print("  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 1 if doc.worst == table.FAIL else 0


# ----------------------------------------------------------------- host / repo


def check_role(doc: Doctor) -> None:
    from polima import role

    capabilities = role.detect()
    doc.heading(f"install role: {capabilities.role} ({capabilities.machine})")

    stages = (
        ("compile", capabilities.can_compile, role.HOST),
        ("deploy", capabilities.can_deploy, role.HOST),
        ("run", capabilities.can_run, role.BOARD),
        ("robot", capabilities.can_robot, role.BOARD),
    )
    for command, enabled, belongs in stages:
        # A host missing board-only extras is normal, and vice versa -- only
        # report a gap when the stage belongs on THIS machine.
        expected_here = (
            belongs == role.HOST and capabilities.role in (role.HOST, role.BOTH)
        ) or (belongs == role.BOARD and capabilities.role in (role.BOARD, role.BOTH))
        if enabled:
            doc.record(table.OK, f"polima-{command}", "available")
        elif expected_here:
            doc.record(table.WARN, f"polima-{command}", capabilities.explain(command))
        else:
            doc.record(table.SKIP, f"polima-{command}", f"{belongs}-side stage, not needed here")

    if capabilities.missing:
        doc.record(
            table.SKIP, "optional modules",
            ", ".join(f"{m} ({p})" for m, p in sorted(capabilities.missing.items())),
        )


def skip_host_sections(doc: Doctor, capabilities) -> None:
    """Say what was skipped and why, rather than silently shrinking the report."""
    doc.heading("host-only checks")
    doc.record(
        table.SKIP, ", ".join(HOST_ONLY_SECTIONS),
        f"not applicable on this {capabilities.machine} board "
        f"-- pass --all to run them anyway",
    )


def check_host(doc: Doctor, host_side: bool = True) -> None:
    """The running interpreter, on either platform.

    Python and numpy are the dependency floor everywhere. A host venv also owns
    the full export stack; only AFE is supplied by an external extension.
    """
    doc.heading("interpreter")
    doc.record(table.OK, "python", f"{sys.version.split()[0]} at {sys.executable}")
    try:
        import numpy

        doc.record(table.OK, "numpy", numpy.__version__)
    except ImportError:
        doc.record(table.FAIL, "numpy", "missing -- polima's only hard dependency")

    if host_side:
        for module in ("torch", "lerobot", "onnx", "onnxruntime", "pyarrow"):
            present = importlib.util.find_spec(module) is not None
            doc.record(
                table.OK if present else table.WARN,
                f"host dependency {module}",
                "installed in current environment" if present else "missing; run `make venv`",
            )


# -------------------------------------------------------------- import matrix


def check_import_matrix(doc: Doctor, config) -> None:
    doc.heading("import matrix (core modules must need only stdlib + numpy)")
    interpreters = {"current": sys.executable}

    compiler_bin = _find_compiler_bin(config)
    if compiler_bin:
        interpreters["model-compiler"] = str(Path(compiler_bin) / "python")

    source_root = str(Path(__file__).resolve().parents[2])
    snippet = (
        "import sys; sys.path.insert(0, %r);\n" % source_root
        + "import importlib\n"
        + f"mods = {list(CORE_MODULES)!r}\n"
        + "bad = []\n"
        + "for m in mods:\n"
        + "    try: importlib.import_module(m)\n"
        + "    except Exception as e: bad.append(f'{m}: {type(e).__name__}: {e}')\n"
        + "print('|'.join(bad))\n"
    )

    for label, python in interpreters.items():
        if not python or not Path(python).exists():
            doc.record(table.SKIP, f"imports [{label}]", "interpreter not found")
            continue
        result = subprocess.run(
            [python, "-c", snippet], capture_output=True, text=True, timeout=120
        )
        failures = result.stdout.strip()
        version = _capture([python, "-c", "import sys;print(sys.version.split()[0])"]).strip()
        if result.returncode != 0:
            doc.record(table.FAIL, f"imports [{label}]", result.stderr.strip()[:160])
        elif failures:
            doc.record(table.FAIL, f"imports [{label}]", failures[:200])
        else:
            doc.record(
                table.OK, f"imports [{label}]", f"py{version}: all {len(CORE_MODULES)} core modules"
            )


# -------------------------------------------------------------- compiler venv


def check_compiler(doc: Doctor, config) -> None:
    doc.heading("model compiler (SiMa ModelSDK)")
    compiler_bin = _find_compiler_bin(config)
    if not compiler_bin:
        doc.record(
            table.FAIL, "compiler bin",
            "not found; tried " + ", ".join(COMPILER_BIN_CANDIDATES),
        )
        return
    doc.record(table.OK, "compiler bin", compiler_bin)

    # Palette's activation is what puts the compiler's shared libraries on the
    # loader path. Absent is normal for a host install and fatal in a container.
    activation = toolchain.find_activation(compiler_bin)
    doc.record(
        table.OK if activation else table.SKIP,
        toolchain.ACTIVATION_SCRIPT,
        f"sourced from {activation}" if activation
        else "not present -- fine unless afe fails on a missing library",
    )

    python = str(Path(compiler_bin) / "python")
    version = _capture([python, "-c", "import sys;print(sys.version.split()[0])"]).strip()
    doc.record(table.OK if version else table.FAIL, "compiler python", version or "unusable")

    afe = _capture(
        [python, "-c",
         "from afe.apis.release_v1 import get_model_sdk_version as v;print(v())"]
    ).strip()
    doc.record(table.OK if afe else table.FAIL, "afe / ModelSDK", afe or "import failed")

    # By IMPORT, never by exec: the llima-compile console script's shebang points
    # at /sdk-extensions/..., which exists only inside the SDK container.
    lmm = _capture(
        [python, "-c",
         "import sima_lmm.host.compile_lmm as m;"
         "print(','.join(n for n in ('main','gen_files') if hasattr(m,n)))"]
    ).strip()
    doc.record(
        table.OK if "gen_files" in lmm else table.FAIL,
        "sima_lmm.host.compile_lmm",
        f"importable ({lmm})" if lmm else "import failed",
    )

    script = Path(compiler_bin) / "llima-compile"
    if script.is_file():
        shebang = script.read_text(errors="replace").splitlines()[0]
        interpreter = shebang.removeprefix("#!").strip().split()[0] if shebang.startswith("#!") else ""
        usable = bool(interpreter) and Path(interpreter).exists()
        doc.record(
            table.OK if usable else table.WARN,
            "llima-compile script",
            "executable" if usable
            else f"shebang {shebang!r} is unresolvable -- invoke "
                 "`$COMPILER_BIN/python -m sima_lmm.host.compile_lmm` instead",
        )


# ------------------------------------------------------------ policies / data


def check_policies(doc: Doctor) -> None:
    doc.heading("policies")
    from polima.policies.registry import BUILTIN, load_all

    specs = load_all(strict=False)
    for name in BUILTIN:
        spec = specs.get(name)
        if spec is None:
            doc.record(table.SKIP, f"policy {name}", "not ported yet")
            continue
        try:
            spec.validate()
            doc.record(
                table.OK, f"policy {name}",
                f"{len(spec.compile.graphs)} graphs, port {spec.wire.default_port}, "
                f"magic {spec.wire.magic_ascii}",
            )
        except Exception as exc:  # noqa: BLE001
            doc.record(table.FAIL, f"policy {name}", str(exc)[:160])


def check_datasets(doc: Doctor, config) -> None:
    doc.heading(f"datasets under {config.paths.dataset_parent}")
    from polima.data.contract import validate
    from polima.data.discover import discover
    from polima.policies.registry import iter_specs

    if not Path(config.paths.dataset_parent).is_dir():
        doc.record(table.SKIP, "datasets", "set POLIMA_DATASET_PARENT to the folder holding your datasets")
        return
    entries = discover(config.paths.dataset_parent)
    if not entries:
        doc.record(table.WARN, "datasets", f"none found under {config.paths.dataset_parent}")
        return

    specs = list(iter_specs())
    for entry in entries:
        verdicts = []
        for spec in specs:
            report = validate(entry.root, spec.dataset)
            verdicts.append(f"{spec.name}={'ok' if report.ok else 'no'}")
        doc.record(
            table.OK, entry.name,
            f"{entry.episodes} eps, {entry.frames} frames, {entry.fps}fps  "
            + " ".join(verdicts),
        )


# ------------------------------------------------------------------- board


def check_board(doc: Doctor, config, *, local: bool = False) -> None:
    """Probe the board -- over ssh from a host, directly when this IS the board."""
    board = config.board
    if not local and not board.host:
        doc.heading("board")
        doc.record(table.WARN, "board address", NO_BOARD_MESSAGE)
        return
    doc.heading("board (this machine)" if local else f"board {board.host}")
    probe = (
        "echo VER=$(uname -m); "
        "echo CORES=$(nproc); "
        f"echo FREE=$(df -BG --output=avail {board.root.split('/')[1] and '/media/nvme'} "
        "2>/dev/null | tail -1 | tr -d ' G'); "
        "echo CMAKE=$(cmake --version 2>/dev/null | head -1 | awk '{print $3}'); "
        "echo SIMALMM=$(ls -d /usr/lib/*/cmake/SimaLMM 2>/dev/null | head -1); "
        "echo JSON=$(ls /usr/include/nlohmann/json.hpp 2>/dev/null); "
        f"echo POLIMA=$(ls -d {board.root} 2>/dev/null)"
    )
    argv = (
        ["sh", "-c", probe] if local
        else ["ssh", *board.ssh_options, "-o", f"ConnectTimeout={int(board.connect_timeout_s)}",
              board.host, probe]
    )
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode != 0:
        doc.record(
            table.WARN, "reachable",
            f"{'probe' if local else 'ssh'} failed: "
            f"{result.stderr.strip().splitlines()[-1] if result.stderr else 'no output'}",
        )
        return

    facts = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    arch = facts.get("VER", "")
    doc.record(
        table.OK if arch == "aarch64" else table.FAIL,
        "this machine" if local else "reachable",
        f"{arch}, {facts.get('CORES', '?')} cores",
    )
    free = facts.get("FREE", "")
    doc.record(table.OK if free else table.WARN, "free space", f"{free}G on /media/nvme")
    doc.record(
        table.OK if facts.get("CMAKE") else table.FAIL, "cmake", facts.get("CMAKE", "absent")
    )
    doc.record(
        table.OK if facts.get("SIMALMM") else table.FAIL,
        "SimaLMM cmake package", facts.get("SIMALMM") or "absent -- find_package will fail",
    )
    doc.record(
        table.OK if facts.get("JSON") else table.WARN,
        "nlohmann/json", facts.get("JSON") or "absent -- vendor it into native/third_party",
    )
    doc.record(
        table.OK if facts.get("POLIMA") else table.SKIP,
        "polima root", facts.get("POLIMA") or f"{board.root} not created yet (first deploy will)",
    )


# ---------------------------------------------------------------- internals


def _capture(argv: list[str], timeout: float = 60) -> str:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return result.stdout if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _find_compiler_bin(config) -> str | None:
    found = toolchain.find_compiler_bin(config)
    return str(found) if found else None


def _conda_python(env: str) -> str | None:
    base = _capture(["conda", "info", "--base"]).strip()
    if not base:
        return None
    candidate = Path(base) / "envs" / env / "bin" / "python"
    return str(candidate) if candidate.exists() else None
