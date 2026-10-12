"""Host/board split.

PoLiMa installs as LLiMa does -- separate console scripts per stage
(polima-compile / polima-deploy on the host, polima / polima-run /
polima-robot on the board). These tests pin the two properties that split has to
guarantee:

  1. the shared core imports with stdlib + numpy alone, so the same
     polima.wire / polima.bundle / polima.policies code is the contract between
     both machines; and
  2. a stage invoked on the wrong machine fails with one clear sentence, not an
     ImportError from inside a vendored library.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from polima import role
from polima.cli import main as cli_main

#: Everything the board must be able to import with numpy alone.
CORE_MODULES = [
    "polima.util.jsonio",
    "polima.util.hashing",
    "polima.util.paths",
    "polima.util.table",
    "polima.config.base",
    "polima.config.loader",
    "polima.config.schema",
    "polima.policies.base",
    "polima.policies.registry",
    "polima.policies.act",
    "polima.data.episodes",
    "polima.data.contract",
    "polima.bundle.layout",
    "polima.wire.protocol",
    "polima.role",
]

#: Modules the core must never pull in at import time.
FORBIDDEN = ["torch", "onnx", "onnxsim", "afe", "lerobot", "flask", "cv2", "draccus"]


def test_core_imports_without_heavy_dependencies():
    """The dependency floor, enforced in a clean interpreter.

    Runs as a subprocess so that modules already imported by the test session
    (pytest pulls in a lot) cannot mask a real dependency.
    """
    source_root = str(__import__("pathlib").Path(cli_main.__file__).resolve().parents[2])
    program = (
        f"import sys; sys.path.insert(0, {source_root!r})\n"
        f"import importlib\n"
        f"for m in {CORE_MODULES!r}: importlib.import_module(m)\n"
        f"leaked = [m for m in {FORBIDDEN!r} if m in sys.modules]\n"
        f"print('LEAKED:' + ','.join(leaked))\n"
    )
    # -I isolates from PYTHONPATH and the user site-dir but keeps site-packages,
    # since numpy is a legitimate dependency. The point is a clean sys.modules,
    # not a clean sys.path.
    result = subprocess.run(
        [sys.executable, "-I", "-c", program], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    leaked = result.stdout.strip().removeprefix("LEAKED:").strip()
    assert not leaked, f"core modules imported heavy dependencies: {leaked}"


def test_every_command_is_assigned_to_a_side():
    assigned = set(role.HOST_COMMANDS) | set(role.BOARD_COMMANDS) | set(role.SHARED_COMMANDS)
    assert set(cli_main.COMMANDS) == assigned


def test_host_and_board_commands_do_not_overlap():
    assert not set(role.HOST_COMMANDS) & set(role.BOARD_COMMANDS)


@pytest.mark.parametrize("command", ["compile", "deploy", "run", "robot"])
def test_capabilities_explain_every_stage(command):
    assert role.detect().explain(command)


def test_shared_commands_always_allowed():
    capabilities = role.detect()
    for command in role.SHARED_COMMANDS:
        assert capabilities.allows(command)


BOARD = role.Capabilities(
    role=role.BOARD, machine="aarch64", can_compile=False,
    can_deploy=False, can_run=True, can_robot=True, missing={},
)


def test_wrong_machine_gives_exit_code_3(monkeypatch, capsys):
    """`polima compile` on a board must not raise ImportError."""
    monkeypatch.setattr(role, "detect", lambda: BOARD)
    assert cli_main.main(["compile", "--policy", "act"]) == 3
    message = capsys.readouterr().err
    assert "not available on this machine" in message
    assert "role: board" in message
    assert "model-compiler" in message


def test_import_build_is_exempt_from_the_compiler_gate(monkeypatch, tmp_path, capsys):
    """--import-build only copies built ELFs, so it must run on a DevKit too.

    Gating the whole `compile` command on afe would block it.
    """
    monkeypatch.setattr(role, "detect", lambda: BOARD)
    # Reaches the command (and fails on the empty dir), rather than exiting 3.
    with pytest.raises(FileNotFoundError):
        cli_main.main(["compile", "--import-build", str(tmp_path / "absent")])


def test_needs_capability_refines_by_argv():
    from polima.cli import compile as compile_cli

    assert compile_cli.needs_capability(["--policy", "act"]) == "compile"
    assert compile_cli.needs_capability(["--import-build", "/x"]) is None
    assert compile_cli.needs_capability(["--import-build=/x"]) is None


def test_split_argv_finds_the_command():
    assert cli_main.split_argv(["run", "--fixture"]) == ([], "run", ["--fixture"])
    assert cli_main.split_argv(["--dry-run", "deploy", "--bundle", "x"]) == (
        ["--dry-run"], "deploy", ["--bundle", "x"]
    )
    assert cli_main.split_argv(["--version"]) == (["--version"], None, [])
