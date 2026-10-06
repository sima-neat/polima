from __future__ import annotations

from polima.cli import main as cli_main


def test_help_command_prints_top_level_help(capsys) -> None:
    assert cli_main.main(["help"]) == 0
    output = capsys.readouterr().out
    assert "usage: polima" in output
    assert "COMMANDS:" in output
    assert "help      show help for PoLiMa or a specific command" in output


def test_help_command_prints_subcommand_help(capsys) -> None:
    assert cli_main.main(["help", "studio"]) == 0
    output = capsys.readouterr().out
    assert "usage: polima studio" in output
    assert "{status,start,stop,restart,enable,disable,open,logs,serve}" in output


def test_missing_dependency_shows_the_real_import_error(capsys, monkeypatch) -> None:
    monkeypatch.setitem(cli_main.COMMANDS, "deploy", ("polima_no_such_module", "x"))
    assert cli_main.main(["deploy", "--help"]) == 2
    error = capsys.readouterr().err
    assert "No module named 'polima_no_such_module'" in error
    assert "not implemented" not in error


def test_python_dash_m_polima_runs_the_cli() -> None:
    import os
    import subprocess
    import sys
    from pathlib import Path

    src = Path(cli_main.__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(src)}
    result = subprocess.run([sys.executable, "-m", "polima", "--version"],
                            capture_output=True, text=True, env=env, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("polima ")
