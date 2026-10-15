from __future__ import annotations

import pytest

from polima.cli import main as cli_main


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


@pytest.mark.parametrize("custom", [False, True])
def test_calibration_backs_up_and_writes_the_directory_used_by_run(tmp_path, monkeypatch, custom):
    from types import SimpleNamespace

    from polima.cli import robot
    from polima.policies.groot import GROOT_SPEC

    monkeypatch.setenv("POLIMA_ROOT", str(tmp_path / "polima"))
    monkeypatch.setattr(robot.Path, "home", lambda: tmp_path)
    calibration_dir = str(tmp_path / "custom") if custom else None
    calibration = robot._calibration_file(GROOT_SPEC, calibration_dir)
    calibration.parent.mkdir(parents=True)
    calibration.write_text('{"existing": true}')
    binary = tmp_path / "lerobot-calibrate"
    binary.touch()
    monkeypatch.setattr(robot, "robot_python", lambda: str(tmp_path / "python"))
    monkeypatch.setattr(robot, "_discover", lambda *a: ([], [], {}, "/dev/arm", []))
    monkeypatch.setenv("LEROBOT_FORCE_PROVIDED_CALIBRATION", "1")
    calls = []
    def launch(argv, **kwargs):
        import fcntl

        lock = tmp_path / "polima/var/run/control.lock"
        with lock.open("a+") as competitor, pytest.raises(BlockingIOError):
            fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert len(kwargs["pass_fds"]) == 1
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(robot.subprocess, "run", launch)
    args = SimpleNamespace(calibration_dir=calibration_dir, yes=True)
    assert robot._calibrate(GROOT_SPEC, args) == 7
    assert f"--robot.calibration_dir={calibration.parent}" in calls[0][0]
    assert "LEROBOT_FORCE_PROVIDED_CALIBRATION" not in calls[0][1]["env"]
    assert next(calibration.parent.glob("*.bak.*")).read_text() == calibration.read_text()


def test_calibration_backup_failure_does_not_launch_hardware(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    from polima.cli import robot
    from polima.policies.groot import GROOT_SPEC

    monkeypatch.setenv("POLIMA_ROOT", str(tmp_path / "polima"))
    (tmp_path / "so-arm101.json").write_text("{}")
    monkeypatch.setattr(robot, "_discover", lambda *a: ([], [], {}, "/dev/arm", []))
    def denied(*args):
        raise PermissionError("backup directory is read-only")
    monkeypatch.setattr(robot.shutil, "copy2", denied)
    monkeypatch.setattr(robot.subprocess, "run", lambda *a, **k: pytest.fail("hardware launched"))
    assert robot._calibrate(GROOT_SPEC, SimpleNamespace(calibration_dir=str(tmp_path), yes=True)) == 1
    assert "read-only" in capsys.readouterr().err


def test_calibration_respects_active_controller(tmp_path, monkeypatch, capsys):
    import fcntl
    from types import SimpleNamespace

    from polima.cli import robot
    from polima.policies.groot import GROOT_SPEC

    monkeypatch.setenv("POLIMA_ROOT", str(tmp_path))
    lock = tmp_path / "var/run/control.lock"
    lock.parent.mkdir(parents=True)
    monkeypatch.setattr(robot, "_discover", lambda *a: pytest.fail("hardware discovered"))
    with lock.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert robot._calibrate(GROOT_SPEC, SimpleNamespace()) == 1
    assert "another PoLiMa controller is active" in capsys.readouterr().err
