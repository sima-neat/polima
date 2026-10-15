"""`polima doctor` checks the platform it is on.

Most checks are host checks -- the workspace checkout, the lerobot clones, the
SiMa compiler environment and training datasets. A DevKit has none of them, so those sections
are skipped there and a healthy DevKit install exits 0.
"""

from __future__ import annotations

import pytest

from polima import role
from polima.cli import doctor
from polima.util import table

#: The sections that only make sense where the build happens.
HOST_SECTIONS = ("check_compiler", "check_datasets")
#: Sections that run everywhere, so the board still gets a useful report.
SHARED_SECTIONS = ("check_role", "check_host", "check_import_matrix", "check_policies")

BOARD = role.Capabilities(
    role=role.BOARD, machine="aarch64", can_compile=False,
    can_deploy=False, can_run=True, can_robot=True, missing={},
)
HOST = role.Capabilities(
    role=role.HOST, machine="x86_64", can_compile=True,
    can_deploy=True, can_run=True, can_robot=False, missing={},
)


@pytest.fixture
def ran(monkeypatch):
    """Replace every section with a recorder, so no test shells out."""
    called: list[str] = []

    def recorder(name):
        def section(*args, **kwargs):
            called.append(name)
        return section

    for name in HOST_SECTIONS + SHARED_SECTIONS + ("check_board",):
        monkeypatch.setattr(doctor, name, recorder(name))
    return called


def _as(monkeypatch, capabilities):
    monkeypatch.setattr(role, "detect", lambda: capabilities)


def test_board_skips_the_host_only_sections(ran, monkeypatch, capsys):
    _as(monkeypatch, BOARD)
    assert doctor.run([]) == 0
    assert not [name for name in ran if name in HOST_SECTIONS]
    for name in SHARED_SECTIONS:
        assert name in ran, f"{name} must run on every platform"


def test_board_says_what_it_skipped(ran, monkeypatch, capsys):
    """A shorter report has to explain itself, or it reads as checks silently lost."""
    _as(monkeypatch, BOARD)
    doctor.run([])
    output = capsys.readouterr().out
    assert "host-only checks" in output
    for section in doctor.HOST_ONLY_SECTIONS:
        assert section in output
    assert "--all" in output


def test_all_overrides_the_platform_gate(ran, monkeypatch):
    _as(monkeypatch, BOARD)
    doctor.run(["--all"])
    for name in HOST_SECTIONS:
        assert name in ran


def test_host_runs_everything(ran, monkeypatch):
    _as(monkeypatch, HOST)
    doctor.run([])
    for name in HOST_SECTIONS + SHARED_SECTIONS:
        assert name in ran


def test_board_probes_itself_rather_than_ssh(monkeypatch):
    """On the board, `board.host` is this machine; ssh'ing to it is a loopback
    that needs a key the board has no reason to hold."""
    argv_seen: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = "VER=aarch64\nCORES=16\nFREE=430\nCMAKE=3.25.1\n"
        stderr = ""

    def fake_run(argv, **kwargs):
        argv_seen.append(argv)
        return Result()

    monkeypatch.setattr(doctor.subprocess, "run", fake_run)
    from polima.config.loader import load

    doc = doctor.Doctor(json_output=True)
    doctor.check_board(doc, load(), local=True)
    assert argv_seen[0][0] == "sh"

    doctor.check_board(doc, load(overrides={"board.host": "sima@192.0.2.10"}), local=False)
    assert argv_seen[1][0] == "ssh"


def test_an_unconfigured_board_is_a_warning_not_an_ssh(monkeypatch):
    monkeypatch.setattr(doctor.subprocess, "run",
                        lambda *a, **k: pytest.fail("no board address: nothing to probe"))
    from polima.config.loader import load

    doc = doctor.Doctor(json_output=True)
    doctor.check_board(doc, load(overrides={"board.host": ""}, use_env=False), local=False)
    assert doc.results[-1]["status"] == table.WARN


def test_a_healthy_board_exits_zero(ran, monkeypatch):
    """`polima-doctor` on a working DevKit exits 0."""
    _as(monkeypatch, BOARD)
    assert doctor.run([]) == 0


def test_explicit_board_flag_still_does_the_remote_probe(monkeypatch):
    """`--board` is an explicit request, so the role must not override it."""
    seen: list[bool] = []
    monkeypatch.setattr(
        doctor, "check_board", lambda doc, config, local=False: seen.append(local)
    )
    _as(monkeypatch, BOARD)
    doctor.run(["--board"])
    assert seen == [False]


def test_skip_notice_is_not_a_failure(monkeypatch):
    doc = doctor.Doctor(json_output=True)
    doctor.skip_host_sections(doc, BOARD)
    assert doc.worst == table.OK


def test_a_host_dependency_that_cannot_be_imported_is_a_failure(monkeypatch):
    """pyarrow built for another numpy is installed, found, and unusable."""
    def broken(name):
        if name == "pyarrow":
            raise ImportError("pyarrow requires NumPy 2.0 or newer")
        return object()

    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(doctor.importlib, "import_module", broken)
    doc = doctor.Doctor(json_output=True)
    doctor.check_host(doc)
    by_check = {result["check"]: result for result in doc.results}
    assert by_check["host dependency pyarrow"]["status"] == table.FAIL
    assert "NumPy 2.0" in by_check["host dependency pyarrow"]["detail"]
    assert by_check["host dependency torch"]["status"] == table.OK


def test_a_compiler_older_than_the_documented_one_is_noticed():
    assert doctor._older_than("2.1.0", "2.1.3")
    assert not doctor._older_than("2.1.3", "2.1.3")
    assert not doctor._older_than("2.2.0", "2.1.3")
    # A development build string is reported as found, not guessed at.
    assert not doctor._older_than("main-abc123", "2.1.3")


def test_import_probe_does_not_expose_foreign_site_packages(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    from polima.cli import doctor

    package = Path(doctor.__file__).resolve().parents[1]
    copied = tmp_path / 'site-packages/polima'
    shutil.copytree(package, copied, ignore=shutil.ignore_patterns('__pycache__', 'static', 'ui'))
    (copied.parent / 'numpy.py').write_text('raise RuntimeError("foreign NumPy imported")')
    monkeypatch.setattr(doctor, '__file__', str(copied / 'cli/doctor.py'))
    monkeypatch.setattr(doctor, '_find_compiler_bin', lambda config: None)
    report = doctor.Doctor(json_output=True)
    doctor.check_import_matrix(report, None)
    assert report.results[0]['status'] == 'ok', report.results


def test_deploy_only_skips_compiler_and_training_checks(ran, monkeypatch):
    _as(monkeypatch, HOST)
    monkeypatch.setattr(doctor, 'check_host', lambda doc, host_side: pytest.fail('training checks') if host_side else None)
    monkeypatch.setattr(doctor, 'check_import_matrix', lambda doc, config, **kw: pytest.fail('compiler imports') if kw.get('include_compiler', True) else None)
    assert doctor.run(['--deploy-only', '--no-devkit']) == 0
    assert not any(name in ran for name in HOST_SECTIONS)
