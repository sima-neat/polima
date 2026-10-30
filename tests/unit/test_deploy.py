"""Deploy logic that can be tested without a DevKit.

These cover the pure logic, especially shell quoting and command shape, which
otherwise fail only on hardware.
"""

from __future__ import annotations

import inspect
import json
import sys

import numpy as np
import pytest

from polima.cli import deploy as deploy_cli
from polima.cli import run as run_cli
from polima.config.base import BoardConfig
from polima.deploy import deploy
from polima.deploy.build import source_hash
from polima.deploy.smoke import (
    DEFAULT_COSINE_MIN,
    DEFAULT_MEAN_ABS_MAX,
    SmokeReport,
    compare,
    compare_against_reference,
)

# ------------------------------------------------------------------- config


def test_deploy_does_not_start_service_by_default():
    assert inspect.signature(deploy).parameters["start_service"].default is False


def test_board_paths():
    board = BoardConfig(root="/media/nvme/polima")
    # `models`, not `bundles`: PoLiMa deploys alongside the hand-built trees
    # already on the board rather than creating a second place to look.
    assert board.bundles_dir == "/media/nvme/polima/models"
    assert board.current_link == "/media/nvme/polima/current"
    assert board.path("models", "abc") == "/media/nvme/polima/models/abc"
    assert board.path("var/log") == "/media/nvme/polima/var/log"
    assert board.bin_dir == "/media/nvme/polima/bin"


def test_board_address_and_user():
    board = BoardConfig(host="sima@192.0.2.10")
    assert board.user == "sima"
    assert board.address == "192.0.2.10"
    assert BoardConfig(host="192.0.2.10").address == "192.0.2.10"


def test_trailing_slashes_do_not_double_up():
    board = BoardConfig(root="/media/nvme/polima/")
    assert board.path("models") == "/media/nvme/polima/models"


def test_deploy_directory_is_configurable():
    """The board's model store is a layout choice, not a constant."""
    assert BoardConfig(bundles_subdir="bundles").bundles_dir.endswith("/bundles")


def test_default_build_jobs_uses_the_devkit_cores():
    """The DevKit has 16 cores; the build must not default to -j2."""
    assert BoardConfig().build_jobs > 2


@pytest.mark.parametrize("resolver", [deploy_cli._resolve_bundle, run_cli._resolve_bundle])
def test_bundle_path_expands_quoted_home(resolver, tmp_path, monkeypatch):
    """The shell does not expand ``~`` inside a quoted --bundle value."""
    bundle_dir = tmp_path / "downloads" / "model" / "bundle"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "bundle.json").write_text(
        json.dumps({"bundle_id": "groot-test-0-deadbeef", "policy": "groot"})
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    bundle = resolver("~/downloads/model/bundle", None)

    assert bundle.root == bundle_dir


def test_board_capture_keeps_remote_stderr_out_of_parsed_stdout(monkeypatch):
    """A remote shell warning must not become a filename or a df value."""
    from polima.deploy import ssh
    from polima.util.proc import Result

    def fake_run(argv, **kwargs):
        assert kwargs["merge_stderr"] is False
        return Result(
            list(argv),
            0,
            0.01,
            stdout="407000000000\n",
            stderr="bash: warning: setlocale: cannot change locale\n",
        )

    monkeypatch.setattr(ssh, "run", fake_run)
    session = ssh.BoardSession(BoardConfig(host="sima@example", control_master=False))

    assert session.capture("df --output=avail /media/nvme | tail -1") == "407000000000"
    assert session.free_bytes("/media/nvme/polima") == 407000000000


def test_process_runner_can_capture_stdout_and_stderr_separately():
    from polima.util.proc import run

    result = run(
        [
            sys.executable,
            "-c",
            "import os; print('407000000000'); os.write(2, b'locale warning: \\xff\\n')",
        ],
        echo=False,
        merge_stderr=False,
    )

    assert result.stdout == "407000000000\n"
    assert result.stderr == "locale warning: �\n"


# -------------------------------------------------------------- build hashing


def test_source_hash_is_deterministic():
    assert source_hash() == source_hash()


def test_source_hash_changes_with_content(tmp_path):
    (tmp_path / "a.cpp").write_text("int main(){}")
    (tmp_path / "CMakeLists.txt").write_text("project(x)")
    first = source_hash(tmp_path)
    (tmp_path / "a.cpp").write_text("int main(){return 1;}")
    assert source_hash(tmp_path) != first


def test_source_hash_ignores_unrelated_files(tmp_path):
    (tmp_path / "a.cpp").write_text("int main(){}")
    before = source_hash(tmp_path)
    (tmp_path / "notes.md").write_text("hello")
    (tmp_path / "build.log").write_text("noise")
    assert source_hash(tmp_path) == before


# --------------------------------------------------------------------- smoke


def test_identical_arrays_pass():
    values = np.arange(600, dtype=np.float32)
    result = compare("x", values, values)
    assert result.ok
    assert result.cosine == pytest.approx(1.0)
    assert result.mean_abs == 0.0


def test_default_thresholds():
    """The default bar: cosine >= 0.999 and mean_abs <= 0.01."""
    assert DEFAULT_COSINE_MIN == 0.999
    assert DEFAULT_MEAN_ABS_MAX == 0.01


def test_small_perturbation_still_passes():
    rng = np.random.default_rng(0)
    expected = rng.standard_normal(600).astype(np.float32)
    actual = expected + rng.standard_normal(600).astype(np.float32) * 0.001
    assert compare("x", actual, expected).ok


def test_large_perturbation_fails():
    rng = np.random.default_rng(0)
    expected = rng.standard_normal(600).astype(np.float32)
    actual = expected + rng.standard_normal(600).astype(np.float32) * 0.5
    assert not compare("x", actual, expected).ok


def test_size_mismatch_fails_without_raising():
    result = compare("x", np.zeros(600, np.float32), np.zeros(300, np.float32))
    assert not result.ok
    assert "size mismatch" in result.note


def test_reference_comparison_is_far_stricter():
    """Two servers running the same ELFs on the same MLA must agree almost
    exactly; only the host-side glue could differ."""
    rng = np.random.default_rng(0)
    expected = rng.standard_normal(600).astype(np.float32)
    drifted = expected + 0.001

    assert compare("loose", drifted, expected).ok          # fine vs pytorch
    assert not compare_against_reference(drifted, expected).ok  # not fine vs a peer
    assert compare_against_reference(expected, expected).ok


def test_report_aggregates():
    report = SmokeReport()
    values = np.ones(10, dtype=np.float32)
    report.add(compare("a", values, values))
    assert report.ok
    report.add(compare("b", values, values * 5))
    assert not report.ok
    assert len(report.to_dict()["results"]) == 2


def test_summary_is_readable():
    values = np.ones(10, dtype=np.float32)
    assert "PASS" in compare("a", values, values).summary()
    assert "FAIL" in compare("b", values, values * 9).summary()


# ---------------------------------------------------------- camera capture


def test_camera_config_emits_mjpg():
    """Two 640x480@30 USB cameras exceed the bus budget in uncompressed YUYV.

    The cameras must use `fourcc: MJPG`; regressing this degrades control
    silently, so it is pinned here.
    """
    from polima.policies.act import ACT_SPEC

    blob = ACT_SPEC.robot.camera_config(
        {"overhead": "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920-video-index0",
         "wrist": "/dev/v4l/by-id/usb-Sonix_CAM1-video-index0"}
    )
    assert blob.count("fourcc: MJPG") == 2
    assert "overhead:" in blob and "wrist:" in blob
    assert "width: 640" in blob and "height: 480" in blob
    assert "fps: 30" in blob


def test_camera_config_honours_fps_override():
    from polima.policies.act import ACT_SPEC

    assert "fps: 15" in ACT_SPEC.robot.camera_config({"overhead": "/dev/x"}, fps=15)


def test_camera_config_skips_absent_devices():
    from polima.policies.act import ACT_SPEC

    blob = ACT_SPEC.robot.camera_config({"overhead": "/dev/x"})
    assert "overhead:" in blob and "wrist:" not in blob


# --------------------------------------------------------------- MLA recovery
#
# A SIGKILLed server leaves DMA buffers behind, CMA fragments, and every
# later load fails with MLA_LOAD_FAILED as if the ELF were corrupt.


class _FakeResult:
    def __init__(self, returncode: int = 0, stdout: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout


class _FakeSession:
    """Records commands and answers `test -x` from a set of present paths."""

    def __init__(self, present: tuple[str, ...] = (), cma: tuple[int, ...] = (100, 200),
                 fail: tuple[str, ...] = ()) -> None:
        self.present = present
        self.commands: list[str] = []
        self._cma = list(cma)
        self._fail = fail

    def run(self, command: str, *, check: bool = True, echo: bool = False,
            timeout: float | None = None) -> _FakeResult:
        self.commands.append(command)
        if command.startswith("test -x"):
            return _FakeResult(0 if any(p in command for p in self.present) else 1)
        if any(marker in command for marker in self._fail):
            return _FakeResult(1)
        return _FakeResult(0)

    def capture(self, command: str) -> str:
        self.commands.append(command)
        if "CmaFree" in command:
            return f"CmaFree: {self._cma.pop(0) if self._cma else 0} kB"
        return ""


def test_mla_wedge_is_recognised_only_from_accelerator_errors():
    from polima.deploy import mla

    assert mla.looks_wedged("fatal: ... errCode=1001 name=MLA_LOAD_FAILED")
    assert mla.looks_wedged("Failed to load model through MLASHM dispatcher")
    assert mla.looks_wedged("simaai-memory: Could not allocate buffer")
    # A bundle that is simply wrong must NOT trigger an accelerator reset.
    assert not mla.looks_wedged("bundle.json names a missing ELF")
    assert not mla.looks_wedged("port 8081 is served by pid(s) [42]")


def test_mla_reset_prefers_the_sdk_recovery_script():
    """fix_devkit_runtime.sh re-inits the memory pool before restarting the
    services, and it is that step -- not the service restart -- that
    defragments CMA. Restarting the dispatcher alone is the weaker fallback."""
    from polima.deploy import mla

    session = _FakeSession(present=("/usr/bin/fix_devkit_runtime.sh",))
    report = mla.reset(session, BoardConfig())

    assert report.ok and report.method == "recovery-script"
    assert any("fix_devkit_runtime.sh" in c for c in session.commands)
    assert not any("systemctl restart" in c for c in session.commands)


def test_mla_reset_falls_back_to_services_without_a_recovery_script():
    from polima.deploy import mla

    session = _FakeSession(present=("/usr/bin/init_mla_memory.sh",))
    report = mla.reset(session, BoardConfig())

    assert report.ok and report.method == "services"
    assert any(mla.DISPATCHER_SERVICE in c for c in session.commands)
    assert any("init_mla_memory.sh" in c for c in session.commands)


def test_mla_reset_tries_passwordless_sudo_before_sending_a_password():
    from polima.deploy import mla

    session = _FakeSession(present=("/usr/bin/fix_devkit_runtime.sh",))
    mla.reset(session, BoardConfig(), password="hunter2")
    sudo = next(c for c in session.commands if "fix_devkit_runtime.sh" in c
                and c.startswith("sudo"))
    assert sudo.index("sudo -n") < sudo.index("sudo -S"), "must try NOPASSWD first"
    assert "-p ''" in sudo, "the prompt must be suppressed so it cannot reach a log"


def test_mla_reset_without_a_password_only_tries_passwordless_sudo(monkeypatch, caplog):
    """No password ships with PoLiMa: without POLIMA_MLA_SUDO_PASSWORD the reset
    uses `sudo -n` alone and, if it fails, says how to fix it."""
    from polima.deploy import mla

    monkeypatch.delenv(mla.SUDO_PASSWORD_ENV, raising=False)
    session = _FakeSession(present=("/usr/bin/fix_devkit_runtime.sh",),
                           fail=("fix_devkit_runtime.sh",))
    with caplog.at_level("WARNING", logger="polima"):
        report = mla.reset(session, BoardConfig())
    sudo = [c for c in session.commands if c.startswith("sudo")]
    assert sudo == ["sudo -n /usr/bin/fix_devkit_runtime.sh"]
    assert report.ok is False
    assert mla.SUDO_PASSWORD_ENV in caplog.text


def test_mla_reset_takes_the_password_from_the_environment(monkeypatch):
    from polima.deploy import mla

    monkeypatch.setenv(mla.SUDO_PASSWORD_ENV, "hunter2")
    session = _FakeSession(present=("/usr/bin/fix_devkit_runtime.sh",))
    mla.reset(session, BoardConfig())
    sudo = next(c for c in session.commands if c.startswith("sudo"))
    assert "sudo -S" in sudo and "hunter2" in sudo


def test_mla_reset_reports_reclaimed_cma():
    from polima.deploy import mla

    session = _FakeSession(present=("/usr/bin/fix_devkit_runtime.sh",),
                           cma=(1_031_888, 1_748_688))
    report = mla.reset(session, BoardConfig())
    assert report.reclaimed_kb == 716_800


def test_mla_reset_reports_failure_rather_than_raising():
    """A failed reset must not mask the load error that prompted it."""
    from polima.deploy import mla

    session = _FakeSession(present=("/usr/bin/fix_devkit_runtime.sh",),
                           fail=("fix_devkit_runtime.sh",))
    report = mla.reset(session, BoardConfig())
    assert report.ok is False


def test_service_start_resets_and_retries_once_on_a_wedged_mla(monkeypatch):
    from polima.deploy import mla, service

    calls: list[int] = []

    def fake_launch(session, board, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("server.log: errCode=1001 name=MLA_LOAD_FAILED")
        return "started"

    monkeypatch.setattr(service, "_launch", fake_launch)
    monkeypatch.setattr(mla, "reset", lambda *a, **k: mla.ResetReport(True, "recovery-script"))

    assert service.start(None, BoardConfig(), port=8081) == "started"
    assert len(calls) == 2, "exactly one retry, after the reset"


def test_service_start_does_not_reset_for_an_unrelated_failure(monkeypatch):
    from polima.deploy import mla, service

    resets: list[int] = []
    monkeypatch.setattr(service, "_launch", _raise_missing_elf)
    monkeypatch.setattr(mla, "reset", lambda *a, **k: resets.append(1))

    with pytest.raises(RuntimeError, match="missing ELF"):
        service.start(None, BoardConfig(), port=8081)
    assert not resets, "a bad bundle must not trigger an accelerator reset"


def _raise_missing_elf(session, board, **kwargs):
    raise RuntimeError("bundle.json names a missing ELF")


def test_service_start_reraises_the_load_error_when_the_reset_fails(monkeypatch):
    from polima.deploy import mla, service

    monkeypatch.setattr(service, "_launch", _raise_mla_wedge)
    monkeypatch.setattr(mla, "reset", lambda *a, **k: mla.ResetReport(False, "services"))

    with pytest.raises(RuntimeError, match="MLA_LOAD_FAILED"):
        service.start(None, BoardConfig(), port=8081)


def _raise_mla_wedge(session, board, **kwargs):
    raise RuntimeError("errCode=1001 name=MLA_LOAD_FAILED")


def test_there_is_no_default_board_address(monkeypatch):
    """A board address is site configuration, never a shipped literal."""
    from polima.config.loader import load

    monkeypatch.delenv("POLIMA_BOARD", raising=False)
    monkeypatch.delenv("POLIMA_BOARD_HOST", raising=False)
    assert BoardConfig().host == ""
    monkeypatch.setenv("POLIMA_BOARD", "sima@192.0.2.10")
    assert load(use_env=True).board.host == "sima@192.0.2.10"


def test_the_example_site_config_loads(tmp_path):
    pytest.importorskip("yaml")
    from polima.config.loader import load
    from polima.util.paths import package_root

    example = tmp_path / "site.yaml"
    example.write_text((package_root() / ".polima.yaml.example").read_text())
    config = load(config_file=example, use_env=False)
    assert config.board.host == "sima@<board-address>"
    assert config.board.root == "/media/nvme/polima"


def test_deploy_without_a_board_says_how_to_name_one(monkeypatch, capsys):
    from polima.cli import deploy as deploy_cli

    monkeypatch.delenv("POLIMA_BOARD", raising=False)
    monkeypatch.delenv("POLIMA_BOARD_HOST", raising=False)
    monkeypatch.setattr("polima.config.loader.config_file_candidates", list)
    assert deploy_cli.run(["--status"]) == 2
    assert "POLIMA_DEVKIT" in capsys.readouterr().err


@pytest.mark.parametrize("layer, expected", [("user", "user/model"), ("env", "env/model"),
                                             ("file", "file/model"), ("cli", "cli/model")])
def test_persistent_config_precedence(tmp_path, monkeypatch, layer, expected):
    import json

    from polima.config import loader

    package = tmp_path / "package.json"
    package.write_text(json.dumps({"paths": {"groot17_backbone": "package/model"},
                                   "log_level": "DEBUG"}))
    monkeypatch.setattr(loader, "config_file_candidates", lambda: [package])
    user = tmp_path / "config/polima/config.json"
    user.parent.mkdir(parents=True)
    user.write_text(json.dumps({"paths": {"groot17_backbone": "user/model"}}))
    monkeypatch.delenv("POLIMA_GROOT17_BACKBONE", raising=False)
    monkeypatch.delenv("POLIMA_LOG_LEVEL", raising=False)
    explicit = tmp_path / "explicit.json"
    explicit.write_text(json.dumps({"paths": {"groot17_backbone": "file/model"}}))
    if layer in ("env", "file", "cli"):
        monkeypatch.setenv("POLIMA_GROOT17_BACKBONE", "env/model")
    config = loader.load(config_file=explicit if layer in ("file", "cli") else None,
                         overrides={"paths.groot17_backbone": "cli/model"} if layer == "cli" else None)
    assert config.paths.groot17_backbone == expected
    assert config.log_level == "DEBUG"
    assert loader.load(use_env=False).paths.groot17_backbone == "user/model"


def test_missing_user_config_preserves_package_defaults(tmp_path, monkeypatch):
    from polima.config import loader

    package = tmp_path / "package.json"
    package.write_text('{"log_level":"DEBUG"}')
    monkeypatch.setattr(loader, "config_file_candidates", lambda: [package])
    config = loader.load(use_env=False)
    assert config.log_level == "DEBUG"
    assert config.paths.groot17_backbone == ""
    assert not (tmp_path / "config").exists()


def test_user_config_defaults_to_home_config_directory(tmp_path, monkeypatch):
    from pathlib import Path

    from polima.config import loader

    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    user = tmp_path / ".config/polima/config.json"
    user.parent.mkdir(parents=True)
    user.write_text('{"paths":{"groot17_backbone":"customer/backbone"}}')
    assert loader.load(use_env=False).paths.groot17_backbone == "customer/backbone"
