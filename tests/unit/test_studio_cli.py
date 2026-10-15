from __future__ import annotations

from argparse import Namespace
from unittest.mock import Mock

from polima.cli import studio


def test_no_action_reports_status(monkeypatch, capsys) -> None:
    monkeypatch.setattr(studio.shutil, "which", lambda _: "/usr/bin/systemctl")
    results = iter([
        Mock(stdout="inactive\n"),
        Mock(stdout="disabled\n"),
    ])
    monkeypatch.setattr(studio.subprocess, "run", lambda *args, **kwargs: next(results))
    monkeypatch.setattr(studio, "_service_url", lambda: "http://192.0.2.1:8080")
    assert studio.run([]) == 0
    assert capsys.readouterr().out == (
        "service: inactive\nboot:    disabled\nurl:     http://192.0.2.1:8080\n"
    )


def test_start_does_not_enable(monkeypatch, capsys) -> None:
    called = []
    monkeypatch.setattr(studio.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(studio, "_wait_until_ready", lambda: True)
    monkeypatch.setattr(studio, "_service_url", lambda: "http://192.0.2.1:8080")
    monkeypatch.setattr(
        studio.subprocess,
        "run",
        lambda command, **kwargs: called.append(command) or Mock(returncode=0),
    )
    assert studio.run(["start"]) == 0
    assert called == [["sudo", "-n", "systemctl", "start", studio.SERVICE]]
    assert capsys.readouterr().out == (
        "PoLiMa Studio started successfully.\nOpen: http://192.0.2.1:8080\n"
    )


def test_restart_waits_for_readiness(monkeypatch, capsys) -> None:
    monkeypatch.setattr(studio.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        studio.subprocess, "run", lambda *args, **kwargs: Mock(returncode=0)
    )
    monkeypatch.setattr(studio, "_wait_until_ready", lambda: True)
    monkeypatch.setattr(studio, "_service_url", lambda: "http://192.0.2.1:8080")
    assert studio.run(["restart"]) == 0
    assert capsys.readouterr().out == (
        "PoLiMa Studio restarted successfully.\nOpen: http://192.0.2.1:8080\n"
    )


def test_start_reports_readiness_timeout(monkeypatch, capsys) -> None:
    monkeypatch.setattr(studio.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        studio.subprocess, "run", lambda *args, **kwargs: Mock(returncode=0)
    )
    monkeypatch.setattr(studio, "_wait_until_ready", lambda: False)
    assert studio.run(["start"]) == 1
    assert "did not become ready within 20 seconds" in capsys.readouterr().err


def test_start_failure_does_not_report_success(monkeypatch, capsys) -> None:
    monkeypatch.setattr(studio.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        studio.subprocess, "run", lambda *args, **kwargs: Mock(returncode=5)
    )
    wait = Mock(return_value=True)
    monkeypatch.setattr(studio, "_wait_until_ready", wait)
    assert studio.run(["start"]) == 5
    wait.assert_not_called()
    assert capsys.readouterr().out == ""


def test_disable_stops_and_disables(monkeypatch) -> None:
    called = []
    monkeypatch.setattr(studio.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        studio.subprocess,
        "run",
        lambda command, **kwargs: called.append(command) or Mock(returncode=0),
    )
    assert studio.run(["disable"]) == 0
    assert called == [["systemctl", "disable", "--now", studio.SERVICE]]


def test_dry_run_does_not_mutate(monkeypatch, capsys) -> None:
    monkeypatch.setattr(studio.os, "geteuid", lambda: 1000)
    assert studio.run(["restart"], Namespace(dry_run=True)) == 0
    assert capsys.readouterr().out == (
        "+ sudo -n systemctl restart polima-studio.service\n"
    )


def test_management_has_no_studio_dependency() -> None:
    assert studio.needs_capability([]) is None
    assert studio.needs_capability(["status"]) is None
    assert studio.needs_capability(["serve"]) == "studio"
    assert studio.needs_capability(["--port", "9000"]) == "studio"


def test_service_url_uses_ssh_device_address(monkeypatch):
    monkeypatch.delenv("POLIMA_STUDIO_PUBLIC_URL", raising=False)
    monkeypatch.setenv("POLIMA_STUDIO_HOST", "0.0.0.0")
    monkeypatch.setenv("POLIMA_STUDIO_PORT", "8080")
    monkeypatch.setenv("SSH_CONNECTION", "192.0.2.2 50000 192.0.2.10 22")
    assert studio._service_url() == "http://192.0.2.10:8080"
    monkeypatch.setenv("SSH_CONNECTION", "2001:db8::2 50000 2001:db8::10 22")
    assert studio._service_url() == "http://[2001:db8::10]:8080"
    monkeypatch.setenv("POLIMA_STUDIO_HOST", "127.0.0.1")
    assert studio._service_url() == "http://127.0.0.1:8080"
    monkeypatch.setenv("POLIMA_STUDIO_PUBLIC_URL", "https://studio.example/")
    assert studio._service_url() == "https://studio.example"


def test_lan_address_falls_back_to_route_without_dns(monkeypatch):
    from polima.util import net
    monkeypatch.setenv("SSH_CONNECTION", "invalid")
    probe = Mock()
    probe.getsockname.return_value = ("192.0.2.20", 45000)
    monkeypatch.setattr(net.socket, "socket", lambda *a: probe)
    assert net.lan_address() == "192.0.2.20"
    probe.close.assert_called_once()
    probe.connect.side_effect = OSError("no route")
    assert net.lan_address() == "127.0.0.1"
