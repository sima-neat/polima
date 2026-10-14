from types import SimpleNamespace

import pytest

from polima.config.base import BoardConfig
from polima.deploy import service


def _session(program, commands):
    def capture(command):
        return program if "/cmdline" in command else "42"

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    return SimpleNamespace(capture=capture, run=run)


def test_stop_timeout_keeps_pid_and_does_not_force_kill():
    commands = []
    session = _session("/media/nvme/polima/bin/polima_server", commands)
    with pytest.raises(RuntimeError, match="did not stop cleanly"):
        service.stop(session, BoardConfig(), timeout=0)
    assert not any("kill -9" in command or "rm -f" in command for command in commands)


def test_stale_pid_file_never_signals_another_program():
    commands = []
    session = _session("/usr/sbin/sshd", commands)
    assert service.stop(session, BoardConfig(), timeout=0) is False
    assert not any(command.startswith("kill 42") for command in commands)
