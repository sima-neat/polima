"""Exercise native stop against local processes, without an accelerator."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _stop_binary(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    source = tmp_path / "stop.cpp"
    source.write_text('''#include "polima/service.hpp"
#include <iostream>
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  try { polima::stop_server(argv[1], 0.2); }
  catch (const std::exception& e) { std::cerr << e.what(); return 1; }
}
''')
    binary = tmp_path / "stop"
    subprocess.run([compiler, "-std=c++17", "-I" + str(ROOT / "runtime/include"),
                    str(source), str(ROOT / "runtime/src/service.cpp"), "-o", str(binary)],
                   check=True)
    return binary


def _process(name, ignore_term):
    trap = "trap '' TERM; " if ignore_term else ""
    child = subprocess.Popen(["bash", "-c", f"{trap}echo ready; exec -a {name} sleep 30"],
                             stdout=subprocess.PIPE)
    assert child.stdout.readline().strip() == b"ready"
    return child


@pytest.mark.parametrize("ignore_term", [True, False])
def test_stop_preserves_live_server_on_timeout(tmp_path, ignore_term):
    binary = _stop_binary(tmp_path)
    child = _process("polima_server", ignore_term)
    try:
        pidfile = tmp_path / "var/run/server.pid"
        pidfile.parent.mkdir(parents=True)
        pidfile.write_text(str(child.pid))
        result = subprocess.run([str(binary), str(tmp_path)], capture_output=True, text=True,
                                check=False)
        if ignore_term:
            assert result.returncode == 1
            assert child.poll() is None
            assert pidfile.exists()
        else:
            assert result.returncode == 0, result.stderr
            assert not pidfile.exists()
            child.wait(timeout=2)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()


def test_stale_pid_file_never_signals_another_program(tmp_path):
    binary = _stop_binary(tmp_path)
    child = _process("unrelated", ignore_term=False)
    try:
        pidfile = tmp_path / "var/run/server.pid"
        pidfile.parent.mkdir(parents=True)
        pidfile.write_text(str(child.pid))
        result = subprocess.run([str(binary), str(tmp_path)], capture_output=True, text=True,
                                check=False)
        assert result.returncode == 0, result.stderr
        assert child.poll() is None
        assert not pidfile.exists()
    finally:
        child.kill()
        child.wait()
