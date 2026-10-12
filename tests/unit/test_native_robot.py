"""Host build of runtime/tests/test_robot.cpp: SO-101 camera roles from a bundle."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

NATIVE = Path(__file__).resolve().parents[2] / "runtime"
COMPILER = shutil.which("g++") or shutil.which("clang++")


@pytest.mark.skipif(COMPILER is None, reason="no C++ compiler on this host")
def test_each_so101_camera_lands_in_its_own_role(tmp_path):
    binary = tmp_path / "test_robot"
    subprocess.run([COMPILER, "-O1", "-std=gnu++20", "-Wall", "-Wextra", "-Werror", f"-I{NATIVE / 'include'}",
                    str(NATIVE / "tests" / "test_robot.cpp"), str(NATIVE / "src" / "robot.cpp"), "-o", str(binary)],
                   check=True)
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
