"""Host build of native/tests/test_kernels.cpp.

The kernels are header-only and need no MLA runtime, so the scalar fallbacks
and the worker-pool chunking are checked here on every host run. The NEON
paths are only exercised on the board (`build/polima_kernel_tests`).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

NATIVE = Path(__file__).resolve().parents[2] / "native"
COMPILER = shutil.which("g++") or shutil.which("clang++")


@pytest.fixture(scope="module")
def kernel_tests(tmp_path_factory) -> Path:
    if COMPILER is None:
        pytest.skip("no C++ compiler on this host")
    binary = tmp_path_factory.mktemp("kernels") / "test_kernels"
    subprocess.run(
        [COMPILER, "-O2", "-std=gnu++20", "-Wall", "-Wextra", "-Werror", "-pthread",
         f"-I{NATIVE / 'include'}", str(NATIVE / "tests" / "test_kernels.cpp"),
         "-o", str(binary)],
        check=True,
    )
    return binary


@pytest.mark.parametrize(
    ("threads", "threshold"),
    [("1", "131072"), ("4", "131072"), ("5", "1000"), ("3", "1")],
)
def test_kernels_bit_exact(kernel_tests: Path, threads: str, threshold: str) -> None:
    env = dict(os.environ, POLIMA_CPU_THREADS=threads, POLIMA_PARALLEL_MIN_VALUES=threshold,
               POLIMA_PARALLEL_MIN_COMPUTE=threshold)
    result = subprocess.run(
        [str(kernel_tests)], env=env, capture_output=True, text=True, timeout=300, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all kernel tests passed" in result.stdout
