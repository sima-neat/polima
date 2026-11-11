"""Host tests for the native runtime/platform compatibility gate."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMPILER = shutil.which("g++") or shutil.which("clang++")


@pytest.fixture(scope="module")
def checker(tmp_path_factory):
    if COMPILER is None:
        pytest.skip("no C++ compiler available")
    root = tmp_path_factory.mktemp("native-compatibility")
    source = root / "check.cpp"
    source.write_text(
        '''#include "polima/compatibility.hpp"
#include <iostream>
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  const auto result = polima::runtime_compatibility(argv[1], argv[2]);
  std::cout << (result.compatible ? "ok" : "fail") << "\\n" << result.detail << "\\n";
  return result.compatible ? 0 : 1;
}
'''
    )
    binary = root / "check"
    subprocess.run(
        [
            COMPILER,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            '-DPOLIMA_EXPECTED_PLATFORM="3.0.0"',
            '-DPOLIMA_EXPECTED_LLIMA="0.6.0"',
            f"-I{ROOT / 'runtime/include'}",
            str(source),
            str(ROOT / "runtime/src/compatibility.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
    )
    return binary


def files(tmp_path, *, platform="3.0.0", llima="0.6.0"):
    buildinfo = tmp_path / "buildinfo"
    status = tmp_path / "status"
    # This is the spacing used by the released Modalix 3.0 B2501 image.
    buildinfo.write_text(f"MACHINE = modalix\nDISTRO_VERSION = {platform}\n")
    status.write_text(
        "Package: unrelated\nVersion: 1\n\n"
        f"Package: sima-lmm-core\nStatus: install ok installed\nVersion: {llima}\n\n"
    )
    return buildinfo, status


def run(checker, paths):
    return subprocess.run(
        [str(checker), *(str(path) for path in paths)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_matching_platform_and_llima_are_accepted(checker, tmp_path):
    result = run(checker, files(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("ok\n")


@pytest.mark.parametrize(
    "platform, llima, reason",
    [
        ("2.1.3", "0.6.0", "installed platform 2.1.3"),
        ("3.0.0", "0.5.0", "sima-lmm-core 0.5.0"),
        ("3.0.0", "0.6.0+develop.9ab65e87d04c", "sima-lmm-core 0.6.0+develop"),
    ],
)
def test_a_different_platform_or_llima_is_refused(checker, tmp_path, platform, llima, reason):
    result = run(checker, files(tmp_path, platform=platform, llima=llima))
    assert result.returncode == 1
    assert reason in result.stdout


def test_missing_identities_are_reported(checker, tmp_path):
    buildinfo, status = files(tmp_path)
    status.write_text("Package: unrelated\nStatus: install ok installed\nVersion: 1\n\n")
    result = run(checker, (buildinfo, status))
    assert result.returncode == 1 and "sima-lmm-core is not installed" in result.stdout
    result = run(checker, (tmp_path / "absent", status))
    assert result.returncode == 1 and "platform version is unavailable" in result.stdout


def test_a_removed_package_does_not_count_as_installed(checker, tmp_path):
    buildinfo, status = files(tmp_path)
    status.write_text(status.read_text().replace("install ok installed", "deinstall ok config-files"))
    assert run(checker, (buildinfo, status)).returncode == 1
