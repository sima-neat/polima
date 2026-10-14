"""Exercise the native fixture checker without an accelerator."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def checker(tmp_path_factory):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    root = tmp_path_factory.mktemp("native-smoke")
    source = root / "check.cpp"
    source.write_text('''#include "polima/smoke.hpp"
#include <iostream>
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  try {
    const std::filesystem::path root(argv[1]);
    const auto result = polima::compare_fixture(root,
      polima::read_f32(root / "actual"), polima::read_f32(root / "expected"),
      polima::read_f32(root / "state"));
    std::cout << result.cosine << " " << result.mean_abs << " " << result.max_abs;
  } catch (const std::exception& e) { std::cerr << e.what(); return 1; }
}
''')
    binary = root / "check"
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-I" + str(ROOT / "runtime/include"),
                    str(source), "-o", str(binary)], check=True)
    return binary


def fixture(root, policy="groot", relative=1):
    (root / "bundle.json").write_text(json.dumps({"policy": policy}))
    (root / "plan.json").write_text(json.dumps({"result": "action_chunk", "profile": {"action_dim": 2},
                                               "runtime_context": {"relative_action_dim": relative}}))
    (root / "constants").mkdir()
    values = {"actual": [101, 22, 121, 42], "expected": [100, 20, 120, 40],
              "state": [10, 5], "constants/action_mean": [0, 0, 0, 0],
              "constants/action_std": [100, 200, 100, 200]}
    for name, data in values.items():
        np.asarray(data, dtype="<f4").tofile(root / name)


@pytest.mark.parametrize("policy", ["act", "smolvla", "groot"])
def test_policy_comparison_units(checker, tmp_path, policy):
    fixture(tmp_path, policy)
    result = subprocess.run([str(checker), str(tmp_path)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    cosine, mean_abs, maximum = map(float, result.stdout.split())
    assert cosine > 0.995
    assert mean_abs == pytest.approx(0.01 if policy == "groot" else 1.5, abs=1e-6)
    assert maximum == pytest.approx(0.01 if policy == "groot" else 2, abs=1e-6)


@pytest.mark.parametrize("file,values", [
    ("constants/action_std", [0, 200, 100, 200]),
    ("constants/action_std", [-1, 200, 100, 200]),
    ("constants/action_std", [float("nan"), 200, 100, 200]),
    ("constants/action_mean", [0]),
    ("actual", [float("nan"), 22, 121, 42]),
    ("expected", [float("inf"), 20, 120, 40]),
    ("expected", [100]),
    ("state", []),
])
def test_invalid_comparison_fails_closed(checker, tmp_path, file, values):
    fixture(tmp_path)
    np.asarray(values, dtype="<f4").tofile(tmp_path / file)
    result = subprocess.run([str(checker), str(tmp_path)], capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert result.stderr


def test_bad_predictions_are_not_hidden_by_normalization(checker, tmp_path):
    fixture(tmp_path)
    np.array([200, 220, 220, 240], dtype="<f4").tofile(tmp_path / "actual")
    result = subprocess.run([str(checker), str(tmp_path)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert float(result.stdout.split()[1]) > 0.03


@pytest.mark.parametrize("relative", [0, 1])
def test_native_matches_python_normalization(checker, tmp_path, relative):
    from polima.deploy.smoke import compare_normalized_actions

    fixture(tmp_path, relative=relative)
    mean = np.array([[30, -10], [20, -30]], dtype="<f4")
    mean.tofile(tmp_path / "constants/action_mean")
    def read(name):
        return np.fromfile(tmp_path / name, dtype="<f4")
    expected = compare_normalized_actions(
        read("actual"), read("expected"), mean,
        read("constants/action_std").reshape(2, 2),
        state=read("state"), relative_dim=relative,
    )
    result = subprocess.run([str(checker), str(tmp_path)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert list(map(float, result.stdout.split())) == pytest.approx(
        [expected.cosine, expected.mean_abs, expected.max_abs], abs=1e-6,
    )


def test_act_does_not_require_groot_constants(checker, tmp_path):
    fixture(tmp_path, "act")
    shutil.rmtree(tmp_path / "constants")
    (tmp_path / "plan.json").unlink()
    result = subprocess.run([str(checker), str(tmp_path)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert float(result.stdout.split()[1]) == 1.5
