"""uint8 camera frames: the native decode equals the Python reference bit for bit.

The server decodes each uint8 request tensor before the plan runs, so a plan fed
raw frames must produce the same bytes as the same plan fed the frames that
`polima.wire.protocol.decode_tensor` normalized on the host.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_device_chain import mini_act

from polima.bundle.device_chain import rewrite
from polima.wire.protocol import decode_tensor

ROOT = Path(__file__).resolve().parents[2]
DECODES = ({"scale": 255.0, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
           {"scale": 255.0, "mean": [0.1, 0.2, 0.3], "std": [2.0, 0.5, 0.25]})


@pytest.fixture(scope="module")
def run_plan(tmp_path_factory):
    compiler = shutil.which("g++")
    if not compiler or not Path("/usr/include/nlohmann/json.hpp").is_file():
        pytest.skip("needs g++ and nlohmann/json")
    binary = tmp_path_factory.mktemp("native") / "run_plan"
    native = ROOT / "tests" / "native"
    subprocess.run([compiler, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror",
                    f"-I{native / 'llima_mock'}", f"-I{ROOT / 'runtime/include'}",
                    str(native / "run_plan.cpp"), str(ROOT / "runtime/src/plan.cpp"),
                    "-o", str(binary)], check=True, capture_output=True, text=True)
    return binary


def _uint8_bundle(root: Path, chained: bool) -> tuple[Path, dict]:
    bundle = mini_act(root)
    plan = json.loads((bundle / "plan.json").read_text())
    graphs = json.loads((bundle / "bundle.json").read_text())["graphs"]
    if chained:
        plan = rewrite(plan, graphs, ["act"])
    images = [t for t in plan["wire"]["request_tensors"] if t["name"].startswith("image")]
    for tensor, decode in zip(images, DECODES, strict=True):
        tensor.update(dtype="uint8", shape=[tensor["elements"] // 3, 3], decode=decode)
    (bundle / "plan.json").write_text(json.dumps(plan))
    return bundle, {t["name"]: t for t in images}


def _run(binary: Path, bundle: Path, inputs: Path, out: Path) -> bytes:
    proc = subprocess.run([str(binary), str(bundle), str(inputs), str(out), "2"],
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    return out.read_bytes()


@pytest.mark.parametrize("chained", [False, True])
def test_native_uint8_decode_matches_host_normalization(tmp_path, run_plan, chained):
    bundle, images = _uint8_bundle(tmp_path / "b", chained)
    rng = np.random.default_rng(7)
    raw, host = tmp_path / "raw", tmp_path / "host"
    shutil.copytree(bundle / "fixtures" / "inputs", raw)
    shutil.copytree(bundle / "fixtures" / "inputs", host)
    for name, tensor in images.items():
        frame = rng.integers(0, 256, tensor["elements"], dtype=np.uint8)
        frame.tofile(raw / f"{name}.u8")
        (raw / f"{name}.f32").unlink()  # the raw run must decode, not fall back
        decode_tensor(frame.reshape(-1, 3), tensor["decode"]).astype("<f4").tofile(host / f"{name}.f32")
    decoded = _run(run_plan, bundle, raw, tmp_path / "raw.f32")
    reference = _run(run_plan, bundle, host, tmp_path / "host.f32")
    assert decoded == reference


def test_native_plan_rejects_a_bad_uint8_decode(tmp_path, run_plan):
    bundle, _ = _uint8_bundle(tmp_path / "b", chained=False)
    plan = json.loads((bundle / "plan.json").read_text())
    plan["wire"]["request_tensors"][0]["decode"]["std"] = [1.0, 0.0, 1.0]
    (bundle / "plan.json").write_text(json.dumps(plan))
    proc = subprocess.run([str(run_plan), str(bundle), str(bundle / "fixtures/inputs"),
                           str(tmp_path / "o.f32")], capture_output=True, text=True, check=False)
    assert proc.returncode == 1 and "zero std" in proc.stderr


# ------------------------------------------------------------- packing


def _stats(path: Path) -> Path:
    np.savez(path, state_mean=np.zeros(6), state_std=np.ones(6),
             action_mean=np.zeros(6), action_std=np.ones(6),
             camera0_mean=np.asarray(DECODES[0]["mean"]), camera0_std=np.asarray(DECODES[0]["std"]),
             camera1_mean=np.asarray(DECODES[1]["mean"]), camera1_std=np.asarray(DECODES[1]["std"]))
    return path


def test_act_plan_wire_carries_each_cameras_decode(tmp_path):
    from polima.bundle.pack import _attach_wire_decode, _plan_dict
    from polima.policies.act import ACT_SPEC

    plan = _plan_dict(ACT_SPEC.build_runtime_plan(), wire=ACT_SPEC)
    _attach_wire_decode(plan, ACT_SPEC, {"normalization_stats.npz": _stats(tmp_path / "s.npz")})
    tensors = {t["name"]: t for t in plan["wire"]["request_tensors"]}
    assert plan["wire"]["version"] == 2
    for index, name in enumerate(("image0", "image1")):
        assert tensors[name]["dtype"] == "uint8"
        np.testing.assert_array_equal(np.float32(tensors[name]["decode"]["mean"]),
                                      np.float32(DECODES[index]["mean"]))
    assert "dtype" not in tensors["state"] and "decode" not in tensors["state"]


def test_float_plans_are_unchanged_by_the_wire_dtype_field():
    """No dtype key for float32 tensors, so GR00T/SmolVLA bundle ids do not move."""
    from polima.bundle.pack import _plan_dict
    from polima.policies.registry import get_policy

    spec = get_policy("groot")
    plan = _plan_dict(spec.build_runtime_plan(), wire=spec)
    assert all("dtype" not in t for t in plan["wire"]["request_tensors"])


def test_act_fixture_frames_are_recovered_exactly(tmp_path):
    from types import SimpleNamespace

    from polima.bundle.importer import _uint8_frame_fixtures
    from polima.policies.act import ACT_SPEC

    root = tmp_path / "build"
    direct = root / "direct_inputs"
    direct.mkdir(parents=True)
    _stats(root / "normalization_stats.npz")
    rng = np.random.default_rng(9)
    frames = [rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(2)]
    for index, frame in enumerate(frames):
        decode_tensor(frame, DECODES[index]).astype("<f4").tofile(direct / f"vision_input_{index}.f32")
    fixtures = _uint8_frame_fixtures(SimpleNamespace(root=root), ACT_SPEC, direct)
    assert sorted(fixtures) == ["inputs/image0.u8", "inputs/image1.u8"]
    for index, name in enumerate(("image0", "image1")):
        np.testing.assert_array_equal(np.fromfile(fixtures[f"inputs/{name}.u8"], np.uint8),
                                      frames[index].reshape(-1))
