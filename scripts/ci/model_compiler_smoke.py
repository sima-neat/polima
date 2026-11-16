#!/usr/bin/env python3
"""Compile small policy-shaped ONNX graphs with one resolved compiler."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

PROFILES = ("act", "groot", "pi05", "llima")


def parse_resolution(text: str) -> dict:
    """Read the JSON object printed after any ``sima-cli`` banner."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and all(
            str(payload.get(key, "")).strip() for key in ("ref", "resolved_spec", "environment")
        ):
            return payload
    raise ValueError("sima-cli printed no complete JSON resolution")


def graph(profile: str, path: Path) -> None:
    rng = np.random.default_rng(0)
    if profile == "act":
        x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 16, 16])
        y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 14, 14])
        weight = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)
        nodes = [helper.make_node("Conv", ["input", "weight"], ["conv"]),
                 helper.make_node("Relu", ["conv"], ["output"])]
        initializers = [numpy_helper.from_array(weight, "weight")]
    elif profile in {"groot", "pi05"}:
        hidden = 16 if profile == "groot" else 32
        x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 16])
        y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, hidden])
        weight = rng.standard_normal((16, hidden)).astype(np.float32)
        activation = "Tanh" if profile == "groot" else "Relu"
        nodes = [helper.make_node("MatMul", ["input", "weight"], ["dense"]),
                 helper.make_node(activation, ["dense"], ["output"])]
        initializers = [numpy_helper.from_array(weight, "weight")]
    else:
        x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 32])
        y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 32])
        nodes = [helper.make_node("Softmax", ["input"], ["output"], axis=-1)]
        initializers = []
    model = helper.make_model(helper.make_graph(nodes, profile, [x], [y], initializers),
                              opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 8)
    onnx.save(model, path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    resolution = parse_resolution(args.resolution.read_text())
    evidence = {
        "compiler": resolution,
        "python": sys.version,
        "machine": platform.machine(),
        "source_commit": os.environ.get("GITHUB_SHA", "local"),
        "profiles": {},
    }
    for profile in PROFILES:
        folder = args.output / profile
        folder.mkdir()
        model = folder / f"{profile}.onnx"
        graph(profile, model)
        command = [
            sys.executable, "-m", "polima.compile.tensor", "--model-path", str(model),
            "--build-dir", str(folder / "build"), "--precision", "bf16", "--no-simplify",
            "--retain-compile-dir", str(folder / "retained"), "--exit-on-stable-elf",
        ]
        subprocess.run(command, check=True)
        elfs = sorted(folder.rglob("*.elf"))
        if not elfs:
            raise RuntimeError(f"{profile}: compiler produced no ELF")
        evidence["profiles"][profile] = {
            "model_sha256": digest(model),
            "elfs": {str(path.relative_to(args.output)): digest(path) for path in elfs},
        }
    (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
