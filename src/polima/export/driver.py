"""Checkpoint -> ONNX graphs + calibration data + fixtures.

The generic half of the export stage. Everything policy-specific -- how the
network is cut into graphs, what shapes those graphs take -- is behind the four
entry points named in `CompilePlan`:

    export_entry          write onnx/ and calibration/
    fixture_entry         write the reference tensors the smoke test compares to
    verify_entry          replay the ONNX chain and compare against PyTorch
    normalization_entry   pull mean/std out of the checkpoint

They are dotted strings, resolved here, so `polima.policies.act` stays importable
in the compiler venv and on the board where torch does not exist. Only this
module and the modules it resolves need torch.

## Verify before compile

The chain is checked under onnxruntime *before* anything is quantized. That
split is the whole point: if the ONNX matches PyTorch and the compiled bundle
does not, the fault is in quantization or on the board, and vice versa. Debugging
a single number at the end of a six-graph pipeline without that split is
guesswork.
"""

from __future__ import annotations

import importlib
import inspect
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from polima.util.hashing import sha256_file

CONTRACT_FILE = "input_contract.json"
VERIFY_REPORT = "onnx_verification_report.json"


@dataclass
class ExportResult:
    build_dir: str
    graphs: list[str] = field(default_factory=list)
    checkpoint: str = ""
    dataset_root: str = ""
    calibration_samples: int = 0
    verification: dict | None = None
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return bool(self.graphs) and (self.verification or {}).get("ok", False)


def resolve(entry: str):
    """`package.module:attribute` -> the attribute."""
    module_name, _, attribute = entry.partition(":")
    if not attribute:
        raise ValueError(f"entry point {entry!r} must be 'module:attribute'")
    return getattr(importlib.import_module(module_name), attribute)


def export(spec, checkpoint: str | Path, build_dir: str | Path, *,
           dataset_root: str | Path | None = None, calibration_samples: int = 8,
           lerobot_src: str | Path | None = None, verify: bool = True,
           seed: int = 123) -> ExportResult:
    """Run the full export for one policy.

    Seeded because the calibration path touches dataset sampling and torch: an
    unseeded export produces different calibration data on every run, which would
    make the ELFs unreproducible and the resume key meaningless.
    """
    import random

    import numpy as np
    import torch

    from polima.export import samples as sampling

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    checkpoint = Path(checkpoint).resolve()
    build_dir = Path(build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    graphs_module = importlib.import_module(spec.compile.export_entry.split(":")[0])
    profile = dict(spec.runtime_context)
    load_policy_kwargs = {"lerobot_src": lerobot_src}
    if "profile" in inspect.signature(graphs_module.load_policy).parameters:
        load_policy_kwargs["profile"] = profile
    policy, image_keys = graphs_module.load_policy(checkpoint, **load_policy_kwargs)
    modules = graphs_module.build_modules(policy)

    observation_keys = [
        spec.dataset.state_key,
        *image_keys,
        *getattr(graphs_module, "EXTRA_OBSERVATION_KEYS", ()),
    ]
    # A policy whose checkpoint is not a LeRobot policy supplies its own draw.
    # GR00T's is a transformers AutoModel over a LeRobot v2.1 dataset, so the
    # shared path's `make_pre_post_processors` has nothing to build from.
    draw = getattr(graphs_module, "load_samples", sampling.load)
    draw_kwargs = {"count": calibration_samples, "lerobot_src": lerobot_src}
    if "profile" in inspect.signature(draw).parameters:
        draw_kwargs["profile"] = profile
    drawn, postprocessor, resolved_root = draw(
        policy, checkpoint, observation_keys, dataset_root, **draw_kwargs
    )
    trace_samples = drawn[:1] if getattr(graphs_module, "TRACE_FIRST_ONLY", False) else drawn
    traces = [graphs_module.trace(modules, sample, image_keys) for sample in trace_samples]

    # Some prompt geometry is only knowable after the checkpoint processor has
    # built a real sample. GR00T uses this hook for the sequence length and one
    # image-token window per camera; the specialized contract is then restored
    # before compilation/packing.
    specialize_trace = getattr(graphs_module, "specialize_spec_from_trace", None)
    if specialize_trace is not None:
        spec = specialize_trace(spec, traces[0], image_keys)

    written = resolve(spec.compile.export_entry)(
        build_dir, modules, drawn, traces, image_keys
    )
    if spec.compile.fixture_entry:
        resolve(spec.compile.fixture_entry)(
            build_dir, drawn[0], traces[0], image_keys, postprocessor
        )
    if spec.compile.normalization_entry:
        normalization = resolve(spec.compile.normalization_entry)
        normalization_kwargs = {}
        if "profile" in inspect.signature(normalization).parameters:
            normalization_kwargs["profile"] = profile
        normalization(
            checkpoint, image_keys, build_dir / "normalization_stats.npz",
            **normalization_kwargs,
        )

    if spec.name in ("groot", "smolvla") and spec.robot.calibration_id == "so-arm101":
        from polima.export.robot import write_so101_rest_pose

        write_so101_rest_pose(
            resolved_root, build_dir / "robot_client/robot_rest_position.json",
            spec.robot.joint_names,
        )

    _write_contract(spec, build_dir, checkpoint, resolved_root, image_keys, written)

    verification = None
    if verify and spec.compile.verify_entry:
        verification = resolve(spec.compile.verify_entry)(
            build_dir / "onnx", build_dir / spec.compile.fixture_file,
            build_dir / VERIFY_REPORT,
            atol=spec.compile.verify_atol, rtol=spec.compile.verify_rtol,
            # Per-graph references land beside the raw inputs, which is where
            # bundle packing looks for them.
            stage_dir=build_dir / "direct_inputs",
        )

    result = ExportResult(
        build_dir=str(build_dir),
        graphs=[path.stem for path in written],
        checkpoint=str(checkpoint),
        dataset_root=str(resolved_root),
        calibration_samples=len(drawn),
        verification=verification,
        duration_s=time.monotonic() - started,
    )
    _record(build_dir, result)
    return result


def _write_contract(spec, build_dir: Path, checkpoint: Path, dataset_root: Path,
                    image_keys, written) -> Path:
    """The build tree's self-description, read later by bundle packing.

    Camera order is the load-bearing field: the board addresses cameras by slot,
    and swapping them produces a policy that runs perfectly and reaches for the
    wrong place.
    """
    contract = {
        "format": "polima-export-v1",
        "policy": spec.name,
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "camera_order": list(image_keys),
        "profile": {
            "state_names": list(spec.dataset.state_names),
            "state_dim": spec.dataset.state_dim,
            "action_dim": spec.dataset.action_dim,
            "camera_keys": list(spec.dataset.camera_keys),
            "fps": spec.dataset.fps,
            "calibration_id": spec.robot.calibration_id,
            "actions_per_chunk": spec.robot.actions_per_chunk,
            "runtime_context": dict(spec.runtime_context),
        },
        "normalization": "checkpoint-owned affine center/scale",
        "batch": 1,
        "fixed_shapes": {
            tensor.name: list(tensor.shape)
            for graph in spec.compile.graphs
            for tensor in (*graph.inputs, *graph.outputs)
        },
        "graphs": [f"{path.stem}.onnx" for path in written],
    }
    path = build_dir / CONTRACT_FILE
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return path


def _record(build_dir: Path, result: ExportResult) -> None:
    path = build_dir / "artifact_manifest.json"
    manifest = {}
    if path.exists():
        try:
            manifest = json.loads(path.read_text())
        except json.JSONDecodeError:
            manifest = {}
    manifest["export"] = asdict(result)
    manifest["onnx_sha256"] = {
        onnx.stem: sha256_file(onnx)
        for onnx in sorted((build_dir / "onnx").glob("*.onnx"))
        if not onnx.stem.endswith("_tensor_prepared")
    }
    manifest["calibration_sha256"] = {
        calibration.stem: sha256_file(calibration)
        for calibration in sorted((build_dir / "calibration").glob("*.npz"))
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
