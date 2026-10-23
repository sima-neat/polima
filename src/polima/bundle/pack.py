"""Assemble a deployable bundle.

One convention, always: <bundle>/models/<graph>/{etc,lib,share}/. ELFs from an
exploded *_mpk.tar.gz and from a models_uncompressed/ tree both land here, so
nothing downstream ever branches on where an ELF came from.

Everything the DevKit needs is inside the bundle and nothing else is: no ONNX, no
checkpoints, no calibration .npz, and above all no .tar.gz -- the deploy step
refuses to ship archives, because the direct-MLA runtime never opens one.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from polima.bundle.layout import Bundle, GraphArtifact, compute_bundle_id
from polima.bundle.retained import ElfCandidate
from polima.compile.precision import effective_precision_label
from polima.policies.base import PolicySpec, RuntimePlan
from polima.util.hashing import sha256_file
from polima.util.jsonio import write_json
from polima.util.logging import get

log = get("bundle.pack")


@dataclass
class BundleInputs:
    """Everything needed to materialize a bundle."""

    spec: PolicySpec
    elfs: dict[str, ElfCandidate]
    dataset: str
    steps: int
    checkpoint: str = ""
    #: destination-relative path -> source file
    fixtures: dict[str, Path] = field(default_factory=dict)
    constants: dict[str, Path] = field(default_factory=dict)
    #: DevKit-side LeRobot launcher and policy transport, relative to robot_client/.
    robot_files: dict[str, Path] = field(default_factory=dict)
    #: graph -> extra files to place beside the ELF (etc/*.json, lib/*.so)
    model_extras: dict[str, list[Path]] = field(default_factory=dict)
    source: str = "compile"
    source_dir: str | None = None
    tool_versions: dict = field(default_factory=dict)
    precisions: dict[str, str] = field(default_factory=dict)
    model_version: str | None = None
    task_label: str | None = None
    #: Rewrites the spec's plan (as plan.json data) before the bundle id is taken: the
    #: GR00T run_bound_chain DiT. Spec graphs the rewritten plan no longer runs may then
    #: lack an ELF.
    plan_rewrite: Callable[[dict], dict] | None = None
    #: name -> GraphArtifact fields (precision, input/output_elements, persistent_bindings,
    #: int8_inputs ...): overrides for spec graphs, whole entries for graphs outside the
    #: spec (their ELFs in `elfs`).
    extra_graphs: dict[str, dict] = field(default_factory=dict)
    #: Bundle-level precision label (default: derived from the per-ELF labels).
    precision_label: str | None = None


def bundle_precision_label(labels: set[str]) -> str:
    """Bundle-level label from per-ELF labels.

    Any INT8-activation ELF makes the bundle "w8a8" (its BF16 helper graphs do not make
    it "mixed"); otherwise INT8 weights make it "mixed"; a uniform build keeps its label.
    """
    def int8_activations(label: str) -> bool:
        return label in ("a8w8", "int8") or label.upper().startswith("A_INT8")

    if len(labels) == 1:  # uniform builds keep their established labels
        label = next(iter(labels))
        return {"a8w8": "w8a8", "w8a16": "mixed"}.get(label, label)
    return "w8a8" if any(int8_activations(label) for label in labels) else "mixed"


def _artifact_precision(graph, recorded: dict[str, str]) -> str:
    """Return the most specific truthful precision label for one ELF."""
    if graph.activation_precision or graph.weight_precision:
        return effective_precision_label(
            graph.precision, graph.activation_precision, graph.weight_precision
        )
    return recorded.get(graph.name, graph.precision)


def build_bundle(inputs: BundleInputs, output_root: str | Path) -> Bundle:
    """Write the bundle tree and return its manifest.

    The bundle id is computed from ELF content plus the canonical plan, so an
    identical rebuild produces an identical id and `polima deploy` becomes a
    no-op.
    """
    spec = inputs.spec
    missing = [name for name in spec.compile.names if name not in inputs.elfs]
    if missing and not inputs.plan_rewrite:
        raise ValueError(f"missing ELF(s) for {missing}; have {sorted(inputs.elfs)}")

    plan = spec.build_runtime_plan()
    plan_data = _plan_dict(plan, wire=spec)
    _attach_wire_decode(plan_data, spec, inputs.fixtures)
    if inputs.plan_rewrite:
        plan_data = inputs.plan_rewrite(plan_data)
        _check_plan_data_against_elfs(plan_data, inputs.elfs)
    else:
        _check_plan_against_elfs(spec, plan, inputs.elfs)
    task_catalog = list(dict(spec.runtime_context).get("task_catalog") or ())
    tasks = [str(task["instruction"]) for task in task_catalog]

    robot_label = {"so-arm101": "so101", "gr1": "gr1"}.get(
        spec.robot.calibration_id, spec.robot.calibration_id or None)
    labels = {_artifact_precision(g, inputs.precisions) for g in spec.compile.graphs if g.name in inputs.elfs}
    precision_label = inputs.precision_label or bundle_precision_label(labels)
    bundle_id = compute_bundle_id(
        policy=spec.name,
        dataset=inputs.task_label or inputs.dataset,
        robot=robot_label,
        version=inputs.model_version,
        precision=precision_label,
        steps=inputs.steps,
        graph_digests={name: candidate.sha256 for name, candidate in inputs.elfs.items()},
        plan=plan_data,
    )

    root = Path(output_root).resolve() / bundle_id
    if root.exists():
        log.info("bundle %s already materialized at %s", bundle_id, root)
    root.mkdir(parents=True, exist_ok=True)

    artifacts: list[GraphArtifact] = []
    for graph in spec.compile.graphs:
        if graph.name not in inputs.elfs:
            continue  # replaced by the plan rewrite
        candidate = inputs.elfs[graph.name]
        model_dir = root / "models" / graph.name
        for sub in ("etc", "lib", "share"):
            (model_dir / sub).mkdir(parents=True, exist_ok=True)

        destination = model_dir / "share" / f"{graph.name}_stage1_mla.elf"
        _copy(candidate.path, destination)

        for extra in inputs.model_extras.get(graph.name, []):
            sub = {".json": "etc", ".yaml": "etc", ".yml": "etc", ".so": "lib"}.get(
                extra.suffix.lower()
            )
            if sub:
                _copy(extra, model_dir / sub / extra.name)

        artifacts.append(
            GraphArtifact(
                name=graph.name,
                elf=str(destination.relative_to(root)),
                sha256=candidate.sha256,
                elf_bytes=destination.stat().st_size,
                # Explicit split precision is part of the graph contract and
                # is more specific than a build manifest's output directory
                # label, which can say bf16 for a graph compiled A=BF16/W=INT8.
                precision=_artifact_precision(graph, inputs.precisions),
                input_elements=sum(t.elements for t in graph.inputs),
                output_elements=sum(t.elements for t in graph.outputs),
                dram_layout=graph.outputs[0].dram_layout,
                logical_width=graph.outputs[0].logical_width,
                logical_channels=graph.outputs[0].logical_channels,
                external_dram_layout=graph.external_dram_layout,
            )
        )
        for key, value in inputs.extra_graphs.get(graph.name, {}).items():
            setattr(artifacts[-1], key, value)

    for name, fields in inputs.extra_graphs.items():
        if name in spec.compile.names:
            continue
        destination = root / "models" / name / "share" / f"{name}_stage1_mla.elf"
        _copy(inputs.elfs[name].path, destination)
        artifacts.append(GraphArtifact(name=name, elf=str(destination.relative_to(root)),
                                       sha256=inputs.elfs[name].sha256,
                                       elf_bytes=destination.stat().st_size, **fields))

    for relative, source in sorted(inputs.fixtures.items()):
        _copy(source, root / "fixtures" / relative)
    for relative, source in sorted(inputs.constants.items()):
        _copy(source, root / "constants" / relative)
    for relative, source in sorted(inputs.robot_files.items()):
        _copy(source, root / "robot_client" / relative)

    # Sidecars are recorded relative to the bundle so the C++ side can resolve
    # them without knowing anything about the host layout.
    sidecars = sorted([
        *(str((root / "constants" / relative).relative_to(root))
          for relative in inputs.constants),
        *(str((root / "robot_client" / relative).relative_to(root))
          for relative in inputs.robot_files),
    ])

    bundle = Bundle(
        root=root,
        policy=spec.name,
        bundle_id=bundle_id,
        # The checkpoint by its last two path parts (e.g. output/checkpoint-50000), never a host path.
        checkpoint="/".join(Path(inputs.checkpoint).parts[-2:]) if inputs.checkpoint else "",
        graphs=artifacts,
        sidecars=sidecars,
        source=inputs.source,
        source_dir=inputs.source_dir,
        tool_versions=inputs.tool_versions,
        smoke=spec.smoke.to_dict(),
        tasks=tasks,
    )

    write_json(bundle.plan_path, plan_data)
    write_json(bundle.manifest_path, bundle.to_dict())
    _assert_no_archives(root)

    log.info(
        "bundle %s: %d graphs, %.1f MiB of ELF, %d fixture(s)",
        bundle_id,
        len(artifacts),
        bundle.total_elf_bytes / 1048576,
        len(inputs.fixtures),
    )
    return bundle


def _attach_wire_decode(plan_data: dict, spec: PolicySpec, fixtures: dict[str, Path]) -> None:
    """Give each uint8 camera tensor the server-side normalization of its camera.

    The server decodes ``(x / scale - mean[c]) / std[c]`` in float32
    (`polima.wire.protocol.decode_tensor`). ``std`` here is the float32 sum
    ``std + eps`` LeRobot's NormalizerProcessor divides by, so the decoded frame
    is bit-identical to the checkpoint's own preprocessing (and to the export's
    PyTorch reference fixtures), not merely within a float32 ulp of it.
    """
    frames = [item for item in plan_data.get("wire", {}).get("request_tensors", [])
              if item.get("dtype") == "uint8"]
    if not frames:
        return
    stats_path = fixtures.get(spec.wire.stats_file)
    if stats_path is None or not Path(stats_path).is_file():
        raise ValueError(f"uint8 camera tensors need {spec.wire.stats_file} to decode them")
    import numpy as np

    images = [t.name for t in spec.wire.request_tensors if t.name.startswith("image")]
    with np.load(stats_path, allow_pickle=False) as stats:
        eps = np.float32(stats["eps"]) if "eps" in stats.files else np.float32(1e-8)
        for item in frames:
            index = images.index(item["name"])
            mean = np.asarray(stats[f"camera{index}_mean"], dtype=np.float32).reshape(-1)
            std = np.asarray(stats[f"camera{index}_std"], dtype=np.float32).reshape(-1) + eps
            if mean.size != item["shape"][-1] or std.size != mean.size:
                raise ValueError(f"{item['name']}: camera{index} stats do not match "
                                 f"{item['shape'][-1]} channels")
            item["decode"] = {"scale": 255.0, "mean": [float(v) for v in mean],
                              "std": [float(v) for v in std]}


def _plan_dict(plan: RuntimePlan, wire: PolicySpec | None = None) -> dict:
    """The JSON the native interpreter reads.

    Kept flat and explicit -- plan.cpp parses this with nlohmann/json and must
    never have to infer anything.
    """
    data = {
        "buffers": dict(plan.buffers),
        "steps": [{"op": s.op, "out": s.out, "args": dict(s.args)} for s in plan.steps],
        "result": plan.result,
        "sidecars": list(plan.sidecars),
        "loops": dict(plan.loops),
    }
    if wire is not None:
        # Host-visible graph output layout is executable behavior, not merely
        # descriptive metadata. Include it in plan.json and therefore in the
        # content-addressed bundle id so a layout correction cannot masquerade
        # as the same bundle.
        data["graph_outputs"] = {
            graph.name: [
                {
                    "name": output.name,
                    "shape": list(output.shape),
                    "dtype": output.dtype,
                    "dram_layout": output.dram_layout,
                    "logical_width": output.logical_width,
                    "logical_channels": output.logical_channels,
                }
                for output in graph.outputs
            ]
            for graph in wire.compile.graphs
        }
        task_catalog = list(dict(wire.runtime_context).get("task_catalog") or ())
        if task_catalog:
            data["task_catalog"] = task_catalog
            data["tasks"] = [str(task["instruction"]) for task in task_catalog]
            data["default_task"] = 0
        data["wire"] = {
            "magic": wire.wire.magic,
            "version": wire.wire.version,
            "default_port": wire.wire.default_port,
            "request_tensors": [
                # dtype only when it is not the float32 default, so float plans
                # (and therefore their bundle ids) are unchanged.
                {"name": t.name, "elements": t.elements, "shape": list(t.shape),
                 **({"dtype": t.dtype} if t.dtype != "float32" else {})}
                for t in wire.wire.request_tensors
            ],
            "response_shape": list(wire.wire.response_shape),
            "response_elements": wire.wire.response_elements,
        }
        # The robot description crosses to the DevKit too. Without it the
        # DevKit would need the Python PolicySpec just to learn which camera is
        # the wrist one -- and the DevKit deliberately has no polima Python.
        robot = wire.robot
        data["robot"] = {
            "camera_roles": [list(pair) for pair in robot.camera_roles],
            "camera_hints": dict(robot.camera_hints),
            "camera_fourcc": robot.camera_fourcc,
            "camera_capture_fps": dict(robot.camera_capture_fps),
            "joint_names": list(robot.joint_names),
            "actions_per_chunk": robot.actions_per_chunk,
            "fps": robot.default_fps,
            "max_relative_target": robot.max_relative_target,
            "aggregate_fn": robot.aggregate_fn,
            "calibration_id": robot.calibration_id,
        }
        data["profile"] = {
            "state_names": list(wire.dataset.state_names),
            "state_dim": wire.dataset.state_dim,
            "action_dim": wire.dataset.action_dim,
            "camera_keys": list(wire.dataset.camera_keys),
            "fps": wire.dataset.fps,
            "calibration_id": robot.calibration_id,
            "actions_per_chunk": robot.actions_per_chunk,
            "runtime_context": dict(wire.runtime_context),
        }
        if wire.runtime_context:
            # Lightweight clients can consume graph geometry without knowing
            # anything about Python-side profile selection.
            data["runtime_context"] = dict(wire.runtime_context)

    return data


def _check_plan_against_elfs(
    spec: PolicySpec, plan: RuntimePlan, elfs: dict[str, ElfCandidate]
) -> None:
    """Every accelerator step must name graphs for which ELFs were supplied."""
    for index, step in enumerate(plan.steps):
        if step.op not in ("run_elf", "run_elf_chain"):
            continue
        graphs = (
            (step.args.get("graph"),)
            if step.op == "run_elf"
            else tuple(step.args.get("graphs", ()))
        )
        for graph in graphs:
            if graph not in elfs:
                raise ValueError(f"plan step {index} runs graph {graph!r} but no ELF was supplied")
            _ = spec.compile.graph(graph)


def _check_plan_data_against_elfs(plan: dict, elfs: dict[str, ElfCandidate]) -> None:
    """Every graph a rewritten plan runs must have an ELF."""
    for index, step in enumerate(plan["steps"]):
        args = step.get("args", {})
        graphs = [args.get("graph")] if step["op"] == "run_elf" else \
            [*args.get("prelude", ()), *args.get("graphs", ())]
        for graph in graphs:
            if graph and graph not in elfs:
                raise ValueError(f"plan step {index} runs graph {graph!r} but no ELF was supplied")


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        destination.exists()
        and destination.stat().st_size == source.stat().st_size
        and sha256_file(destination) == sha256_file(source)
    ):
        return
    shutil.copy2(source, destination)


def _assert_no_archives(root: Path) -> None:
    """A bundle must never contain a .tar.gz.

    The DevKit runtime never opens an archive. Checking at pack time lets the
    message point at the offending file.
    """
    archives = [p for p in root.rglob("*.tar.gz")]
    if archives:
        raise RuntimeError(
            "bundle contains archives, which the direct-MLA runtime cannot read: "
            + ", ".join(str(p.relative_to(root)) for p in archives)
        )
