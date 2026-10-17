"""Adopt an existing build tree as a bundle, without recompiling.

This validates the deploy + runtime + wire + client stack against ELFs already
known to work on a DevKit, independently of the compile path. If the imported
bundle serves correct actions, any later failure belongs to the compiler, not
the runtime.

A build tree is messier than it looks. `retained/` can hold several variants
per graph, including abandoned experiments (`vision_backbone_rejected_rank4`,
`decoder_action_tail_v2`, ...), and `input_contract.json` also lists the
`_tensor_prepared` intermediates. Neither can be globbed; both are resolved
against the PolicySpec's declared graph names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path

from polima.bundle import retained as retained_module
from polima.bundle.layout import Bundle
from polima.bundle.pack import BundleInputs, build_bundle
from polima.bundle.retained import ElfCandidate
from polima.bundle.unpack import find_mpk, from_mpk
from polima.policies.base import PolicySpec
from polima.util.hashing import sha256_file
from polima.util.jsonio import read_json_or, write_json
from polima.util.logging import get

log = get("bundle.import")

#: Suffixes the exporter appends to intermediate ONNX graphs; never real graphs.
INTERMEDIATE_SUFFIXES = ("_tensor_prepared", "_simplified", "_prepared")

#: Directories that hold already-unpacked models, in preference order.
DEPLOYED_TREES = ("models_uncompressed", "models")


@dataclass
class BuildTree:
    root: Path
    format: str
    checkpoint: str = ""
    dataset_root: str = ""
    graphs: tuple[str, ...] = ()
    precisions: dict[str, str] = dataclass_field(default_factory=dict)
    verification: dict | None = None

    @property
    def dataset_name(self) -> str:
        return Path(self.dataset_root).name if self.dataset_root else "unknown"

    @property
    def steps(self) -> int:
        """Training step count, read off the checkpoint path (.../checkpoints/100000/...)."""
        match = re.search(r"/checkpoints/(\d+)/", self.checkpoint)
        return int(match.group(1)) if match else 0


def detect(build_dir: str | Path) -> BuildTree:
    """Identify a build tree and read what it declares about itself.

    Detection order matches the manifest formats a tree can carry:
      1. input_contract.json  format "act-modalix-v1"
      2. artifact_manifest.json format "act-modalix-build-v1" / "groot-..."
    """
    root = Path(build_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"no such build directory: {root}")

    contract = read_json_or(root / "input_contract.json") or {}
    manifest = read_json_or(root / "artifact_manifest.json") or {}
    report = read_json_or(root / "onnx_verification_report.json")

    fmt = contract.get("format") or manifest.get("format")
    if not fmt:
        # A tree without a manifest: if the compiled models are there, that is enough to import; the
        # things a manifest would have supplied (dataset, steps) come from
        # --dataset / --steps, and their absence is reported rather than guessed.
        if _has_models(root):
            fmt = "unlabelled"
        else:
            raise ValueError(
                f"{root} has no recognisable manifest "
                "(expected input_contract.json or artifact_manifest.json) "
                "and no models_uncompressed/, models/ or retained/ to import"
            )

    return BuildTree(
        root=root,
        format=fmt,
        checkpoint=contract.get("checkpoint") or manifest.get("checkpoint") or "",
        dataset_root=contract.get("dataset_root", ""),
        graphs=tuple(_declared_graphs(contract)),
        precisions=_compiled_precisions(manifest),
        verification=report,
    )


def _compiled_precisions(manifest: dict) -> dict[str, str]:
    """Actual successful precision per graph from PoLiMa's compile report.

    This cannot be inferred from the policy declaration: a graph may have used
    a fallback, and ``bundle.json`` must record the precision each ELF actually
    has.
    """
    precisions: dict[str, str] = {}
    for graph in manifest.get("graphs", []):
        if not isinstance(graph, dict):
            continue
        name = graph.get("name")
        precision = graph.get("precision")
        if isinstance(name, str) and isinstance(precision, str):
            from polima.compile.precision import effective_precision_label

            precisions[name] = effective_precision_label(
                precision,
                graph.get("activation_precision"),
                graph.get("weight_precision"),
            )
    return precisions


def _has_models(root: Path) -> bool:
    """Whether the tree holds compiled models under any known layout."""
    for tree in (*DEPLOYED_TREES, "retained"):
        directory = root / tree
        if directory.is_dir() and any(directory.iterdir()):
            return True
    return any(root.glob("**/*_mpk.tar.gz"))


def _declared_graphs(contract: dict) -> list[str]:
    """Graph stems from input_contract.json, minus the intermediates.

    Every graph appears twice: `vision_backbone.onnx` and `vision_backbone_tensor_prepared.onnx`.
    """
    names: list[str] = []
    for entry in contract.get("graphs", []):
        stem = Path(str(entry)).stem
        if any(stem.endswith(suffix) for suffix in INTERMEDIATE_SUFFIXES):
            continue
        if stem not in names:
            names.append(stem)
    return names


def resolve_elfs(build: BuildTree, spec: PolicySpec) -> dict[str, ElfCandidate]:
    """Find one ELF per graph the policy declares.

    Source order, most-trustworthy first:
      1. models_uncompressed/<graph>/share/  -- a deploy-ready tree
      2. models/<graph>/share/               -- an already-deployed tree
      3. retained/<graph>/                   -- the compiler's working dir
      4. **/<graph>_mpk.tar.gz               -- unpack on the fly

    Selection is always by the policy's declared graph name. Globbing `retained/`
    would pick up `*_rejected_*` and `*_v2` experiments.
    """
    declared = set(build.graphs)
    if declared and declared != set(spec.compile.names):
        extra = sorted(declared - set(spec.compile.names))
        absent = sorted(set(spec.compile.names) - declared)
        log.warning(
            "manifest graphs differ from policy %s: extra=%s missing=%s",
            spec.name,
            extra,
            absent,
        )

    resolved: dict[str, ElfCandidate] = {}
    for spec_graph in spec.compile.graphs:
        graph = spec_graph.name
        candidate = _resolve_one(build.root, graph)
        # Hand-built trees name directories for the build, not the graph --
        # `vision_llima_bf16` rather than `vision`. Aliases are tried in order
        # and only after the canonical name, so a tree using both is unambiguous.
        for alias in getattr(spec_graph, "aliases", ()):
            if candidate is not None:
                break
            candidate = _resolve_one(build.root, alias)
            if candidate is not None:
                log.info("  %-24s matched the name %r", graph, alias)
        if candidate is None:
            raise FileNotFoundError(
                f"no ELF for graph {graph!r} under {build.root}; looked in "
                f"{', '.join(DEPLOYED_TREES)}, retained/, and *_mpk.tar.gz"
                + (
                    f" (also tried {list(spec_graph.aliases)})"
                    if spec_graph.aliases
                    else ""
                )
            )
        resolved[graph] = candidate
        log.info("  %-24s %s (%s)", graph, candidate.sha256[:12], candidate.variant)
    return resolved


def _resolve_one(root: Path, graph: str) -> ElfCandidate | None:
    for tree in DEPLOYED_TREES:
        try:
            return retained_module.from_deployed_tree(root / tree, graph)
        except FileNotFoundError:
            pass

    try:
        return retained_module.select(root / "retained", graph)
    except (FileNotFoundError, ValueError) as exc:
        log.debug("retained/ lookup for %s: %s", graph, exc)

    archive = find_mpk(root, graph)
    if archive is not None:
        destination = root / ".polima_unpacked" / graph
        result = from_mpk(archive, destination)
        elf = next((p for p in result.elfs if p.name.startswith(graph)), result.elfs[0])
        from polima.util.hashing import sha256_file

        return ElfCandidate(
            graph=graph,
            path=elf,
            variant=f"mpk:{archive.name}",
            sha256=sha256_file(elf),
            size=elf.stat().st_size,
        )
    return None


def collect_fixtures(build: BuildTree, spec: PolicySpec) -> dict[str, Path]:
    """Gather the golden inputs and expected outputs.

    An ACT build ships per-stage goldens (`<graph>_input.f32` /
    `<graph>_output.f32`) as well as end-to-end ones, which is strictly better
    than an end-to-end-only check: a wrong ELF can be identified by name
    instead of just failing the final cosine.
    """
    fixtures: dict[str, Path] = {}
    direct = build.root / "direct_inputs"

    if direct.is_dir():
        for index, tensor in enumerate(spec.wire.request_tensors):
            source = direct / f"vision_input_{index}.f32"
            if source.is_file():
                fixtures[f"inputs/{tensor.name}.f32"] = source
        state = direct / "state.f32"
        if state.is_file():
            fixtures["inputs/state.f32"] = state

        expected = direct / "expected_normalized_actions.f32"
        if expected.is_file():
            fixtures["expected/normalized_actions.f32"] = expected

        # Per-graph goldens, for stage-by-stage verification.
        for graph in spec.compile.names:
            for kind in ("input", "output"):
                source = direct / f"{graph}_{kind}.f32"
                if source.is_file():
                    fixtures[f"stages/{graph}_{kind}.f32"] = source
        # Iterative policies emit one golden per invocation, for example
        # suffix_00_output.f32 through suffix_09_output.f32. Preserve them so
        # --dump-stages can bisect hardware drift against the reference ONNX chain.
        for graph in spec.compile.names:
            for source in sorted(direct.glob(f"{graph}_[0-9][0-9]_output.f32")):
                fixtures[f"stages/{source.name}"] = source
        for index in (0, 1):
            source = direct / f"vision_output_{index}.f32"
            if source.is_file():
                fixtures[f"stages/vision_output_{index}.f32"] = source

    # The end-to-end fixture lives in one portable NPZ; materialize its wire
    # tensors so `polima run --fixture` can send them.
    fixture_archive = build.root / spec.compile.fixture_file
    if fixture_archive.is_file():
        materialized = build.root / ".polima_fixtures"
        materialized.mkdir(parents=True, exist_ok=True)
        import numpy as np

        normalization: dict[str, np.ndarray] = {}
        raw_state = None
        stats_path = build.root / "normalization_stats.npz"
        if spec.name in ("smolvla", "groot") and stats_path.is_file():
            with np.load(stats_path, allow_pickle=False) as stats:
                normalization = {
                    name: np.asarray(stats[name], dtype="<f4").copy()
                    for name in ("state_mean", "state_std", "action_mean", "action_std")
                }

        with np.load(fixture_archive, allow_pickle=False) as archive:
            for tensor in spec.wire.request_tensors:
                if f"inputs/{tensor.name}.f32" in fixtures:
                    continue
                if tensor.name not in archive:
                    raise ValueError(f"{fixture_archive.name} has no wire tensor {tensor.name!r}")
                values = np.asarray(archive[tensor.name], dtype="<f4")
                # SmolVLA's ONNX vision graph is NCHW, while its client sends
                # prepared camera frames as HWC. Fixtures exercise the wire
                # contract, so mirror the client.
                if (
                    spec.name == "smolvla"
                    and tensor.name.startswith("image")
                    and values.ndim == 4
                    and values.shape[:2] == (1, 3)
                ):
                    values = values[0].transpose(1, 2, 0)
                # SmolVLA and GR00T normalize state inside the server plan; the
                # exporter records the normalized model input, so send the raw
                # wire value instead.
                if spec.name in ("smolvla", "groot") and tensor.name == "state":
                    if not normalization:
                        raise FileNotFoundError(
                            f"{stats_path} is required to materialize raw {spec.name} state"
                        )
                    if "raw_state" in archive:
                        values = np.asarray(archive["raw_state"], dtype="<f4").reshape(-1)
                    elif spec.runtime_context.get("state_encoding"):
                        raise ValueError(
                            f"{fixture_archive.name}: sincos fixture requires original raw_state; "
                            "encoded BF16 state cannot be affine-denormalized"
                        )
                    else:
                        values = values.reshape(-1)[: spec.dataset.state_dim]
                        values = values * normalization["state_std"] + normalization["state_mean"]
                    raw_state = values.copy()
                values = values.reshape(-1)
                if values.size != tensor.elements:
                    raise ValueError(
                        f"{fixture_archive.name}:{tensor.name} holds {values.size} "
                        f"floats; wire contract needs {tensor.elements}"
                    )
                destination = materialized / f"{tensor.name}.f32"
                values.tofile(destination)
                fixtures[f"inputs/{tensor.name}.f32"] = destination

            expected_names = (
                ("action", "normalized_actions", "normalized_action")
                if spec.name == "smolvla"
                else ("final_action", "normalized_actions", "normalized_action")
                if spec.name == "groot"
                else ("normalized_actions", "normalized_action")
            )
            expected_key = next((name for name in expected_names if name in archive), None)
            if expected_key and "expected/normalized_actions.f32" not in fixtures:
                expected = np.asarray(archive[expected_key], dtype="<f4").reshape(-1)
                if spec.name == "groot":
                    if not normalization or raw_state is None:
                        raise FileNotFoundError(
                            f"{stats_path} and raw state are required to decode GR00T actions"
                        )
                    horizon = spec.robot.actions_per_chunk
                    action_dim = spec.dataset.action_dim
                    expected = np.asarray(archive[expected_key], dtype="<f4")
                    model_width = spec.graph("action_tail").outputs[0].shape[-1]
                    expected = expected.reshape(-1, model_width)[:horizon, :action_dim]
                    expected = np.clip(expected, -1.0, 1.0)
                    expected = expected * normalization["action_std"] + normalization["action_mean"]
                    relative_dim = int(spec.runtime_context.get("relative_action_dim", 0))
                    if relative_dim:
                        expected[:, :relative_dim] += raw_state[:relative_dim]
                    expected = expected.reshape(-1)
                elif spec.name == "smolvla" and expected_key != "action":
                    # The SmolVLA plan denormalizes on the board; compare raw actions.
                    if not normalization:
                        raise FileNotFoundError(
                            f"{stats_path} is required to materialize raw SmolVLA actions"
                        )
                    expected = (
                        expected.reshape(-1, spec.dataset.action_dim) * normalization["action_std"]
                        + normalization["action_mean"]
                    ).reshape(-1)
                if expected.size != spec.wire.response_elements:
                    raise ValueError(
                        f"{fixture_archive.name}:{expected_key} holds {expected.size} "
                        f"floats; wire response needs {spec.wire.response_elements}"
                    )
                destination = materialized / "normalized_actions.f32"
                expected.tofile(destination)
                fixtures["expected/normalized_actions.f32"] = destination

    for name in ("normalization_stats.npz", spec.compile.fixture_file):
        source = build.root / name
        if source.is_file():
            fixtures[name] = source
    return fixtures


def collect_sidecars(build: BuildTree, spec: PolicySpec) -> dict[str, Path]:
    """Collect runtime constants from a hand-built policy tree.

    Constants are read by the names in ``plan.json``; means and standard
    deviations stored packed in one file are split under those names.
    """
    names = tuple(spec.build_runtime_plan().sidecars)
    if not names:
        return {}

    constants_dir = build.root / "constants"
    collected: dict[str, Path] = {}
    for name in names:
        for candidate in (constants_dir / name, constants_dir / f"{name}.f32"):
            if candidate.is_file():
                collected[name] = candidate
                break

    normalization_names = ("state_mean", "state_std", "action_mean", "action_std")
    widths = (
        spec.dataset.state_dim,
        spec.dataset.state_dim,
        spec.dataset.action_dim,
        spec.dataset.action_dim,
    )
    missing_normalization = [
        name for name in normalization_names if name in names and name not in collected
    ]
    packed = constants_dir / "normalization_stats.f32"
    if missing_normalization and packed.is_file():
        import numpy as np

        values = np.fromfile(packed, dtype="<f4")
        expected = sum(widths)
        if values.size != expected:
            raise ValueError(
                f"{packed} has {values.size} float32 values; expected {expected} "
                "(state mean/std, action mean/std)"
            )
        staged = build.root / ".polima_sidecars"
        staged.mkdir(parents=True, exist_ok=True)
        offset = 0
        for name, width in zip(normalization_names, widths):
            target = staged / name
            target.write_bytes(values[offset : offset + width].tobytes())
            collected[name] = target
            offset += width

    missing = [name for name in names if name not in collected]
    if missing:
        raise FileNotFoundError(f"missing runtime sidecar(s) {missing} under {constants_dir}")
    return collected


#: SO-101 client assets included in deployment bundles.
SO101_CLIENT = Path(__file__).resolve().parents[1] / "robot/assets/so101"
#: Client files older bundles carried, removed when such a bundle is upgraded.
RETIRED_CLIENT = ("scripts/act_som_client.py",)


def _is_so101(spec: PolicySpec) -> bool:
    return spec.robot.calibration_id == "so-arm101" and spec.dataset.state_dim == 6


def collect_robot_files(
    spec: PolicySpec, source: BuildTree | Path | None = None
) -> dict[str, Path]:
    """The DevKit-side robot launchers to pack with an SO-101 bundle.

    Launchers, the control client and the dataset rest pose; the arm's calibration belongs to the arm, so each
    DevKit makes its own (`polima robot calibrate`, or Studio).

    `source` is the build being packed (its dataset gives ACT its rest pose) or the
    tree that already holds ``robot_client/robot_rest_position.json`` (GR00T).
    """
    if not _is_so101(spec) or spec.name not in ("act", "groot"):
        return {}
    build = source if isinstance(source, BuildTree) else None
    asset_root = build.root if build is not None else source
    files = {name: SO101_CLIENT / name for name in ("start.sh", "preview_robot_cameras.py", "run_robot_client_with_live_view.py")}
    rest = asset_root / "robot_client/robot_rest_position.json" if asset_root else None
    if rest is not None and rest.is_file():
        files["robot_rest_position.json"] = rest
    elif spec.name == "act" and build and build.dataset_root \
            and (Path(build.dataset_root) / "meta/info.json").is_file():
        from polima.export.robot import write_rest_pose

        files["robot_rest_position.json"] = write_rest_pose(
            Path(build.dataset_root), build.root / "robot_rest_position.json",
            spec.robot.joint_names, spec.robot.calibration_id)
    elif spec.name == "groot":
        raise FileNotFoundError("an SO-101 GR00T bundle needs robot_client/robot_rest_position.json "
                                "from its training dataset")
    else:
        log.warning("no training dataset recorded; the bundle needs a rest pose "
                    "(robot_client/robot_rest_position.json) before it can drive the arm")
    return files


def hydrate_robot_client(bundle: Bundle, spec: PolicySpec) -> bool:
    """Bring an already-packed bundle's robot client up to date.

    Bundle ids describe model arithmetic, not launchers, so a newer PoLiMa can
    replace the client without recompiling identical ELFs or changing the id.
    The bundle's own rest pose and calibration are kept. Returns whether the
    manifest changed.
    """
    import shutil

    recorded = set(bundle.sidecars)
    changed = False
    for relative, source in collect_robot_files(spec, bundle.root).items():
        destination = bundle.root / "robot_client" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file() or sha256_file(destination) != sha256_file(source):
            shutil.copy2(source, destination)
            changed = True
        recorded.add(str(destination.relative_to(bundle.root)))
    if _is_so101(spec):
        # The control loop is PoLiMa's own: retire client files older bundles carried.
        for relative in (*(f"robot_client/{name}" for name in RETIRED_CLIENT),
                         "normalization_stats.npz"):
            if (bundle.root / relative).is_file():
                (bundle.root / relative).unlink()
                changed = True
            recorded.discard(relative)
    elif spec.name == "act":
        fixture = bundle.root / "fixtures/normalization_stats.npz"
        destination = bundle.root / "normalization_stats.npz"
        if fixture.is_file():
            if not destination.is_file() or sha256_file(destination) != sha256_file(fixture):
                shutil.copy2(fixture, destination)
                changed = True
            recorded.add("normalization_stats.npz")

    sidecars = sorted(recorded)
    if sidecars != bundle.sidecars:
        bundle.sidecars = sidecars
        changed = True
    if changed:
        write_json(bundle.manifest_path, bundle.to_dict())
    return changed


def hydrate_runtime_metadata(bundle: Bundle, spec: PolicySpec) -> bool:
    """Upgrade graph layout metadata without touching content-addressed ELFs.

    Where a bundle does not record the compiler's physical output contract,
    PoLiMa copies the current policy declaration into ``bundle.json`` so its
    generic runner decodes each graph exactly as the compiled ELF emits it.
    """
    changed = False
    declared = {graph.name: graph for graph in spec.compile.graphs}
    for artifact in bundle.graphs:
        graph = declared.get(artifact.name)
        if graph is None:
            continue
        output = graph.outputs[0]
        values = {
            "dram_layout": output.dram_layout,
            "logical_width": output.logical_width,
            "logical_channels": output.logical_channels,
            "external_dram_layout": graph.external_dram_layout,
        }
        if artifact.persistent_input_elements:
            # A persistent-binding ELF (run_dit_chain) was compiled with its own
            # external layout; the generic policy declaration does not describe it.
            del values["external_dram_layout"]
        for field, value in values.items():
            if getattr(artifact, field) != value:
                setattr(artifact, field, value)
                changed = True
    if bundle.smoke != spec.smoke.to_dict():
        bundle.smoke = spec.smoke.to_dict()
        changed = True
    if changed:
        write_json(bundle.manifest_path, bundle.to_dict())
    return changed


def import_build(
    build_dir: str | Path,
    spec: PolicySpec,
    *,
    output_root: str | Path,
    dataset: str | None = None,
    steps: int | None = None,
    source: str = "import",
) -> Bundle:
    """Turn a build tree into a PoLiMa bundle.

    `source` records provenance in bundle.json: `import` (the default)
    for an adopted tree, `polima compile` for a tree PoLiMa built itself. The
    packing is identical either way, because the tree layouts are the same.
    """
    build = detect(build_dir)
    log.info("importing %s build from %s", build.format, build.root)

    elfs = resolve_elfs(build, spec)
    fixtures = collect_fixtures(build, spec)
    constants = collect_sidecars(build, spec)
    robot_files = collect_robot_files(spec, build)

    tool_versions = {"source_format": build.format}
    if build.verification:
        tool_versions["onnx_max_abs"] = build.verification.get("max_abs")

    bundle = build_bundle(
        BundleInputs(
            spec=spec,
            elfs=elfs,
            dataset=dataset or build.dataset_name,
            steps=steps if steps is not None else build.steps,
            checkpoint=build.checkpoint,
            fixtures=fixtures,
            constants=constants,
            robot_files=robot_files,
            source=source,
            source_dir=str(build.root),
            tool_versions=tool_versions,
            precisions=build.precisions,
        ),
        output_root,
    )
    return bundle
