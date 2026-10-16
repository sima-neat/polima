"""The policy plugin contract -- PoLiMa's central abstraction.

Adding a policy should mean writing one PolicySpec, not copying ~3,000 lines.
A spec declares, declaratively and in one place:

  * DatasetContract  -- what the on-disk LeRobotDataset must look like
  * CompilePlan      -- the ONNX subgraph decomposition and how each graph compiles
  * RuntimePlan      -- what runs on the MLA, in what order, with what host glue
  * WireSpec         -- the TCP protocol between the robot client and the server
  * RobotSpec        -- cameras, joints, chunk size for the robot client

Everything here is a frozen dataclass and JSON round-trippable, because a spec
has to cross three interpreters: the training conda env (torch), the
model-compiler venv (afe, no torch), and the board (py3.11, no torch).
Consequently `builder`/`entrypoint`-style fields hold *dotted import paths as
strings*, not callables -- they are resolved lazily, in the interpreter that can
actually import them.

STDLIB ONLY. No numpy, no torch. See the dependency-floor note in pyproject.toml.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from math import prod
from typing import Any, Literal

Layout = Literal["NCHW", "NHWC"]
Precision = Literal["int8", "bf16", "mixed"]
DramLayout = Literal["plain", "hwc16"]
ExternalDramLayout = Literal["compiler", "HWC", "HWC16"]
Compiler = Literal["afe", "llima"]
Device = Literal["modalix", "mlsoc"]


class SpecError(ValueError):
    """A PolicySpec is internally inconsistent."""


def resolve(dotted: str) -> Any:
    """Import 'pkg.module:attr' lazily.

    Specs name their torch modules and entry points this way so that importing
    polima.policies.act does not drag torch into the compiler venv or the board.
    """
    if ":" not in dotted:
        raise SpecError(f"expected 'module:attr', got {dotted!r}")
    module_name, attr = dotted.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise SpecError(f"{module_name} has no attribute {attr!r}") from exc


# --------------------------------------------------------------------- tensors


@dataclass(frozen=True)
class TensorSpec:
    """A fully static tensor at a graph or wire boundary.

    `dram_layout` describes how the MLA lays this tensor out in DRAM when it
    crosses the accelerator boundary. 'hwc16' means the 16-lane byte-planed
    tessellated form that TensorDRAMLayout.HWC16 produces; the runtime needs
    logical_width/logical_channels to reverse it. Bare ModelSDK ELFs can expose
    plain output even where an MPK-based application applies this transform.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str = "float32"
    dram_layout: DramLayout = "plain"
    logical_width: int | None = None
    logical_channels: int | None = None

    @property
    def elements(self) -> int:
        return prod(self.shape) if self.shape else 0

    def validate(self, where: str) -> None:
        if not self.shape or any(d <= 0 for d in self.shape):
            raise SpecError(f"{where}: {self.name} must be fully static, got {self.shape}")
        if self.dram_layout == "hwc16" and not (self.logical_width and self.logical_channels):
            raise SpecError(
                f"{where}: {self.name} is hwc16 but lacks logical_width/logical_channels"
            )


# --------------------------------------------------------------------- dataset


@dataclass(frozen=True)
class DatasetContract:
    """What a dataset must look like for this policy to train on it.

    Making the rules data means every policy gets the same validation.
    """

    state_names: tuple[str, ...]
    state_dim: int
    action_dim: int
    camera_keys: tuple[str, ...]
    camera_shape: tuple[int, int, int]  # HWC, as stored in meta/info.json
    fps: int
    #: LeRobot feature names. Standard across every policy here, but named rather
    #: than hardcoded because export and normalization both index by them, and a
    #: dataset that renamed them would otherwise fail deep inside torch.
    state_key: str = "observation.state"
    action_key: str = "action"
    codebase_version: str | tuple[str, ...] = "v3.0"  # one version, or the accepted ones
    single_task: bool = True  # False => language-conditioned
    task_canonicalizer: str | None = None  # dotted path
    rename_map: Mapping[str, str] = field(default_factory=dict)
    empty_cameras: int = 0
    extra_meta_files: tuple[str, ...] = ()

    def validate(self, where: str) -> None:
        if len(self.state_names) != self.state_dim:
            raise SpecError(
                f"{where}: {len(self.state_names)} state_names but state_dim={self.state_dim}"
            )
        if not self.camera_keys:
            raise SpecError(f"{where}: at least one camera key is required")
        if len(self.camera_shape) != 3:
            raise SpecError(f"{where}: camera_shape must be (H, W, C)")


# --------------------------------------------------------------------- compile


@dataclass(frozen=True)
class CalibrationSource:
    kind: Literal["npz", "raw_f32", "random"] = "npz"
    producer: str | None = None
    samples: int = 8


@dataclass(frozen=True)
class GraphSpec:
    """One ONNX subgraph -> one MLA ELF.

    `name` is the single identifier used end to end: the ONNX stem, the ELF stem,
    the bundle's models/<name>/ directory, and the plan.json `graph` key. Keeping
    them equal is what lets bundle.model_elf(name) never branch.
    """

    name: str
    builder: str  # dotted path -> nn.Module factory
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    layout: Layout = "NCHW"
    compiler: Compiler = "afe"
    precision: Precision = "bf16"
    precision_fallback: tuple[Precision, ...] = ()
    #: Optional split quantization. ``precision`` remains the attempt/fallback
    #: label and supplies either side that is not overridden.  Some large
    #: transformer graphs need BF16 activations for range while quantizing the
    #: weights to INT8 to avoid a ModelSDK BF16 code-generation failure.
    activation_precision: Precision | None = None
    weight_precision: Precision | None = None
    calibration: CalibrationSource = field(default_factory=CalibrationSource)
    mla_tessellation: bool = True
    elf_from: Literal["retained", "mpk"] = "retained"
    #: ModelSDK occasionally hangs after closing a complete retained ELF for
    #: large graphs.  Opt in to the stable-file watchdog in that case.
    exit_on_stable_elf: bool = False
    #: Directory names this graph is known by in hand-built trees. PoLiMa names
    #: graphs for what they are (`vision`); hand-built trees may encode the
    #: precision and compiler in the directory (`vision_llima_bf16`). Listing the
    #: aliases is what lets --import-build adopt an existing build instead of
    #: demanding it be recompiled under new names.
    aliases: tuple[str, ...] = ()
    llima_args: tuple[str, ...] = ()
    #: Force the compiler-visible MLA input and output tensors to the same DRAM
    #: representation.  HWC is the contract used by device-resident chains:
    #: one ELF's output buffer can then be bound directly to the next ELF's
    #: input without a download, detessellation and upload.
    external_dram_layout: ExternalDramLayout = "compiler"
    #: ModelSDK 2.1's MPK path assumes four dimensions when an explicit HWC
    #: layout is selected. GR00T's token graphs are [N,W,C], so compilation
    #: wraps their public boundary as [N,1,W,C] with numerical no-op reshapes.
    promote_rank3_hwc: bool = False

    @property
    def elf_name(self) -> str:
        return f"{self.name}_stage1_mla.elf"

    @property
    def precisions(self) -> tuple[Precision, ...]:
        """Try `precision` first, followed by each explicitly declared fallback."""
        return (self.precision, *self.precision_fallback)

    def validate(self, where: str) -> None:
        where = f"{where}.{self.name}"
        if not self.inputs or not self.outputs:
            raise SpecError(f"{where}: needs at least one input and one output")
        for tensor in (*self.inputs, *self.outputs):
            tensor.validate(where)
        if self.compiler == "llima" and self.mla_tessellation:
            raise SpecError(f"{where}: llima-compiled graphs do not take MLA tessellation")
        if self.mla_tessellation and self.external_dram_layout != "compiler":
            raise SpecError(
                f"{where}: mla_tessellation and external_dram_layout are mutually exclusive"
            )
        if self.promote_rank3_hwc and self.external_dram_layout == "compiler":
            raise SpecError(f"{where}: promote_rank3_hwc requires an explicit external_dram_layout")
        if (
            self.promote_rank3_hwc
            and self.layout != "NHWC"
            and any(len(tensor.shape) == 3 for tensor in self.inputs)
        ):
            raise SpecError(f"{where}: promoting a rank-3 input requires layout='NHWC'")
        if self.promote_rank3_hwc and (
            len(self.inputs) != 1
            or len(self.outputs) != 1
            or len(self.inputs[0].shape) not in (3, 4)
            or len(self.outputs[0].shape) not in (3, 4)
            or (len(self.inputs[0].shape) == 4 and len(self.outputs[0].shape) == 4)
        ):
            raise SpecError(
                f"{where}: HWC promotion requires one input/output and at least one rank-3 boundary"
            )


@dataclass(frozen=True)
class CompilePlan:
    graphs: tuple[GraphSpec, ...]  # ordered; also the export order
    export_entry: str
    verify_entry: str | None = None
    fixture_entry: str | None = None
    normalization_entry: str | None = None
    #: Reference tensors written by `fixture_entry` and read back by the verify
    #: step and by bundle packing. Named here so the generic export driver has no
    #: policy-specific filename in it; ACT keeps `act_fixture.npz` because
    #: existing build trees use that name.
    fixture_file: str = "fixture.npz"
    verify_atol: float = 1e-4
    verify_rtol: float = 1e-3

    def graph(self, name: str) -> GraphSpec:
        for graph in self.graphs:
            if graph.name == name:
                return graph
        raise KeyError(f"no graph {name!r}; have {[g.name for g in self.graphs]}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(g.name for g in self.graphs)


# ---------------------------------------------------------- on-device runtime


#: Opcodes the native plan interpreter (runtime/src/plan.cpp) understands.
#: ACT needs only run_elf/pack/gather_strided.
OPCODES = (
    "run_elf",  # Runner[graph].run(in -> out)
    "run_elf_chain",  # one upload/download around compatible device-resident ELFs
    "pack",  # scatter sub-spans into one zeroed buffer at fixed offsets
    "slice",  # contiguous copy-out
    "gather_strided",  # for i<count: copy(src + i*stride, take)
    "pixel_unshuffle",  # fold a grid x grid x C map into (grid/f)^2 x C*f^2
    "scale",  # multiply by a scalar
    "clip_unit",  # clamp each value to [-1, 1]
    "matvec",  # y[o] = b[o] + sum_i W[o*K+i] * x[i], from .f32 sidecars
    "sincos_time",  # min/max-period sin/cos time embedding
    "sincos_state",  # grouped raw-angle sin/cos state encoding
    "euler",  # loop body xN with x -= dt*v
    "normalize",  # (x - mean) / std
    "denormalize",  # x * std + mean
    # Graph calls on plan-wide device buffers with uploads/downloads around them
    # (runtime/include/polima/device_chain.hpp); the native Plan checks them at load.
    "run_device_chain",
)


@dataclass(frozen=True)
class Step:
    op: str
    out: str
    args: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, where: str, buffers: Mapping[str, int]) -> None:
        if self.op not in OPCODES:
            raise SpecError(f"{where}: unknown opcode {self.op!r}; known: {list(OPCODES)}")
        if self.out not in buffers:
            raise SpecError(f"{where}: writes undeclared buffer {self.out!r}")


@dataclass(frozen=True)
class RuntimePlan:
    """Serialized to plan.json and interpreted on the DevKit.

    The ELF paths, element counts and packing offsets are data, so one
    polima_server serves every bundle.
    """

    buffers: Mapping[str, int]
    steps: tuple[Step, ...]
    result: str
    sidecars: tuple[str, ...] = ()
    loops: Mapping[str, int] = field(default_factory=dict)
    #: Plan-wide BF16 device buffers (name -> elements) that run_device_chain steps
    #: bind graph inputs and outputs to; they persist across steps.
    device_buffers: Mapping[str, int] = field(default_factory=dict)

    def validate(self, where: str, compile_plan: CompilePlan | None = None) -> None:
        if self.result not in self.buffers:
            raise SpecError(f"{where}: result {self.result!r} is not a declared buffer")
        for index, step in enumerate(self.steps):
            step.validate(f"{where}.steps[{index}]", self.buffers)
            if step.op in ("run_elf", "run_elf_chain"):
                graphs = (
                    (step.args.get("graph"),)
                    if step.op == "run_elf"
                    else tuple(step.args.get("graphs", ()))
                )
                if not graphs or any(not isinstance(graph, str) for graph in graphs):
                    raise SpecError(f"{where}.steps[{index}]: {step.op} needs graph name(s)")
                if compile_plan is not None:
                    unknown = [graph for graph in graphs if graph not in compile_plan.names]
                    if unknown:
                        raise SpecError(
                            f"{where}.steps[{index}]: {step.op} names unknown graph(s) {unknown}"
                        )
                for source in step.args.get("in", ()):
                    if source not in self.buffers:
                        raise SpecError(
                            f"{where}.steps[{index}]: reads undeclared buffer {source!r}"
                        )
                if step.op == "run_elf_chain" and len(step.args.get("in", ())) != 1:
                    raise SpecError(
                        f"{where}.steps[{index}]: run_elf_chain requires exactly one input buffer"
                    )
                if step.op == "run_elf_chain" and len(graphs) < 2:
                    raise SpecError(
                        f"{where}.steps[{index}]: run_elf_chain requires at least two graphs"
                    )
                if step.op == "run_elf_chain" and compile_plan is not None:
                    chain = [compile_plan.graph(graph) for graph in graphs]
                    for graph in chain:
                        if graph.external_dram_layout != "HWC":
                            raise SpecError(
                                f"{where}.steps[{index}]: shared graph {graph.name!r} "
                                "must compile with external_dram_layout='HWC'"
                            )
                        if len(graph.inputs) != 1 or len(graph.outputs) != 1:
                            raise SpecError(
                                f"{where}.steps[{index}]: shared graph {graph.name!r} "
                                "must have one input and one output"
                            )
                    for left, right in pairwise(chain):
                        if left.outputs[0].elements != right.inputs[0].elements:
                            raise SpecError(
                                f"{where}.steps[{index}]: shared edge {left.name}->{right.name} "
                                "has different element counts"
                            )
                    shared_elements = chain[0].outputs[0].elements
                    if any(graph.outputs[0].elements != shared_elements for graph in chain):
                        raise SpecError(
                            f"{where}.steps[{index}]: run_elf_chain requires one fixed "
                            "intermediate buffer size"
                        )


# ------------------------------------------------------------------------ wire


@dataclass(frozen=True)
class WireSpec:
    """The DevKit TCP protocol.

    The framing is data, so polima.wire.protocol derives the packers and there is
    one client for every policy.
    """

    magic: int
    version: int = 1
    default_port: int = 8092
    request_header: str = "<IIII"  # magic, version, request_id, flags
    response_header: str = "<IIIIfI"  # magic, version, request_id, status, ms, count
    request_tensors: tuple[TensorSpec, ...] = ()
    response_shape: tuple[int, ...] = ()
    normalization_side: Literal["client", "server"] = "client"
    stats_file: str = "normalization_stats.npz"

    @property
    def response_elements(self) -> int:
        return prod(self.response_shape) if self.response_shape else 0

    @property
    def magic_ascii(self) -> str:
        """Readable form -- 0x4D544341 is 'ACTM' little-endian."""
        try:
            return self.magic.to_bytes(4, "little").decode("ascii")
        except (UnicodeDecodeError, OverflowError):
            return hex(self.magic)


# ----------------------------------------------------------------------- robot


@dataclass(frozen=True)
class RobotSpec:
    camera_roles: tuple[tuple[str, str], ...]  # (lerobot camera key, UI label)
    joint_names: tuple[str, ...]
    actions_per_chunk: int
    default_fps: int = 30
    max_relative_target: int = 12  # degrees; the safety clamp
    aggregate_fn: str = "weighted_average"
    #: Ask for the next plan once the queue falls to this fraction of a chunk.
    chunk_threshold: float = 0.5
    task_string: str | None = None
    image_preprocessor: str | None = None  # dotted path; None => identity
    #: Camera pixel format.
    camera_fourcc: str = "MJPG"
    #: Per-role camera capture rates, independent of the control rate.
    camera_capture_fps: Mapping[str, int] = field(default_factory=dict)
    #: role -> substring of the /dev/v4l/by-id name identifying that camera.
    #: Discovery matches on these instead of falling back on enumeration order:
    #: /dev/videoN is assigned in plug order, so two cameras can silently swap
    #: across a reboot, and a swapped pair produces no error at all -- the policy
    #: runs and the arm reaches for the wrong place.
    camera_hints: Mapping[str, str] = field(default_factory=dict)
    #: Whether `polima robot` offers --calibrate (backs up the existing
    #: calibration, then runs lerobot-calibrate).
    supports_calibrate: bool = True
    calibration_id: str = "so-arm101"

    def camera_config(self, devices: Mapping[str, str], fps: int | None = None) -> str:
        """Render lerobot's --robot.cameras draccus blob.

        """
        entries = ", ".join(
            f"{key}: {{type: opencv, index_or_path: '{devices[key]}', "
            f"width: 640, height: 480, "
            f"fps: {fps or self.camera_capture_fps.get(key, self.default_fps)}, "
            f"fourcc: {self.camera_fourcc}}}"
            for key, _ in self.camera_roles
            if key in devices
        )
        return "{ " + entries + "}"

    @property
    def camera_keys(self) -> tuple[str, ...]:
        return tuple(key for key, _ in self.camera_roles)


def camera_hints_for_profile(
    calibration_id: str,
    roles: tuple[str, ...],
    inherited: Mapping[str, str],
) -> Mapping[str, str]:
    """Stable camera identities for a specialized robot profile."""
    return {role: hint for role, hint in inherited.items() if role in roles}


def camera_capture_for_profile(
    calibration_id: str,
    roles: tuple[str, ...],
    inherited_fourcc: str,
    inherited_fps: Mapping[str, int],
) -> tuple[str, Mapping[str, int]]:
    """Camera pixel format and capture rates for a specialized robot profile."""
    return inherited_fourcc, {
        role: int(inherited_fps[role]) for role in roles if role in inherited_fps
    }


# ------------------------------------------------------------------- the spec


@dataclass(frozen=True)
class SmokeSpec:
    """Pass/fail bar for the deployed-vs-PyTorch fixture check.

    Defaults are ACT's. A policy
    whose graphs are quantized lands further from the reference and says so
    here, rather than every policy sharing one bar only ACT can meet.
    """

    cosine_min: float = 0.999
    mean_abs_max: float = 0.01

    def to_dict(self) -> dict:
        return {"cosine_min": self.cosine_min, "mean_abs_max": self.mean_abs_max}


@dataclass(frozen=True)
class PolicySpec:
    name: str
    display_name: str
    dataset: DatasetContract
    compile: CompilePlan
    wire: WireSpec
    robot: RobotSpec
    runtime_plan_builder: str
    checkpoint_validator: str | None = None
    #: Optional pure-Python adapter that specializes a shared policy architecture
    #: for the robot profile recorded by a checkpoint/build contract.
    profile_entry: str | None = None
    #: Policy-specific shape/runtime facts discovered from a checkpoint trace.
    #: GR00T uses this for prompt length and image-token windows; keeping them
    #: here lets a build contract restore the exact runtime plan at pack time.
    runtime_context: Mapping[str, Any] = field(default_factory=dict)
    bundle_format: str = "polima-bundle-v1"
    #: What the deployed pipeline must match to be called good. Per-policy
    #: because the default is ACT's, and ACT is a small all-BF16 graph whose
    #: MLA output tracks PyTorch far more closely than a quantized VLA's does.
    smoke: SmokeSpec = field(default_factory=SmokeSpec)

    def graph(self, name: str) -> GraphSpec:
        return self.compile.graph(name)

    def build_runtime_plan(self, context: Any = None) -> RuntimePlan:
        return resolve(self.runtime_plan_builder)(self, context)

    def specialize_profile(self, **context: Any) -> PolicySpec:
        """Return the robot-profile variant selected by checkpoint metadata."""
        if not self.profile_entry:
            return self
        specialized = resolve(self.profile_entry)(self, **context)
        specialized.validate()
        return specialized

    def validate(self) -> None:
        """Cross-field consistency. Called by the registry at import time, so a
        malformed spec fails loudly at `polima doctor` rather than mid-deploy."""
        where = f"policy[{self.name}]"
        self.dataset.validate(where)

        if not self.compile.graphs:
            raise SpecError(f"{where}: no graphs declared")
        seen: set[str] = set()
        for graph in self.compile.graphs:
            if graph.name in seen:
                raise SpecError(f"{where}: duplicate graph name {graph.name!r}")
            seen.add(graph.name)
            graph.validate(f"{where}.compile")

        # The wire's action count must match what the robot client expects.
        expected = self.robot.actions_per_chunk * self.dataset.action_dim
        if self.wire.response_elements != expected:
            raise SpecError(
                f"{where}: wire.response_shape={self.wire.response_shape} is "
                f"{self.wire.response_elements} elements, but "
                f"actions_per_chunk*action_dim = {expected}"
            )

        # camera_roles may name either the short role ("overhead") or the full
        # feature key ("observation.images.overhead"); both must resolve.
        short_keys = {key.rsplit(".", 1)[-1] for key in self.dataset.camera_keys}
        unknown_cameras = set(self.robot.camera_keys) - short_keys - set(self.dataset.camera_keys)
        if unknown_cameras:
            raise SpecError(
                f"{where}: robot.camera_roles reference {sorted(unknown_cameras)}, "
                f"which are not in dataset.camera_keys {list(self.dataset.camera_keys)}"
            )
        if len(self.robot.camera_roles) != len(self.dataset.camera_keys):
            raise SpecError(
                f"{where}: {len(self.robot.camera_roles)} camera_roles but "
                f"{len(self.dataset.camera_keys)} dataset camera_keys"
            )

        plan = self.build_runtime_plan()
        plan.validate(f"{where}.plan", self.compile)

        # Every accelerator boundary must match the graph element counts. A
        # shared chain exposes only its first input and final output to the host.
        for index, step in enumerate(plan.steps):
            if step.op not in ("run_elf", "run_elf_chain"):
                continue
            graph_names = (
                (step.args["graph"],) if step.op == "run_elf" else tuple(step.args["graphs"])
            )
            first = self.compile.graph(graph_names[0])
            last = self.compile.graph(graph_names[-1])
            out_elements = sum(t.elements for t in last.outputs)
            declared = plan.buffers[step.out]
            if declared != out_elements:
                raise SpecError(
                    f"{where}.plan.steps[{index}]: buffer {step.out!r} holds {declared} "
                    f"elements but graph {last.name!r} outputs {out_elements}"
                )
            in_elements = sum(t.elements for t in first.inputs)
            sources = step.args.get("in", ())
            supplied = sum(plan.buffers[s] for s in sources)
            if supplied != in_elements:
                raise SpecError(
                    f"{where}.plan.steps[{index}]: inputs {list(sources)} supply {supplied} "
                    f"elements but graph {first.name!r} expects {in_elements}"
                )

    def summary(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "graphs": list(self.compile.names),
            "state_dim": self.dataset.state_dim,
            "action_dim": self.dataset.action_dim,
            "cameras": list(self.dataset.camera_keys),
            "fps": self.dataset.fps,
            "actions_per_chunk": self.robot.actions_per_chunk,
            "wire": {
                "magic": hex(self.wire.magic),
                "magic_ascii": self.wire.magic_ascii,
                "default_port": self.wire.default_port,
            },
        }


def sequence_names(items: Sequence[Any]) -> list[str]:
    return [getattr(i, "name", str(i)) for i in items]
