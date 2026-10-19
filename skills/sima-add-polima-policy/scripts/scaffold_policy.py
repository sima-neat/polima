#!/usr/bin/env python3
"""Create a new PoLiMa policy package and its unit-test skeleton.

usage: scaffold_policy.py NAME [--display-name TEXT] [--magic ABCD] [--port N] [--repo DIR]

Writes, under the repository:

  src/polima/policies/NAME/__init__.py   PolicySpec (dataset, graphs, wire, robot), registered
  src/polima/policies/NAME/runtime.py    build_plan: per-camera vision -> pack -> action head -> drop padding
  src/polima/policies/NAME/graphs.py     export, fixture, verification and checkpoint hooks (TODO)
  tests/unit/test_NAME.py                spec and plan validate; hooks are wired

The spec and plan validate as generated, so the tests pass before any model
code exists; every model-specific piece raises NotImplementedError naming the
reference to read. The template is ACT-shaped (two 480x640 cameras, six
joints, client-side mean/std normalization, no language, one pass, BF16 pass
mark): replace the shapes, and add the language input, denoising loop, LLiMa
prefix, or server-side normalization your policy needs from the references.
Nothing is registered in `registry.BUILTIN` until you add it.
"""
from __future__ import annotations

import argparse
import keyword
import re
import sys
from pathlib import Path

INIT = '''"""{display_name}.

Scaffolded by skills/sima-add-polima-policy/scripts/scaffold_policy.py: the
shapes are placeholders. Follow the skill's references in phase order.
"""

from __future__ import annotations

{imports}

{upper}_SPEC = PolicySpec(
    name="{name}",
    display_name="{display_name}",
    # --------------------------------------------------------------- dataset --
    dataset=DatasetContract(
        state_names=JOINT_NAMES,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        camera_keys=("observation.images.overhead", "observation.images.wrist"),
        camera_shape=(HEIGHT, WIDTH, 3),
        fps=30,
        codebase_version="v3.0",
        single_task=True,  # False when the policy reads a language instruction
    ),
    # --------------------------------------------------------------- compile --
    # One GraphSpec per ONNX graph / ELF: references/graph-decomposition.md.
    compile=CompilePlan(
        export_entry="polima.policies.{name}.graphs:export_all",
        verify_entry="polima.policies.{name}.graphs:verify_chain",
        fixture_entry="polima.policies.{name}.graphs:write_fixtures",
        fixture_file="{name}_fixture.npz",
        verify_atol=1e-4,
        verify_rtol=1e-3,
        graphs=(
            GraphSpec(
                name="vision",
                builder="polima.policies.{name}.graphs:VisionGraph",
                inputs=(TensorSpec("image", (1, 3, HEIGHT, WIDTH)),),
                outputs=(TensorSpec("camera_tokens", (1, 1, CAMERA_TOKENS, HIDDEN)),),
                layout="NCHW",
                precision="bf16",
                calibration=CalibrationSource("npz", samples=8),
            ),
            # State token, then each camera's tokens, packed on the host.
            GraphSpec(
                name="action_head",
                builder="polima.policies.{name}.graphs:ActionHeadGraph",
                inputs=(TensorSpec("packed", (1, 1, PACKED_TOKENS, HIDDEN)),),
                # Widened to 16 lanes for MLA alignment; the plan drops the padding.
                outputs=(TensorSpec("padded_actions", (1, 1, CHUNK, PADDED_ACTION_DIM)),),
                layout="NHWC",
                precision="bf16",
                calibration=CalibrationSource("npz", samples=8),
            ),
        ),
    ),
    # ------------------------------------------------------------------ wire --
    # references/wire-and-robot.md: own magic and port; float32 tensors.
    wire=WireSpec(
        magic={magic},  # "{magic_ascii}" little-endian
        version=1,
        default_port={port},
        request_tensors=(
            TensorSpec("image0", (HEIGHT, WIDTH, 3)),
            TensorSpec("image1", (HEIGHT, WIDTH, 3)),
            TensorSpec("state", (STATE_DIM,)),
        ),
        response_shape=(CHUNK, ACTION_DIM),
        normalization_side="client",
        stats_file="normalization_stats.npz",
    ),
    # ----------------------------------------------------------------- robot --
    robot=RobotSpec(
        camera_roles=(("overhead", "Overhead"), ("wrist", "Wrist")),
        joint_names=JOINT_NAMES,
        actions_per_chunk=CHUNK,
        default_fps=30,
        max_relative_target=12,
    ),
    runtime_plan_builder="polima.policies.{name}.runtime:build_plan",
    checkpoint_validator="polima.policies.{name}.graphs:validate_checkpoint",
    # The pass mark of a BF16 build; INT8-weight builds use 0.995 / 0.03.
    smoke=SmokeSpec(cosine_min=0.999, mean_abs_max=0.01),
)

register_policy({upper}_SPEC)

__all__ = ["{upper}_SPEC"]
'''

RUNTIME = '''"""{display_name}: the runtime plan the board executes (references/runtime-plan-and-opcodes.md).

Placeholder geometry: two cameras through one vision graph, the state and
camera tokens packed into the action head's input, padded action lanes dropped.
"""

from __future__ import annotations

from typing import Any

from polima.policies.base import PolicySpec, RuntimePlan, Step

JOINT_NAMES = (
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
)
STATE_DIM = len(JOINT_NAMES)
ACTION_DIM = STATE_DIM
HEIGHT, WIDTH = 480, 640
HIDDEN = 512
CAMERA_TOKENS = 300
PACKED_TOKENS = 1 + 2 * CAMERA_TOKENS   # state token, then camera 0, then camera 1
CHUNK = 50
PADDED_ACTION_DIM = 16                  # MLA channel alignment

IMAGE_ELEMENTS = HEIGHT * WIDTH * 3
CAMERA_ELEMENTS = CAMERA_TOKENS * HIDDEN
PACKED_ELEMENTS = PACKED_TOKENS * HIDDEN
PADDED_ELEMENTS = CHUNK * PADDED_ACTION_DIM


def build_plan(spec: PolicySpec, context: Any = None) -> RuntimePlan:
    state_dim, action_dim = spec.dataset.state_dim, spec.dataset.action_dim
    steps = [
        Step("run_elf", "cam0", {{"graph": "vision", "in": ["image0"]}}),
        Step("run_elf", "cam1", {{"graph": "vision", "in": ["image1"]}}),
        Step("pack", "packed", {{"size": PACKED_ELEMENTS, "parts": [
            {{"src": "state", "dst_offset": 0, "count": state_dim}},
            {{"src": "cam0", "dst_offset": HIDDEN, "count": CAMERA_ELEMENTS}},
            {{"src": "cam1", "dst_offset": (1 + CAMERA_TOKENS) * HIDDEN, "count": CAMERA_ELEMENTS}},
        ]}}),
        Step("run_elf", "padded", {{"graph": "action_head", "in": ["packed"]}}),
        Step("gather_strided", "actions", {{"src": "padded", "stride": PADDED_ACTION_DIM,
                                            "take": action_dim, "count": CHUNK}}),
    ]
    return RuntimePlan(
        buffers={{"image0": IMAGE_ELEMENTS, "image1": IMAGE_ELEMENTS, "state": state_dim,
                  "cam0": CAMERA_ELEMENTS, "cam1": CAMERA_ELEMENTS, "packed": PACKED_ELEMENTS,
                  "padded": PADDED_ELEMENTS, "actions": CHUNK * action_dim}},
        steps=tuple(steps),
        result="actions",
        sidecars=(),  # client-side normalization needs no server constants
    )
'''

GRAPHS = '''"""{display_name}: static graph modules, export, fixtures and verification.

Runs in the policy's own export interpreter (torch and the model's framework),
never in the ModelSDK environment. Each hook below names the reference that
explains it; replace the NotImplementedError bodies in phase order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_REFS = "skills/sima-add-polima-policy/references"


def _todo(what: str, reference: str) -> NotImplementedError:
    return NotImplementedError(f"{{what}}: see {{_REFS}}/{{reference}}")


def validate_checkpoint(checkpoint: Path) -> None:
    """Refuse checkpoints this policy cannot run (shapes, chunk length, unsupported options)."""
    raise _todo("validate_checkpoint", "feasibility.md")


def load_policy(checkpoint: Path, **_: Any):
    """Load the original policy (the PyTorch reference) from its framework."""
    raise _todo("load_policy", "graph-decomposition.md#export")


def build_modules(policy: Any) -> dict:
    """Static, rank-4 torch modules, one per GraphSpec, matching the reference in FP32."""
    raise _todo("build_modules", "graph-decomposition.md#static-graph-modules")


def trace(modules: dict, sample: dict, image_keys: list[str]) -> dict:
    """Run one observation through the modules and capture every graph input (calibration)."""
    raise _todo("trace", "graph-decomposition.md#calibration")


def export_all(build_dir: Path, modules: dict, samples: list, traces: list, image_keys: list[str]) -> None:
    """ONNX (opset 17, dynamo=False) and calibration/<graph>.npz for every graph."""
    raise _todo("export_all", "graph-decomposition.md#export")


def write_fixtures(build_dir: Path, sample: dict, trace: dict, image_keys: list[str], postprocessor: Any) -> None:
    """Wire inputs and the ORIGINAL policy's expected actions, for several frames."""
    raise _todo("write_fixtures", "graph-decomposition.md#fixtures-and-verification")


def verify_chain(onnx_dir: Path, fixture: Path, report: Path, atol: float, rtol: float, stage_dir: Path) -> dict:
    """Replay the ONNX chain in onnxruntime against the fixture; write the report."""
    raise _todo("verify_chain", "validation-matrix.md")


class VisionGraph:  # replace with a torch.nn.Module built from checkpoint tensors
    def __init__(self, *_: Any, **__: Any) -> None:
        raise _todo("VisionGraph", "graph-decomposition.md#static-graph-modules")


class ActionHeadGraph:  # replace with a torch.nn.Module built from checkpoint tensors
    def __init__(self, *_: Any, **__: Any) -> None:
        raise _todo("ActionHeadGraph", "graph-decomposition.md#static-graph-modules")
'''

TEST = '''"""{display_name}: spec, runtime plan and wiring (scaffolded; extend per phase)."""

import pytest

from polima.policies.base import resolve
from polima.policies.registry import get_policy


def test_spec_validates_and_its_plan_matches_the_graphs():
    spec = get_policy("{name}")
    spec.validate()
    plan = spec.build_runtime_plan()
    plan.validate("{name}", spec.compile)
    assert plan.buffers[plan.result] == spec.wire.response_elements


def test_every_dotted_entry_point_resolves():
    spec = get_policy("{name}")
    for path in (spec.runtime_plan_builder, spec.checkpoint_validator, spec.compile.export_entry,
                 spec.compile.verify_entry, spec.compile.fixture_entry,
                 *(graph.builder for graph in spec.compile.graphs)):
        assert resolve(path) is not None


def test_model_hooks_name_their_reference_until_implemented():
    from polima.policies.{name} import graphs

    with pytest.raises(NotImplementedError, match="references/"):
        graphs.load_policy(None)
'''

IMPORTS = {
    "polima.policies.{name}.runtime": ("ACTION_DIM", "CAMERA_TOKENS", "CHUNK", "HEIGHT", "HIDDEN", "JOINT_NAMES",
                                       "PACKED_TOKENS", "PADDED_ACTION_DIM", "STATE_DIM", "WIDTH"),
    "polima.policies.base": ("CalibrationSource", "CompilePlan", "DatasetContract", "GraphSpec", "PolicySpec",
                             "RobotSpec", "SmokeSpec", "TensorSpec", "WireSpec"),
    "polima.policies.registry": ("register_policy",),
}


def import_block(name: str) -> str:
    """The spec module's imports, sorted by module as the repository's linter wants."""
    lines = []
    for module, names in sorted((m.format(name=name), n) for m, n in IMPORTS.items()):
        if len(names) == 1:
            lines.append(f"from {module} import {names[0]}")
        else:
            lines.append(f"from {module} import (\n" + "".join(f"    {n},\n" for n in names) + ")")
    return "\n".join(lines)


CHECKLIST = """
Next, in phase order (skills/sima-add-polima-policy/SKILL.md):
  1. Fill references/feasibility.md and pick the route.
  2. Replace the placeholder shapes in policies/{name}/runtime.py and __init__.py.
  3. Implement policies/{name}/graphs.py (static modules -> export -> fixtures -> verify).
  4. Compile: per-graph precision and layout; a compile/{name}.py staged builder for long builds,
     routed from src/polima/cli/compile.py.
  5. Robot: src/polima/robot/policy.py (_prepare_request, predict), src/polima/robot/so101.py
     (SUPPORTED_POLICIES, RECIPES, load_contract), src/polima/bundle/importer.py (robot files, rest pose).
  6. Integration: registry.BUILTIN, workbench/app.py POLICIES, workbench/inputs.py prerequisites,
     workbench/jobs.py compile_command, cli/doctor.py, config paths.{name}_python (+ loader ENV_MAP),
     README tables, docs/{name}.md, docs/your-own-model.md.
  7. pytest tests/unit/test_{name}.py && ruff check src tests
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("name", help="policy package name (lowercase identifier), e.g. pi05")
    parser.add_argument("--display-name", help="human-readable name (default: NAME)")
    parser.add_argument("--magic", help="four ASCII characters for the wire magic (default: from NAME)")
    parser.add_argument("--port", type=int, default=8099,
                        help="default server port (default 8099; in use: ACT 8092, GR00T 8093, Pi0.5 8094, "
                             "SmolVLA 8081)")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3],
                        help="repository root (default: the one containing this skill)")
    args = parser.parse_args(argv)

    name = args.name
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or keyword.iskeyword(name):
        parser.error("NAME must be a lowercase Python identifier")
    magic_ascii = (args.magic or (name.upper().replace("_", "") + "XXXX")[:4])
    if len(magic_ascii) != 4 or not magic_ascii.isascii():
        parser.error("--magic must be four ASCII characters")
    repo = args.repo.resolve()
    package = repo / "src" / "polima" / "policies" / name
    test = repo / "tests" / "unit" / f"test_{name}.py"
    if not (repo / "src" / "polima" / "policies" / "base.py").is_file():
        parser.error(f"{repo} is not a PoLiMa repository")
    if not test.parent.is_dir():  # check every target before writing anything
        parser.error(f"{test.parent} does not exist")
    if package.exists() or test.exists():
        parser.error(f"{package} or {test} already exists; refusing to overwrite")

    values = {"name": name, "upper": name.upper(), "display_name": args.display_name or name,
              "magic": hex(int.from_bytes(magic_ascii.encode(), "little")), "magic_ascii": magic_ascii,
              "port": args.port}
    values["imports"] = import_block(name)
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(INIT.format(**values))
    (package / "runtime.py").write_text(RUNTIME.format(**values))
    (package / "graphs.py").write_text(GRAPHS.format(**values))
    test.write_text(TEST.format(**values))
    for path in (package / "__init__.py", package / "runtime.py", package / "graphs.py", test):
        print(f"wrote {path.relative_to(repo)}")
    print(CHECKLIST.format(name=name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
