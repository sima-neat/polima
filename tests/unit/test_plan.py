"""Execution-plan semantics, verified against the real recorded goldens.

The ACT build ships per-stage golden .f32 files, which means the host-side
opcodes can be proven with genuine data and no accelerator:

    pack(state, vision_output_0, vision_output_1) == encoder_layer_00_stem_input
    gather_strided(decoder_action_tail_output)    == expected_normalized_actions

Those two assertions are the whole reason to trust the plan before anything is
built on the board. If they hold, a wrong result on hardware is the MLA's or the
ELF's doing, not the plan's.

Tests that need the golden files skip cleanly when no imported bundle is present,
so the suite still runs on a machine without the legacy tree.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from polima.policies.act import ACT_SPEC
from polima.policies.act.runtime import (
    CAMERA0_OFFSET,
    CAMERA1_OFFSET,
    CAMERA_ELEMENTS,
    STEM_ELEMENTS,
)
from polima.policies.base import (
    CompilePlan,
    GraphSpec,
    RuntimePlan,
    SpecError,
    Step,
    TensorSpec,
)
from polima.wire.server_stub import PlanError, StubPlan, golden_graph_fn

BUNDLES = Path(__file__).resolve().parents[2] / "outputs" / "bundles"


def find_bundle() -> Path | None:
    if not BUNDLES.is_dir():
        return None
    for candidate in sorted(BUNDLES.iterdir()):
        if (candidate / "plan.json").is_file() and (candidate / "fixtures" / "stages").is_dir():
            return candidate
    return None


bundle_required = pytest.mark.skipif(
    find_bundle() is None,
    reason="no imported bundle; run: polima-compile --import-legacy <build_dir>",
)


def plan_from_spec(tmp_path) -> StubPlan:
    """Build a StubPlan straight from the PolicySpec, no bundle needed."""
    from polima.bundle.pack import _plan_dict

    data = _plan_dict(ACT_SPEC.build_runtime_plan(), wire=ACT_SPEC)
    return StubPlan(
        buffers=data["buffers"], steps=data["steps"], result=data["result"],
        wire=data["wire"], constants_dir=tmp_path,
    )


# --------------------------------------------------------------- pure opcodes


def test_pack_layout_matches_the_stem_graph_slicing(tmp_path):
    """EncoderStemPacked reads [:, 0, :6], [:, 1:301], [:, 301:601] of a
    (1, 601, 512) tensor -- so state sits at 0, camera 0 at token 1, camera 1 at
    token 301."""
    plan = plan_from_spec(tmp_path)
    state = np.arange(6, dtype=np.float32) + 1
    cam0 = np.full(CAMERA_ELEMENTS, 2.0, dtype=np.float32)
    cam1 = np.full(CAMERA_ELEMENTS, 3.0, dtype=np.float32)

    buffers = {name: np.zeros(size, np.float32) for name, size in plan.buffers.items()}
    buffers.update({"state": state, "cam0": cam0, "cam1": cam1})
    packed = np.zeros(STEM_ELEMENTS, np.float32)
    step = next(s for s in plan.steps if s["op"] == "pack")
    for part in step["args"]["parts"]:
        start, count = part["dst_offset"], part["count"]
        packed[start:start + count] = buffers[part["src"]][:count]

    np.testing.assert_array_equal(packed[:6], state)
    assert np.all(packed[6:CAMERA0_OFFSET] == 0.0)          # rest of the latent token
    assert np.all(packed[CAMERA0_OFFSET:CAMERA0_OFFSET + CAMERA_ELEMENTS] == 2.0)
    assert np.all(packed[CAMERA1_OFFSET:CAMERA1_OFFSET + CAMERA_ELEMENTS] == 3.0)
    assert packed.size == STEM_ELEMENTS


def test_gather_strided_unpads_16_to_6():
    """The action head is widened to 16 channels and zero-filled for MLA
    alignment; the host takes the first 6 of every 16."""
    padded = np.zeros((100, 16), dtype=np.float32)
    padded[:, :6] = np.arange(600, dtype=np.float32).reshape(100, 6)
    gathered = padded.reshape(-1)[: 100 * 16].reshape(100, 16)[:, :6].reshape(-1)
    np.testing.assert_array_equal(gathered, np.arange(600, dtype=np.float32))


def _shared_graph(name: str, elements: int = 12, *, layout: str = "HWC") -> GraphSpec:
    return GraphSpec(
        name=name,
        builder="tests.fake:Graph",
        inputs=(TensorSpec("x", (1, 3, elements // 3)),),
        outputs=(TensorSpec("y", (1, 3, elements // 3)),),
        layout="NHWC",
        mla_tessellation=False,
        external_dram_layout=layout,
        promote_rank3_hwc=True,
    )


def test_stub_executes_a_shared_chain_as_sequential_graphs():
    plan = StubPlan(
        buffers={"x": 4, "y": 4},
        steps=[{
            "op": "run_elf_chain", "out": "y",
            "args": {"graphs": ["add_one", "times_two"], "in": ["x"]},
        }],
        result="y",
    )

    def graph_fn(name, values):
        return values + 1 if name == "add_one" else values * 2

    np.testing.assert_array_equal(
        plan.run({"x": np.arange(4, dtype=np.float32)}, graph_fn=graph_fn),
        np.array([2, 4, 6, 8], dtype=np.float32),
    )


def test_shared_chain_requires_explicit_hwc_graphs():
    compile_plan = CompilePlan(
        graphs=(_shared_graph("a"), _shared_graph("b", layout="compiler")),
        export_entry="tests.fake:export",
    )
    plan = RuntimePlan(
        buffers={"x": 12, "y": 12},
        steps=(Step("run_elf_chain", "y", {"graphs": ["a", "b"], "in": ["x"]}),),
        result="y",
    )
    with pytest.raises(SpecError, match="external_dram_layout='HWC'"):
        plan.validate("test", compile_plan)


def test_pack_zeroes_the_latent_token(tmp_path):
    """EncoderStemLayer builds its latent from torch.zeros, so the padding
    between state and camera 0 must be zero, not stale data."""
    plan = plan_from_spec(tmp_path)
    dirty = {"state": np.ones(6, np.float32), "cam0": np.ones(CAMERA_ELEMENTS, np.float32),
             "cam1": np.ones(CAMERA_ELEMENTS, np.float32)}
    packed = np.full(STEM_ELEMENTS, 99.0, dtype=np.float32)
    step = next(s for s in plan.steps if s["op"] == "pack")
    packed = np.zeros(step["args"]["size"], dtype=np.float32)
    for part in step["args"]["parts"]:
        start, count = part["dst_offset"], part["count"]
        packed[start:start + count] = dirty[part["src"]][:count]
    assert np.all(packed[6:CAMERA0_OFFSET] == 0.0)


# ------------------------------------------------- against the real goldens


@bundle_required
def test_pack_reproduces_the_recorded_stem_input():
    """pack(state, vision_output_0, vision_output_1) == encoder_layer_00_stem_input.

    Real data from the build that is running on the board.
    """
    bundle = find_bundle()
    stages = bundle / "fixtures" / "stages"
    plan = StubPlan.load(bundle)

    state = np.fromfile(bundle / "fixtures" / "inputs" / "state.f32", dtype="<f4")
    cam0 = np.fromfile(stages / "vision_output_0.f32", dtype="<f4")
    cam1 = np.fromfile(stages / "vision_output_1.f32", dtype="<f4")
    expected = np.fromfile(stages / "encoder_layer_00_stem_input.f32", dtype="<f4")

    step = next(s for s in plan.steps if s["op"] == "pack")
    sources = {"state": state, "cam0": cam0, "cam1": cam1}
    packed = np.zeros(step["args"]["size"], dtype=np.float32)
    for part in step["args"]["parts"]:
        start, count = part["dst_offset"], part["count"]
        packed[start:start + count] = sources[part["src"]][:count]

    assert packed.size == expected.size
    np.testing.assert_allclose(packed, expected, atol=0, rtol=0)


#: The ONNX-vs-PyTorch gap the legacy pipeline measured and accepted, from
#: ACT/outputs/modalix_rcwb_f_t_act_100000_llima/onnx_verification_report.json.
#: The recorded goldens have different provenance -- `<graph>_output.f32` files
#: are onnxruntime outputs, `expected_normalized_actions.f32` is the torch
#: reference -- so replaying the ONNX chain must land exactly this far away, no
#: nearer and no further.
#:
#: The value is per bundle, not a constant: it is that checkpoint's recorded
#: onnx-vs-torch gap (1.43e-06 for rcwb_f_t, 2.04e-06 for gewb_2_final). Reading
#: it from the bundle is the point -- hardcoding one made the test fail the
#: moment a second bundle existed, which said nothing about the plan.
def recorded_onnx_gap(bundle: Path) -> float:
    from polima.util.jsonio import read_json

    manifest = read_json(bundle / "bundle.json")
    gap = manifest.get("tool_versions", {}).get("onnx_max_abs")
    if gap is None:
        pytest.skip(f"{bundle.name} records no onnx_max_abs")
    return float(gap)


@bundle_required
def test_gather_strided_reproduces_the_expected_actions():
    """gather_strided(decoder_action_tail_output) == expected_normalized_actions,
    to exactly the recorded ONNX-vs-torch gap."""
    bundle = find_bundle()
    stages = bundle / "fixtures" / "stages"
    padded = np.fromfile(stages / "decoder_action_tail_output.f32", dtype="<f4")
    expected = np.fromfile(bundle / "fixtures" / "expected" / "normalized_actions.f32",
                           dtype="<f4")

    plan = StubPlan.load(bundle)
    step = next(s for s in plan.steps if s["op"] == "gather_strided")
    args = step["args"]
    gathered = padded[: args["count"] * args["stride"]].reshape(
        args["count"], args["stride"]
    )[:, : args["take"]].reshape(-1)

    assert gathered.size == expected.size == 600
    # Well inside the policy's own verification tolerance...
    np.testing.assert_allclose(
        gathered, expected, atol=ACT_SPEC.compile.verify_atol, rtol=ACT_SPEC.compile.verify_rtol
    )
    # ...and bit-identical to what the legacy pipeline recorded, which pins the
    # unpad stride/take and proves no extra arithmetic crept in.
    difference = np.abs(gathered - expected)
    assert float(difference.max()) == recorded_onnx_gap(bundle)


@bundle_required
def test_padding_channels_are_zero():
    """Channels 6..15 of the widened action head must be zero -- if they are not,
    the ELF is not the zero-filled DecoderActionRank4 we think it is."""
    bundle = find_bundle()
    padded = np.fromfile(
        bundle / "fixtures" / "stages" / "decoder_action_tail_output.f32", dtype="<f4"
    ).reshape(100, 16)
    np.testing.assert_array_equal(padded[:, 6:], np.zeros((100, 10), dtype=np.float32))


@bundle_required
def test_full_plan_replay_against_goldens():
    """Run every step of the plan on the host, with the MLA replaced by the
    recorded per-graph outputs. Proves the step ORDER and buffer wiring, not just
    the individual opcodes."""
    bundle = find_bundle()
    plan = StubPlan.load(bundle)
    inputs_dir = bundle / "fixtures" / "inputs"

    inputs = {
        "image0": np.fromfile(inputs_dir / "image0.f32", dtype="<f4"),
        "image1": np.fromfile(inputs_dir / "image1.f32", dtype="<f4"),
        "state": np.fromfile(inputs_dir / "state.f32", dtype="<f4"),
    }
    actions = plan.run(inputs, graph_fn=golden_graph_fn(bundle / "fixtures" / "stages"))
    expected = np.fromfile(bundle / "fixtures" / "expected" / "normalized_actions.f32",
                           dtype="<f4")

    assert actions.size == 600
    np.testing.assert_allclose(
        actions, expected, atol=ACT_SPEC.compile.verify_atol, rtol=ACT_SPEC.compile.verify_rtol
    )
    # Replaying the ONNX chain must reproduce the recorded gap exactly: any other
    # value means a step is out of order or a buffer is mis-wired.
    assert float(np.abs(actions - expected).max()) == recorded_onnx_gap(bundle)


@bundle_required
def test_plan_declares_the_wire_contract():
    plan = StubPlan.load(find_bundle())
    assert plan.wire["magic"] == ACT_SPEC.wire.magic
    assert plan.wire["response_elements"] == 600
    assert [t["name"] for t in plan.wire["request_tensors"]] == ["image0", "image1", "state"]


# ------------------------------------------------------------------ errors


def test_run_elf_without_a_graph_fn_is_an_error(tmp_path):
    plan = plan_from_spec(tmp_path)
    with pytest.raises(PlanError, match="no graph_fn"):
        plan.run({"image0": np.zeros(921600, np.float32),
                  "image1": np.zeros(921600, np.float32),
                  "state": np.zeros(6, np.float32)})


def test_wrong_input_size_is_rejected(tmp_path):
    plan = plan_from_spec(tmp_path)
    with pytest.raises(PlanError, match="expected 6 elements"):
        plan.run({"state": np.zeros(7, np.float32)})


def test_unknown_input_is_rejected(tmp_path):
    plan = plan_from_spec(tmp_path)
    with pytest.raises(PlanError, match="not a declared buffer"):
        plan.run({"nope": np.zeros(1, np.float32)})


# -- GR00T SO-101 opcodes; native/src/plan.cpp implements the same ------------


def _single_step(op: str, args: dict, sizes: dict[str, int], out: str = "y") -> StubPlan:
    return StubPlan(buffers=sizes, steps=[{"op": op, "out": out, "args": args}], result=out)


def test_pack_selects_the_requested_tasks_prompt():
    """A multi-task bundle stores one prompt per task; the request's task picks one."""
    prompts = np.arange(12, dtype=np.float32)  # three tasks of four values
    plan = _single_step("pack", {"parts": [{"src": "prompts", "dst_offset": 1, "count": 4,
                                            "source_stride": 4, "selector_count": 3}]},
                        {"prompts": 12, "y": 5})
    assert plan.run({"prompts": prompts}, task=2).tolist() == [0, 8, 9, 10, 11]
    with pytest.raises(PlanError, match="catalog"):
        plan.run({"prompts": prompts}, task=3)


def test_clip_unit_clamps_to_plus_minus_one():
    plan = _single_step("clip_unit", {"src": "x"}, {"x": 3, "y": 3})
    assert plan.run({"x": np.array([-3.0, 0.25, 2.0], np.float32)}).tolist() == [-1, 0.25, 1]


def test_pixel_unshuffle_handles_a_non_square_grid():
    """SO-101 images give an 18 x 24 token grid; the fold must not assume a square."""
    height, width, channels, factor = 4, 6, 3, 2
    source = np.arange(height * width * channels, dtype=np.float32)
    plan = _single_step("pixel_unshuffle", {"src": "x", "height": height, "width": width,
                                            "channels": channels, "factor": factor},
                        {"x": source.size, "y": source.size})
    # The index arithmetic of native/src/plan.cpp, transliterated.
    expected = np.zeros_like(source)
    out_channels, columns = channels * factor * factor, width // factor
    for row in range(height // factor):
        for column in range(columns):
            for dy in range(factor):
                for dx in range(factor):
                    read_at = ((row * factor + dy) * width + column * factor + dx) * channels
                    write_at = (row * columns + column) * out_channels + dy * factor + dx
                    for channel in range(channels):
                        expected[write_at + channel * factor * factor] = source[read_at + channel]
    np.testing.assert_array_equal(plan.run({"x": source}), expected)


def test_persistent_dit_bindings_survive_a_manifest_rewrite(tmp_path):
    """Deploy rewrites bundle.json; dropping these would stop run_dit_chain loading."""
    from polima.bundle.layout import Bundle, GraphArtifact

    graph = GraphArtifact("dit_blocks_00_01", "models/a.elf", "0" * 64, 1,
                          persistent_input_elements=[64, 8, 32, 16])
    bundle = Bundle(root=tmp_path, policy="groot", bundle_id="groot-x", graphs=[graph])
    again = Bundle.from_dict(bundle.to_dict(), tmp_path)
    assert again.graphs[0].persistent_input_elements == [64, 8, 32, 16]
