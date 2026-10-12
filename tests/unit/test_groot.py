"""GR00T's spec and execution plan, pinned against the DevKit sources.

GR00T is the first policy PoLiMa cuts into more graphs than a person can check
by eye -- forty-five -- and the first whose pipeline is two device-resident
chains with host arithmetic between them. Both facts make the interesting
failures silent: a transposed channel fold, a mask fed to the wrong half of the
blocks, or an Euler update with the wrong sign all produce finite, plausible,
wrong actions.

So the geometry is verified against the C++ that currently runs this policy on
the board (`GR00T-N1.6/devkit/*.cpp`) rather than restated here, and the host
arithmetic is verified by replaying the plan with the MLA stubbed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from polima.policies.groot import GROOT_SPEC
from polima.policies.groot import runtime as rt
from polima.policies.registry import get_policy
from polima.wire.server_stub import StubPlan

GROOT_TREE = Path(__file__).resolve().parents[3] / "GR00T-N1.6"
DEVKIT = GROOT_TREE / "devkit"
BUILD = GROOT_TREE / "outputs" / "modalix_groot_n1d6_base_gr1"

needs_devkit = pytest.mark.skipif(
    not (DEVKIT / "groot_action_llima" / "groot_action_llima.cpp").is_file(),
    reason="GR00T-N1.6 devkit sources not present",
)
needs_fixture = pytest.mark.skipif(
    not (BUILD / "eagle_fixture" / "eagle_fixture.npz").is_file(),
    reason="GR00T-N1.6 build tree not present",
)


def devkit_constants(name: str) -> dict[str, int]:
    """`constexpr size_t kFoo = 123;` -> {"kFoo": 123}, arithmetic excluded."""
    source = (DEVKIT / name / f"{name}.cpp").read_text(encoding="utf-8")
    pattern = re.compile(r"constexpr (?:size_t|int|int64_t) (k\w+) = (\d+);")
    return {match.group(1): int(match.group(2)) for match in pattern.finditer(source)}


# ------------------------------------------------------------------ the spec


def test_spec_registers_with_forty_five_graphs():
    spec = get_policy("groot")
    assert spec is GROOT_SPEC
    assert len(spec.compile.graphs) == 45
    # 26 Eagle stages + 19 action graphs, as the two export reports record.
    assert sum(name.startswith("eagle_") for name in spec.compile.names) == 26


def test_the_two_chains_cover_every_eagle_stage_but_the_connector():
    chained = set(rt.VISION_CHAIN) | set(rt.QWEN_CHAIN)
    eagle = {name for name in GROOT_SPEC.compile.names if name.startswith("eagle_")}
    # The connector is the one Eagle graph left out, because the host has to
    # fold 18x18x1152 into 9x9x4608 immediately before it.
    assert eagle - chained == {"eagle_vision_connector"}
    assert len(rt.VISION_CHAIN) == 16 and len(rt.QWEN_CHAIN) == 9


def test_chained_graphs_expose_a_flat_hwc_boundary():
    # run_elf_chain binds one ELF's output buffer straight to the next ELF's
    # input. That is only sound if neither end is tessellated.
    for name in (*rt.VISION_CHAIN, *rt.QWEN_CHAIN):
        graph = GROOT_SPEC.graph(name)
        assert graph.external_dram_layout == "HWC", name
        assert not graph.mla_tessellation, name
        assert graph.layout == "NHWC", name
        # Rank-3 [1, tokens, channels] stages need HWC promotion. The Qwen pairs
        # take a rank-4 [1, 1, tokens, channels] hidden state plus the task's
        # attention mask, so there is nothing to promote (and promotion is only
        # defined for one input).
        rank3 = len(graph.inputs[0].shape) == 3
        assert graph.promote_rank3_hwc == rank3, name


def test_qwen_pairs_take_the_task_attention_mask():
    for name in rt.QWEN_CHAIN:
        if not name.startswith("eagle_qwen_"):
            continue
        hidden, mask = GROOT_SPEC.graph(name).inputs
        assert hidden.shape == (1, 1, rt.SEQUENCE, rt.LANGUAGE_CHANNELS), name
        assert mask.name == "attention_mask"
        assert mask.shape == (1, rt.SEQUENCE, rt.SEQUENCE, 1), name


def test_unchained_outputs_download_as_plain_bf16():
    # Measured on Modalix (MyCobot and SO-101 fixtures): these ELFs return
    # ordinary contiguous BF16. Decoding them as HWC16 byte planes produced
    # non-finite values; plain decoding matched ONNX. Every SO-101 bundle that
    # ran on the arm on 2026-10-05 declares them plain.
    for name in ("eagle_vision_connector", "state_project", "action_project",
                 "action_tail", *rt.block_names()):
        output = GROOT_SPEC.graph(name).outputs[0]
        assert output.dram_layout == "plain", name
        assert output.logical_width is None and output.logical_channels is None, name


def test_wire_magic_is_grut():
    assert GROOT_SPEC.wire.magic_ascii == "GRUT"
    assert GROOT_SPEC.wire.default_port == 8093


def test_dataset_needs_the_v2_1_downgrade_in_its_own_environment():
    converter = GROOT_SPEC.dataset.converter
    assert converter is not None
    assert converter.target_codebase_version == "v2.1"
    # The converter and the trainer need incompatible lerobot versions, which is
    # why this is a second env and not a function call.
    assert converter.conda_env != GROOT_SPEC.train.conda_env


# ---------------------------------------------------------------- geometry


@needs_devkit
def test_action_geometry_matches_the_devkit_binary():
    constants = devkit_constants("groot_action_llima")
    assert constants["kStateElements"] == rt.STATE_LANE
    assert constants["kWidth"] == rt.CHUNK
    assert constants["kActionChannels"] == rt.ACTION_LANE
    assert constants["kHiddenWidth"] == rt.HIDDEN_WIDTH
    assert constants["kHiddenChannels"] == rt.HIDDEN_CHANNELS
    assert constants["kTembElements"] == rt.TEMB_ELEMENTS
    assert constants["kBackboneWidth"] == rt.SEQUENCE
    assert constants["kBackboneChannels"] == rt.LANGUAGE_CHANNELS
    assert constants["kMaskElements"] == rt.MASK_ELEMENTS
    assert constants["kBlockPairs"] == rt.BLOCK_PAIRS


@needs_devkit
def test_eagle_geometry_matches_the_devkit_binary():
    constants = devkit_constants("groot_eagle_llima")
    assert constants["kImageSide"] == rt.IMAGE_SIDE
    assert constants["kPatch"] == rt.PATCH
    assert constants["kPatchSide"] == rt.PATCH_SIDE
    assert constants["kVisionWidth"] == rt.VISION_WIDTH
    assert constants["kPatchChannels"] == rt.PATCH_CHANNELS
    assert constants["kVisionChannels"] == rt.VISION_CHANNELS
    assert constants["kConnectorWidth"] == rt.CONNECTOR_WIDTH
    assert constants["kConnectorChannels"] == rt.CONNECTOR_CHANNELS
    assert constants["kSequence"] == rt.SEQUENCE
    assert constants["kLanguageChannels"] == rt.LANGUAGE_CHANNELS
    assert constants["kVisionPairs"] == rt.VISION_PAIRS
    assert constants["kQwenPairs"] == rt.QWEN_PAIRS


@needs_fixture
def test_image_tokens_are_contiguous_where_the_pack_assumes():
    # The language embedding is a `pack` of one constant plus one live buffer,
    # which is only correct while the image tokens form one unbroken run.
    fixture = np.load(BUILD / "eagle_fixture" / "eagle_fixture.npz")
    positions = np.flatnonzero(fixture["input_ids"][0] == 151669)
    assert positions.size == rt.CONNECTOR_WIDTH
    assert positions[0] == rt.IMAGE_TOKEN_START
    assert np.all(np.diff(positions) == 1)


# -------------------------------------------------------------------- plan


def plan_dict() -> dict:
    plan = GROOT_SPEC.build_runtime_plan()
    return {
        "buffers": dict(plan.buffers),
        "steps": [{"op": s.op, "out": s.out, "args": dict(s.args)} for s in plan.steps],
        "result": plan.result,
    }


def graph_calls(op_filter=("run_elf", "run_elf_chain")) -> list[str]:
    calls = []
    for step in plan_dict()["steps"]:
        if step["op"] == "run_elf":
            calls.append(step["args"]["graph"])
        elif step["op"] == "run_elf_chain":
            calls.extend(step["args"]["graphs"])
    return calls


def test_plan_runs_each_graph_the_right_number_of_times():
    calls = graph_calls()
    # Eagle runs once; every action graph runs once per denoise step.
    assert calls.count("eagle_vision_patch") == 1
    assert calls.count("eagle_output_norm") == 1
    assert calls.count("state_project") == 1
    assert calls.count("action_project") == rt.DENOISE_STEPS
    assert calls.count("action_tail") == rt.DENOISE_STEPS
    for name in rt.block_names():
        assert calls.count(name) == rt.DENOISE_STEPS, name
    # 26 Eagle + 1 state + 4 x (1 project + 16 pairs + 1 tail)
    assert len(calls) == 26 + 1 + rt.DENOISE_STEPS * 18


def test_denoise_loop_is_unrolled_not_looped():
    assert not GROOT_SPEC.build_runtime_plan().loops


def test_blocks_alternate_text_and_image_masks_by_pair_parity():
    # export_groot_modalix_action.py picks `text if start % 4 == 0 else image`
    # over block starts, which is pair parity. One mask everywhere still runs.
    seen = {}
    for step in plan_dict()["steps"]:
        if step["op"] != "run_elf":
            continue
        graph = step["args"]["graph"]
        if not graph.startswith("dit_blocks_"):
            continue
        mask = step["args"]["in"][3]
        seen.setdefault(graph, set()).add(mask)
    assert len(seen) == rt.BLOCK_PAIRS
    for pair, name in enumerate(rt.block_names()):
        expected = "text_mask" if pair % 2 == 0 else "image_mask"
        assert seen[name] == {expected}, name


def test_euler_adds_because_groot_integrates_forward():
    # The opcode computes `out - scalar * src`; GR00T's update is `x += DT * v`.
    steps = [s for s in plan_dict()["steps"] if s["op"] == "euler"]
    assert len(steps) == rt.DENOISE_STEPS
    assert all(step["args"]["scalar"] == pytest.approx(-rt.DT) for step in steps)


def test_state_is_normalized_before_padding_not_after():
    # Normalizing the 128-wide lane would apply the statistics to the 122
    # padding slots, turning structural zeros into -mean/std.
    ops = [(s["op"], s["out"]) for s in plan_dict()["steps"]]
    assert ops.index(("normalize", "state_normalized")) < ops.index(("pack", "state_lane"))
    normalize = next(s for s in plan_dict()["steps"] if s["op"] == "normalize")
    assert GROOT_SPEC.build_runtime_plan().buffers[normalize["out"]] == rt.STATE_DIM


def test_no_elf_reads_and_writes_the_same_buffer():
    for step in plan_dict()["steps"]:
        if step["op"] in ("run_elf", "run_elf_chain"):
            assert step["out"] not in step["args"]["in"], step


def test_vision_tokens_land_on_the_image_window():
    pack = next(s for s in plan_dict()["steps"]
                if s["op"] == "pack" and s["out"] == "language_embeddings")
    prompt, image = pack["args"]["parts"]
    assert prompt["sidecar"] and prompt["count"] == rt.BACKBONE_ELEMENTS
    assert not image.get("sidecar")
    assert image["dst_offset"] == rt.IMAGE_TOKEN_START * rt.LANGUAGE_CHANNELS
    assert image["count"] == rt.CONNECTOR_WIDTH * rt.LANGUAGE_CHANNELS


# ------------------------------------------------------- host arithmetic


def test_pixel_unshuffle_matches_the_channel_order_torch_produces():
    # out[(i*s+j), c*f^2 + dy*f + dx] = src[(i*f+dy)*grid + (j*f+dx), c].
    # Transposing this still yields finite features, so it is checked directly.
    grid, channels, factor = 6, 3, 2
    side = grid // factor
    source = np.arange(grid * grid * channels, dtype=np.float32)
    expected = np.zeros((side * side, channels * factor * factor), dtype=np.float32)
    for i in range(side):
        for j in range(side):
            for c in range(channels):
                for dy in range(factor):
                    for dx in range(factor):
                        expected[i * side + j, c * factor * factor + dy * factor + dx] = (
                            source[((i * factor + dy) * grid + (j * factor + dx)) * channels + c]
                        )

    plan = StubPlan(
        buffers={"src": source.size, "dst": expected.size},
        steps=[{"op": "pixel_unshuffle", "out": "dst",
                "args": {"src": "src", "grid": grid, "channels": channels,
                         "factor": factor}}],
        result="dst",
    )
    got = plan.run({"src": source})
    assert np.array_equal(got, expected.ravel())


def test_plan_replays_end_to_end_with_the_mla_stubbed(tmp_path):
    """Buffer plumbing, start to finish.

    Every ELF is replaced by a deterministic function of its input, so this says
    nothing about the model -- only that all 45 graphs receive the element counts
    the spec declares and that the result reaches the wire's shape.
    """
    plan_json = plan_dict()
    sizes = plan_json["buffers"]
    seen: list[tuple[str, int]] = []

    def graph_fn(name: str, values: np.ndarray) -> np.ndarray:
        seen.append((name, values.size))
        graph = GROOT_SPEC.graph(name)
        expected = sum(tensor.elements for tensor in graph.inputs)
        assert values.size == expected, f"{name}: got {values.size}, spec says {expected}"
        out = sum(tensor.elements for tensor in graph.outputs)
        # A cheap, input-dependent stand-in: enough that a mis-plumbed buffer
        # changes the answer, cheap enough to run 45 graphs in a unit test.
        return np.full(out, float(values.sum() % 7.0), dtype=np.float32)

    constants = {
        "prompt_embedding": np.zeros(rt.BACKBONE_ELEMENTS, dtype=np.float32),
        "image_additive_mask": np.zeros(rt.MASK_ELEMENTS, dtype=np.float32),
        "text_additive_mask": np.zeros(rt.MASK_ELEMENTS, dtype=np.float32),
        # The Qwen pairs' causal/key-padding mask.
        "qwen_attention_mask": np.zeros(rt.SEQUENCE * rt.SEQUENCE, dtype=np.float32),
        "state_mean": np.zeros(rt.STATE_DIM, dtype=np.float32),
        "state_std": np.ones(rt.STATE_DIM, dtype=np.float32),
        "action_mean": np.zeros(rt.ACTION_DIM, dtype=np.float32),
        "action_std": np.ones(rt.ACTION_DIM, dtype=np.float32),
    }
    for index in range(rt.DENOISE_STEPS):
        constants[f"tau_embedding_{index}"] = np.zeros(rt.TAU_ELEMENTS, dtype=np.float32)
        constants[f"timestep_embedding_{index}"] = np.zeros(rt.TEMB_ELEMENTS, dtype=np.float32)

    for name, values in constants.items():
        values.astype("<f4").tofile(tmp_path / name)

    plan = StubPlan(buffers=sizes, steps=plan_json["steps"],
                    result=plan_json["result"], constants_dir=tmp_path)
    result = plan.run(
        {
            "patches": np.linspace(0, 1, sizes["patches"], dtype=np.float32),
            "state": np.arange(rt.STATE_DIM, dtype=np.float32),
            "noise": np.zeros(sizes["noise"], dtype=np.float32),
        },
        graph_fn=graph_fn,
    )
    assert result.size == GROOT_SPEC.wire.response_elements == rt.CHUNK * rt.ACTION_DIM
    assert len(seen) == 26 + 1 + rt.DENOISE_STEPS * 18


# ------------------------------------------------------------- the torch side

torch = pytest.importorskip("torch", reason="graphs.py is the only torch-side module")
nn = torch.nn
graphs = pytest.importorskip("polima.policies.groot.graphs")


def test_the_client_and_the_export_patchify_identically():
    """A disagreement here would calibrate on one tensor and infer on another.

    `patchify` runs at export against NCHW pixels; `patchify_for_wire` runs in
    the live client against an HWC frame. They must produce the same 324x588.
    """
    pixels = torch.randn(1, 3, rt.IMAGE_SIDE, rt.IMAGE_SIDE)
    from_export = graphs.patchify(pixels, rt.PATCH)[0].numpy()
    from_wire = graphs.patchify_for_wire(pixels[0].permute(1, 2, 0).numpy())
    assert from_export.shape == (rt.VISION_WIDTH, rt.PATCH_CHANNELS)
    assert np.array_equal(from_export, from_wire)


def test_the_opcode_reproduces_torch_pixel_unshuffle_at_eagle_size():
    """The board's fold and the export's fold, on the real 18x18x1152 map."""
    hidden = torch.randn(1, rt.VISION_WIDTH, rt.VISION_CHANNELS)
    expected = graphs.pixel_unshuffle_tokens(hidden, rt.PATCH_SIDE, rt.UNSHUFFLE)
    assert tuple(expected.shape) == (1, rt.CONNECTOR_WIDTH, rt.CONNECTOR_CHANNELS)
    plan = StubPlan(
        buffers={"src": hidden.numel(), "dst": expected.numel()},
        steps=[{"op": "pixel_unshuffle", "out": "dst",
                "args": {"src": "src", "grid": rt.PATCH_SIDE,
                         "channels": rt.VISION_CHANNELS, "factor": rt.UNSHUFFLE}}],
        result="dst",
    )
    assert np.array_equal(plan.run({"src": hidden.numpy().ravel()}),
                          expected.numpy().ravel())


def test_vision_post_norm_matmuls_equal_the_layernorm_they_replace():
    layer_norm = nn.LayerNorm(rt.VISION_CHANNELS)
    nn.init.normal_(layer_norm.weight)
    nn.init.normal_(layer_norm.bias)
    hidden = torch.randn(1, rt.VISION_WIDTH, rt.VISION_CHANNELS)
    assert torch.allclose(graphs.VisionPostNorm(layer_norm)(hidden),
                          layer_norm(hidden), atol=2e-5)


def test_output_norm_equals_rms_then_layernorm():
    """The fused stage is Qwen's final RMSNorm followed by the head's vlln."""

    class Rms(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(width))
            self.variance_epsilon = 1e-6

        def forward(self, hidden):
            scale = torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
            return hidden * scale * self.weight

    rms = Rms(rt.LANGUAGE_CHANNELS)
    vlln = nn.LayerNorm(rt.LANGUAGE_CHANNELS)
    nn.init.normal_(vlln.weight)
    nn.init.normal_(vlln.bias)
    hidden = torch.randn(1, rt.SEQUENCE, rt.LANGUAGE_CHANNELS)
    assert torch.allclose(graphs.OutputNorm(rms, vlln)(hidden), vlln(rms(hidden)), atol=5e-5)


def test_the_norm_rewrites_emit_no_reduction_operators(tmp_path):
    """The whole point of the matmul form: Modalix cannot place the reductions."""
    onnx = pytest.importorskip("onnx")
    runtime = pytest.importorskip("onnxruntime")

    layer_norm = nn.LayerNorm(rt.VISION_CHANNELS)
    hidden = torch.randn(1, rt.VISION_WIDTH, rt.VISION_CHANNELS)
    path = tmp_path / "eagle_vision_post_norm.onnx"
    expected = graphs._export_onnx(graphs.VisionPostNorm(layer_norm), (hidden,), path,
                                   ("hidden",), ("output",))
    onnx.checker.check_model(str(path))
    operators = {node.op_type for node in onnx.load(str(path)).graph.node}
    assert not operators & {"ReduceMean", "ReduceSumSquare", "LayerNormalization"}
    session = runtime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert np.allclose(session.run(None, {"hidden": hidden.numpy()})[0], expected, atol=1e-5)


def test_additive_masks_are_complementary_over_the_image_window():
    image = torch.zeros(1, rt.SEQUENCE, dtype=torch.bool)
    window = slice(rt.IMAGE_TOKEN_START, rt.IMAGE_TOKEN_START + rt.CONNECTOR_WIDTH)
    image[0, window] = True
    attention = torch.ones(1, rt.SEQUENCE, dtype=torch.bool)
    image_additive, text_additive = graphs.additive_masks(image, attention)
    assert bool((image_additive[0, window] == 0).all())
    assert bool((image_additive[0, :rt.IMAGE_TOKEN_START] == -10000).all())
    # Every position is open in exactly one of the two.
    assert bool(((image_additive == 0) ^ (text_additive == 0)).all())


#: SO-101 joint widths, as a GR00T checkpoint groups them.
ARM, GRIPPER = 5, 1


def _so101_checkpoint(root: Path, horizon: int = 4) -> Path:
    """The metadata files of a GR00T SO-101 checkpoint, in the layout GR00T saves.

    Shapes follow a real SO-101 checkpoint: absolute state/action statistics per
    group, and per-step `relative_action` statistics for the relative arm group.
    The weights are not needed for normalization or validation.
    """
    checkpoint = root / "ckpt"
    checkpoint.mkdir(parents=True)

    def write(name, value):
        (checkpoint / name).write_text(json.dumps(value), encoding="utf-8")

    def block(low, high):
        low, high = np.asarray(low, float), np.asarray(high, float)
        return {"min": low.tolist(), "max": high.tolist(),
                "mean": ((low + high) / 2).tolist(), "std": (high - low).tolist()}

    write("config.json", {"action_horizon": 50, "model_name": "nvidia/Eagle-Block2A-2B-v2"})
    write("embodiment_id.json", {"new_embodiment": 10, "gr1": 20})
    write("eagle_preprocessor_config.json", {
        "do_convert_rgb": True, "do_resize": False, "do_rescale": True, "do_normalize": True,
        "rescale_factor": 1 / 255, "image_mean": [0.5] * 3, "image_std": [0.5] * 3,
    })
    group = {"modality_keys": ["single_arm", "gripper"], "delta_indices": [0]}
    write("processor_config.json", {"processor_kwargs": {
        "model_name": "nvidia/Eagle-Block2A-2B-v2",
        "use_percentiles": False,
        "use_relative_action": True,
        "modality_configs": {"new_embodiment": {
            "video": {"modality_keys": ["front", "wrist"], "delta_indices": [0]},
            "state": group,
            "action": {**group, "delta_indices": list(range(horizon)),
                       "action_configs": [{"rep": "RELATIVE"}, {"rep": "ABSOLUTE"}]},
        }},
    }})
    # Arm joint 4 never moved: min == max, so its half-range is zero.
    arm_low, arm_high = [-10.0, -20.0, -30.0, 5.0, -50.0], [10.0, 20.0, 30.0, 5.0, 50.0]
    steps = np.arange(1, horizon + 1, dtype=float)[:, None] * np.ones(ARM)
    write("statistics.json", {"new_embodiment": {
        "state": {"single_arm": block(arm_low, arm_high), "gripper": block([0.0], [40.0])},
        "action": {"single_arm": block(arm_low, arm_high), "gripper": block([2.0], [42.0])},
        "relative_action": {"single_arm": block(-steps, steps)},
    }})
    return checkpoint


def test_normalization_trims_to_real_joints_and_survives_a_still_joint(tmp_path):
    """GR00T maps each group's [min, max] to [-1, 1]; PoLiMa stores that as
    mean = centre and std = half-range, for the real joints only.

    A joint that never moved has a zero range. Dividing by it on the board
    produces inf with no traceback to read, so it is clamped to 1 at export.
    """
    horizon = 4
    checkpoint = _so101_checkpoint(tmp_path, horizon)

    loaded = np.load(graphs.write_normalization(checkpoint, [], tmp_path / "stats.npz"))
    assert loaded["state_mean"].tolist() == [0.0, 0.0, 0.0, 5.0, 0.0, 20.0]
    assert loaded["state_std"].tolist() == [10.0, 20.0, 30.0, 1.0, 50.0, 20.0]
    # Relative arm actions use the per-step relative statistics; the absolute
    # gripper uses the ordinary action block, repeated for every step.
    action_mean = loaded["action_mean"].reshape(horizon, ARM + GRIPPER)
    action_std = loaded["action_std"].reshape(horizon, ARM + GRIPPER)
    assert np.allclose(action_mean[:, :ARM], 0.0)
    assert np.allclose(action_std[:, :ARM], np.arange(1, horizon + 1)[:, None])
    assert np.allclose(action_mean[:, ARM], 22.0)
    assert np.allclose(action_std[:, ARM], 20.0)
    # Written under the names runtime.SIDECARS lists, so packing needs no rename.
    written = {path.name for path in (tmp_path / "constants").iterdir()}
    assert written == {"state_mean", "state_std", "action_mean", "action_std"}
    assert np.fromfile(tmp_path / "constants" / "state_std", dtype="<f4").tolist() == [
        10.0, 20.0, 30.0, 1.0, 50.0, 20.0]


def test_validator_flags_a_checkpoint_with_no_statistics(tmp_path):
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "config.json").write_text("{}", encoding="utf-8")
    problems = graphs.validate_checkpoint(bare)
    assert len(problems) == 3
    for name in ("processor_config.json", "statistics.json", "embodiment_id.json"):
        assert any(name in problem for problem in problems), name

    checkpoint = _so101_checkpoint(tmp_path)
    assert graphs.validate_checkpoint(checkpoint) == []
    (checkpoint / "statistics.json").unlink()
    problems = graphs.validate_checkpoint(checkpoint)
    assert len(problems) == 1 and "statistics.json" in problems[0]
