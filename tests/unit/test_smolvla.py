"""SmolVLA's spec and execution plan, verified against the legacy goldens.

SmolVLA is the first policy whose runtime is not a straight feed-forward chain:
it integrates a learned velocity field over ten Euler steps, so two of its four
graphs run ten times and the plan carries a time embedding, a strided scatter and
an integration update that ACT never exercised.

That machinery is verified the same way ACT's was -- by replaying the plan with
the MLA stubbed by recorded per-stage outputs. If the host arithmetic reproduces
the recorded pipeline, a wrong result on hardware is the ELF's doing, not the
plan's.

The golden tree is a legacy build, so these tests skip cleanly without it.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from polima.bundle.pack import _plan_dict
from polima.policies.registry import get_policy
from polima.policies.smolvla import runtime as rt
from polima.wire.server_stub import StubPlan

LEGACY = (Path(__file__).resolve().parents[3] / "SmolVLA" / "outputs"
          / "modalix_gewb_smolvla_045000_v2")
CONSTANTS = LEGACY / "final_som_compile" / "constants"
SAMPLE = LEGACY / "reference" / "sample_000000.npz"

needs_goldens = pytest.mark.skipif(
    not (CONSTANTS / "reference_prefix_embeddings.f32").is_file() or not SAMPLE.is_file(),
    reason="legacy SmolVLA build tree not present",
)


def _f32(name: str) -> np.ndarray:
    return np.fromfile(CONSTANTS / name, dtype="<f4")


@pytest.fixture(scope="module")
def sidecars(tmp_path_factory) -> Path:
    """Stage the sidecars under the names the plan uses."""
    target = tmp_path_factory.mktemp("smolvla_constants")
    for source, name in (
        ("empty_image_embedding.f32", "empty_image_embedding"),
        ("language_embedding.f32", "language_embedding"),
        ("state_project_weight.f32", "state_project_weight"),
        ("state_project_bias.f32", "state_project_bias"),
    ):
        (target / name).write_bytes((CONSTANTS / source).read_bytes())
    stats = _f32("normalization_stats.f32")
    for name, values in (("state_mean", stats[0:6]), ("state_std", stats[6:12]),
                         ("action_mean", stats[12:18]), ("action_std", stats[18:24])):
        values.astype("<f4").tofile(target / name)
    return target


def _plan(constants: Path) -> StubPlan:
    spec = get_policy("smolvla")
    data = _plan_dict(spec.build_runtime_plan(), wire=spec)
    return StubPlan(buffers=data["buffers"], steps=data["steps"], result=data["result"],
                    wire=data["wire"], constants_dir=constants)


# ------------------------------------------------------------------ the spec


def test_spec_registers_with_four_graphs():
    spec = get_policy("smolvla")
    assert [g.name for g in spec.compile.graphs] == ["vision", "prefix", "suffix", "denoise"]


def test_vision_is_the_only_nchw_graph():
    """It is the only one taking a raw image; everything downstream is tokens."""
    spec = get_policy("smolvla")
    assert [g.name for g in spec.compile.graphs if g.layout == "NCHW"] == ["vision"]


def test_vision_is_compiled_by_afe_like_the_rest():
    """LLiMa produces the vision ONNX; it does not compile it.
    compile_deploy_smolvla_som.sh runs the same afe wrapper over this graph as
    over the others. The `_llima_` in the legacy directory name is about the
    ONNX's origin, and it misled this spec once."""
    vision = get_policy("smolvla").compile.graph("vision")
    assert vision.compiler == "afe"
    # Plain HWC, not tessellated: measured on Modalix 2026-08-20, the HWC16
    # output contract scored cosine 0.978 against 0.9996 for this one, for a
    # 2 ms saving on a 300 ms chunk.
    assert not vision.mla_tessellation
    assert vision.external_dram_layout == "HWC"
    assert vision.promote_rank3_hwc
    assert vision.precision == "bf16"
    assert vision.activation_precision is None
    assert vision.weight_precision is None
    assert vision.elf_from == "retained"
    assert vision.exit_on_stable_elf
    assert vision.llima_args == ("--no-simplify",)
    assert vision.layout == "NCHW"


def test_compiled_output_layouts_match_modalix_downloads():
    """Every SmolVLA output downloads as plain contiguous BF16.

    Measured on Modalix (2026-09-10): decoding the bare denoise ELF's output
    as HWC16 byte planes produced non-finite values, while the same words read
    as plain BF16 were finite and in range. The SO-101 SmolVLA bundle that ran
    on the arm declares denoise plain.
    """
    graphs = {graph.name: graph for graph in get_policy("smolvla").compile.graphs}
    assert (
        graphs["vision"].outputs[0].dram_layout,
        graphs["vision"].outputs[0].logical_width,
        graphs["vision"].outputs[0].logical_channels,
    ) == ("plain", None, None)
    assert graphs["suffix"].outputs[0].dram_layout == "plain"
    assert graphs["prefix"].outputs[0].dram_layout == "plain"
    assert graphs["prefix"].external_dram_layout == "HWC"
    assert not graphs["prefix"].mla_tessellation
    assert (
        graphs["denoise"].outputs[0].dram_layout,
        graphs["denoise"].outputs[0].logical_width,
        graphs["denoise"].outputs[0].logical_channels,
    ) == ("plain", None, None)
    assert graphs["denoise"].outputs[0].shape == (1, 50, 32)


def test_action_graphs_request_smolvla_shape_inference():
    graphs = {graph.name: graph for graph in get_policy("smolvla").compile.graphs}
    assert "--infer-shapes" in graphs["prefix"].llima_args
    assert graphs["suffix"].llima_args == ("--infer-shapes",)
    assert "--infer-shapes" in graphs["denoise"].llima_args


def test_rmsnorm_graphs_skip_the_simplifier():
    """onnxsim drops RMSNorm's Mul(weight) and breaks ModelSDK's RMSNorm fusion."""
    graphs = {graph.name: graph for graph in get_policy("smolvla").compile.graphs}
    assert "--no-simplify" in graphs["prefix"].llima_args
    assert "--no-simplify" in graphs["denoise"].llima_args


def test_fusible_rms_matches_llama_rmsnorm():
    torch = pytest.importorskip("torch")
    from polima.policies.smolvla.graphs import fusible_rms

    norm = torch.nn.Module()
    norm.weight = torch.nn.Parameter(torch.randn(960))
    norm.variance_epsilon = 1e-5
    x = torch.randn(1, 241, 960) * 40.0
    reference = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * norm.weight
    torch.testing.assert_close(fusible_rms(norm, x), reference, rtol=1e-5, atol=1e-5)


def test_fusible_rms_exports_the_fusible_pattern(tmp_path):
    torch = pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    from polima.policies.smolvla.graphs import fusible_rms

    class Norm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(8))
            self.variance_epsilon = 1e-6

        def forward(self, x):
            return fusible_rms(self, x)

    path = tmp_path / "rms.onnx"
    torch.onnx.export(Norm(), torch.randn(1, 4, 8), str(path), opset_version=17)
    ops = [node.op_type for node in onnx.load(str(path)).graph.node]
    assert "ReduceMean" in ops and "Sqrt" in ops and "Div" in ops
    assert "Reciprocal" not in ops and "Pow" not in ops


def test_language_conditioned_datasets_may_hold_several_tasks():
    """Unlike ACT, a multi-task dataset is legitimate here."""
    assert get_policy("smolvla").dataset.single_task is False


def test_wire_magic_is_smol():
    assert rt.WIRE_MAGIC.to_bytes(4, "little") == b"LOMS"      # "SMOL" little-endian
    assert get_policy("smolvla").wire.default_port == 8081


def test_geometry_matches_the_legacy_server():
    """Every one of these is a literal in smolvla_som_server.cpp."""
    assert rt.IMAGE_ELEMENTS == 512 * 512 * 3
    assert rt.PREFIX_ELEMENTS == 241 * 960
    assert rt.CACHE_ELEMENTS == 2467840
    assert rt.SUFFIX_IN_ELEMENTS == 50 * 752
    assert rt.SUFFIX_OUT_ELEMENTS == 50 * 720
    assert rt.DENOISE_IN_ELEMENTS == 1223 * 2048
    assert rt.DENOISE_SUFFIX_OFFSET == 1205 * 2048
    assert rt.NOISE_ELEMENTS == 50 * 32
    assert rt.RESPONSE_ELEMENTS == 300


def test_time_embedding_periods_span_min_to_max():
    """period = 0.004 * 1000^fraction, so the top of the sweep is exactly 4.0."""
    assert rt.TIME_MIN_PERIOD == 0.004
    assert rt.TIME_MAX_PERIOD == pytest.approx(rt.TIME_MIN_PERIOD * 1000.0)


# ------------------------------------------------------------------ the plan


def test_plan_runs_each_graph_the_right_number_of_times():
    """Two cameras, one prefix, and ten flow-matching steps -- 22 MLA calls."""
    steps = get_policy("smolvla").build_runtime_plan().steps
    calls: dict[str, int] = {}
    for step in steps:
        if step.op == "run_elf":
            calls[step.args["graph"]] = calls.get(step.args["graph"], 0) + 1
    assert calls == {"vision": 2, "prefix": 1, "suffix": 10, "denoise": 10}


def test_denoise_loop_is_unrolled_not_looped():
    """The interpreter has no control flow; ten iterations are ten step groups."""
    steps = get_policy("smolvla").build_runtime_plan().steps
    assert sum(1 for s in steps if s.op == "euler") == rt.DENOISE_STEPS
    assert sum(1 for s in steps if s.op == "sincos_time") == rt.DENOISE_STEPS


def test_timesteps_descend_from_one():
    times = [s.args["scalar"] for s in get_policy("smolvla").build_runtime_plan().steps
             if s.op == "sincos_time"]
    assert times[0] == pytest.approx(1.0)
    assert times[-1] == pytest.approx(0.1)
    assert times == sorted(times, reverse=True)


def test_state_is_normalized_before_padding_not_after():
    """Normalizing the 32-wide lane directly would turn the 26 structural zeros
    into -mean/std. The plan normalizes 6, then packs into the lane."""
    plan = get_policy("smolvla").build_runtime_plan()
    normalize = next(s for s in plan.steps if s.op == "normalize")
    assert plan.buffers[normalize.out] == rt.STATE_DIM
    pack = next(s for s in plan.steps if s.op == "pack" and s.out == "state_lane")
    assert plan.buffers[pack.out] == rt.ACTION_LANE


def test_prefix_sections_are_contiguous_and_cover_241_tokens():
    plan = get_policy("smolvla").build_runtime_plan()
    pack = next(s for s in plan.steps if s.out == "prefix_embeddings")
    parts = sorted(pack.args["parts"], key=lambda p: p["dst_offset"])
    assert [p["dst_offset"] // rt.HIDDEN for p in parts] == [0, 64, 128, 192, 240]
    assert sum(p["count"] for p in parts) == rt.PREFIX_ELEMENTS


# --------------------------------------------------- replay against goldens


@needs_goldens
def test_plan_replay_reproduces_the_recorded_actions(sidecars):
    """The whole host-side chain, with the MLA replaced by recorded outputs.

    Covers normalize, matvec, pack (buffer and sidecar parts), scale, the
    sincos time embedding, the strided scatter that builds the suffix input,
    ten Euler updates, the 32->6 unpad and denormalize.
    """
    npz = np.load(SAMPLE)
    gold_vision = _f32("reference_vision_embedding.f32")
    gold_cache = _f32("reference_packed_cache.f32")
    gold_suffix = _f32("reference_suffix_calibration_10.f32")[:rt.SUFFIX_OUT_ELEMENTS]
    velocities = [npz[f"velocity_{i:02d}"].astype(np.float32).ravel() for i in range(10)]

    seen: dict[str, list[np.ndarray]] = {}

    def graph_fn(name, value):
        seen.setdefault(name, []).append(value.copy())
        if name == "vision":
            return gold_vision
        if name == "prefix":
            return gold_cache
        if name == "suffix":
            return gold_suffix
        return velocities[len(seen["denoise"]) - 1]

    actions = _plan(sidecars).run(
        {
            "image0": npz["prepared_image_0"].astype(np.float32).ravel(),
            "image1": npz["prepared_image_1"].astype(np.float32).ravel(),
            "state": npz["raw_state"].astype(np.float32).ravel(),
            "noise": npz["noise"].astype(np.float32).ravel(),
        },
        graph_fn=graph_fn,
    )

    assert {k: len(v) for k, v in seen.items()} == {
        "vision": 2, "prefix": 1, "suffix": 10, "denoise": 10
    }
    # The recorded actions come from the same arithmetic, so this is exact.
    np.testing.assert_array_equal(actions, npz["actions"].astype(np.float32).ravel())


@needs_goldens
def test_prefix_pack_matches_the_recorded_embeddings(sidecars):
    """The fixed sections are exact; the vision section carries the bf16
    rounding of the stored golden, which is why it gets a tolerance."""
    npz = np.load(SAMPLE)
    gold_vision = _f32("reference_vision_embedding.f32")
    gold_prefix = _f32("reference_prefix_embeddings.f32")
    gold_cache = _f32("reference_packed_cache.f32")
    gold_suffix = _f32("reference_suffix_calibration_10.f32")[:rt.SUFFIX_OUT_ELEMENTS]
    captured: list[np.ndarray] = []

    def graph_fn(name, value):
        if name == "vision":
            return gold_vision
        if name == "prefix":
            captured.append(value.copy())
            return gold_cache
        if name == "suffix":
            return gold_suffix
        return np.zeros(rt.NOISE_ELEMENTS, dtype=np.float32)

    _plan(sidecars).run(
        {
            "image0": npz["prepared_image_0"].astype(np.float32).ravel(),
            "image1": npz["prepared_image_1"].astype(np.float32).ravel(),
            "state": npz["raw_state"].astype(np.float32).ravel(),
            "noise": npz["noise"].astype(np.float32).ravel(),
        },
        graph_fn=graph_fn,
    )

    prefix = captured[0]
    scaled = rt.VISION_ELEMENTS
    assert np.abs(prefix[:scaled] - gold_prefix[:scaled]).max() < 2e-4
    # empty-image, language and state sections come from sidecars and arithmetic
    assert np.abs(prefix[2 * scaled:] - gold_prefix[2 * scaled:]).max() < 1e-6


@needs_goldens
def test_image_tokens_are_scaled_by_sqrt_hidden(sidecars):
    """SmolVLM scales image embeddings by sqrt(hidden) before the language
    model. Dropping it does not error -- it just puts vision tokens at the wrong
    magnitude relative to language ones."""
    gold_vision = _f32("reference_vision_embedding.f32")
    gold_prefix = _f32("reference_prefix_embeddings.f32")
    ratio = gold_prefix[:rt.VISION_ELEMENTS] / np.where(gold_vision == 0, np.nan, gold_vision)
    assert np.nanmedian(ratio) == pytest.approx(rt.IMAGE_SCALE, rel=1e-3)


@pytest.mark.parametrize("dtype_name", ["bfloat16", "float32"])
def test_vision_position_ids_match_transformers_in_each_dtype(dtype_name):
    """Transformers buckets patch coordinates in the pixel dtype; BF16 and FP32 differ."""
    torch = pytest.importorskip("torch")
    smolvlm = pytest.importorskip("transformers.models.smolvlm.modeling_smolvlm")
    from transformers.models.smolvlm.configuration_smolvlm import SmolVLMVisionConfig

    from polima.policies.smolvla.graphs import vision_position_ids

    dtype = getattr(torch, dtype_name)
    config = SmolVLMVisionConfig(hidden_size=8, image_size=512, patch_size=16, num_channels=3)
    embeddings = smolvlm.SmolVLMVisionEmbeddings(config).to(dtype)
    seen = {}
    def capture(module, args, output):
        seen["ids"] = args[0]

    embeddings.position_embedding.register_forward_hook(capture)
    pixels = torch.zeros(1, 3, 512, 512, dtype=dtype)
    embeddings(pixels, torch.ones(1, 32, 32, dtype=torch.bool))
    ours = vision_position_ids(32, embeddings.num_patches_per_side, dtype)
    assert torch.equal(ours, seen["ids"])
    if dtype == torch.bfloat16:  # the grid SmolVLA was trained on: no shift
        assert torch.equal(ours[0], torch.arange(1024))


def _host_plan_dict():
    spec = get_policy("smolvla")
    return _plan_dict(spec.build_runtime_plan(), spec)


def test_device_plan_runs_prefix_and_ten_steps_in_one_program():
    from polima.policies.smolvla import device

    plan = device.device_plan(_host_plan_dict(), ("cache", "x", "tbias"))
    chains = [s for s in plan["steps"] if s["op"] == "run_device_chain"]
    assert len(chains) == 1
    assert not [s for s in plan["steps"] if s["op"] in ("euler", "sincos_time")
                or s["args"].get("graph") in ("prefix", "suffix", "denoise")]
    calls = chains[0]["args"]["calls"]
    assert [c["graph"] for c in calls] == ["prefix"] + [device.STEP_GRAPH] * rt.DENOISE_STEPS
    assert calls[0]["out"] == [["cache", 0, rt.CACHE_ELEMENTS]]
    for k, call in enumerate(calls[1:]):
        current, following = ("x_a", "x_b") if k % 2 == 0 else ("x_b", "x_a")
        # compiled order: cache, x, tbias
        assert [span[0] for span in call["in"]] == ["cache", current, device.tbias_sidecar(k)]
        assert call["out"] == [[following, 0, device.X_ELEMENTS]]
    assert chains[0]["args"]["downloads"] == [{"src": "x_a", "count": device.X_ELEMENTS, "dst": "x"}]
    cached = {u["dst"] for u in chains[0]["args"]["uploads"] if u.get("cache")}
    assert cached == {device.tbias_sidecar(k) for k in range(rt.DENOISE_STEPS)}
    assert set(plan["sidecars"]) >= cached
    # the unpad + denormalize tail still reads x
    tail = plan["steps"][plan["steps"].index(chains[0]) + 1:]
    assert tail[0]["op"] == "gather_strided" and tail[0]["args"]["src"] == "x"


def test_device_plan_refuses_a_rewritten_plan():
    from polima.policies.smolvla import device

    plan = device.device_plan(_host_plan_dict())
    with pytest.raises(ValueError):
        device.device_plan(plan)


def test_split_vision_runs_llima_tower_then_connector_per_camera():
    from polima.policies.smolvla import device

    plan = device.split_vision(_host_plan_dict())
    graphs = [(s["args"]["graph"], s["args"]["in"][0], s["out"]) for s in plan["steps"] if s["op"] == "run_elf"
              and s["args"]["graph"] in ("vision", "connector")]
    assert graphs == [("vision", "image0", "tokens0"), ("connector", "tokens0", "vision0_scaled"),
                      ("vision", "image1", "tokens1"), ("connector", "tokens1", "vision1_scaled")]
    assert not [s for s in plan["steps"] if s["op"] == "scale"]  # the image scale is in the connector
    assert plan["buffers"]["tokens0"] == device.VISION_TOKENS


def test_optimized_plan_is_one_device_program_between_state_and_unpad():
    from polima.policies.smolvla import device

    plan = device.optimized_plan(_host_plan_dict(), ("x", "tbias", "cache"))
    assert [s["op"] for s in plan["steps"]] == [
        "normalize", "pack", "matvec", "run_device_chain", "gather_strided", "denormalize"]
    chain = plan["steps"][3]["args"]
    assert [c["graph"] for c in chain["calls"]] == (
        ["vision", "connector"] * 2 + ["prefix"] + [device.STEP_GRAPH] * rt.DENOISE_STEPS)
    # the connectors write the two camera blocks of the prefix input; the rest is uploaded
    assert [c["out"] for c in chain["calls"][1:4:2]] == [[["prefix_in", 0, rt.VISION_ELEMENTS]],
                                                         [["prefix_in", rt.VISION_ELEMENTS, rt.VISION_ELEMENTS]]]
    prefix_uploads = sorted((u["dst_offset"], u["count"], u.get("cache", False))
                            for u in chain["uploads"] if u["dst"] == "prefix_in")
    assert prefix_uploads == [(2 * rt.VISION_ELEMENTS, rt.VISION_ELEMENTS, True),
                              (3 * rt.VISION_ELEMENTS, rt.LANGUAGE_ELEMENTS, True),
                              (rt.STATE_TOKEN_INDEX * rt.HIDDEN, rt.HIDDEN, False)]
    # every prefix row is written exactly once
    spans = sorted([(c["out"][0][1], c["out"][0][2]) for c in chain["calls"] if c["graph"] == "connector"]
                   + [(o, n) for o, n, _ in prefix_uploads])
    assert sum(n for _, n in spans) == rt.PREFIX_ELEMENTS
    assert all(a + n == b for (a, n), (b, _) in itertools.pairwise(spans))


def test_cli_routes_smolvla_builds_but_not_the_build_dir_paths():
    from polima.cli.compile import _staged_build

    assert _staged_build(["--policy", "smolvla", "--checkpoint", "c", "--dataset", "d", "--output", "o"], "smolvla")
    assert not _staged_build(["--policy", "smolvla", "--build-dir", "b"], "smolvla")
    assert not _staged_build(["--policy", "act", "--checkpoint", "c"], "smolvla")


def test_staged_builder_compiles_the_prefix_with_the_spec_flags(tmp_path):
    from polima.compile import smolvla as builder

    (tmp_path / "build.json").write_text('{"checkpoint": "c", "dataset": "d"}')
    jobs = {job["name"]: job for job in builder.graph_jobs(builder.Build(tmp_path))}
    assert set(jobs) == {"prefix", "connector", "denoise_step"}
    assert {"--no-simplify", "--infer-shapes"} <= set(jobs["prefix"]["flags"])
    assert jobs["denoise_step"]["persistent"] and jobs["denoise_step"]["weights"] == "int8"
    assert jobs["connector"]["weights"] == "bf16" and jobs["connector"]["layout"] == "NCHW"


def test_vision_connector_conv_equals_pixel_shuffle_and_projection():
    torch = pytest.importorskip("torch")
    smolvlm = pytest.importorskip("transformers.models.smolvlm.modeling_smolvlm")
    from transformers.models.smolvlm.configuration_smolvlm import SmolVLMConfig

    from polima.policies.smolvla import graphs

    config = SmolVLMConfig(vision_config={"hidden_size": 12}, text_config={"hidden_size": 10}, scale_factor=2)
    connector = smolvlm.SmolVLMConnector(config).eval()
    model = type("M", (), {})()
    model.vlm_with_expert = type("V", (), {"get_vlm_model": lambda self: type("L", (), {"connector": connector})()})()
    conv = graphs.VisionConnector(model).eval()
    tokens = torch.randn(1, 64, 12)  # an 8 x 8 grid
    with torch.no_grad():
        expected = connector(tokens) * graphs.IMAGE_SCALE
        got = conv.rows(conv(conv.grid(tokens)))
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-4)
