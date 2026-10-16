"""Pi0.5 policy: spec, runtime plan, client request building and the static graphs."""
import json
from pathlib import Path

import numpy as np
import pytest

from polima.policies.pi05 import client
from polima.policies.pi05 import runtime as rt
from polima.policies.pi05.profile import Geometry, build_spec, geometry_from_config
from polima.policies.registry import get_policy

ONNX = Geometry(**{**Geometry.base().__dict__, "backend": "onnx"})


def test_registered_spec_is_pi05_base_on_llima():
    spec = get_policy("pi05")
    names = spec.compile.names
    assert names[0] == "vision" and names[-1] == "expert" and "attn" in names
    assert [n for n in names if n.startswith("pre_")] == [rt.llima_pre(i) for i in range(18)]
    assert [n for n in names if n.startswith("post_")] == [rt.llima_post(i) for i in range(17)]
    assert spec.wire.magic_ascii == "PI05" and spec.wire.default_port == 8094
    assert spec.wire.response_shape == (50, 32)
    assert spec.runtime_context["backend"] == "llima" and spec.runtime_context["prefix_tokens"] == 1024
    shapes = {t.name: t.shape for t in spec.wire.request_tensors}
    assert shapes == {"image0": (224, 224, 3), "image1": (224, 224, 3), "image2": (224, 224, 3),
                      "language": (256, 2048), "prefix_bias": (1024,), "suffix_rope": (100, 256), "noise": (50, 32)}


def test_onnx_backend_spec_keeps_the_native_prefix():
    spec = build_spec(ONNX)
    assert spec.compile.names == ("vision", *rt.block_names(), "expert")
    assert spec.runtime_context["prefix_tokens"] == 968
    assert [t.shape for t in spec.wire.request_tensors][:4] == [(256, 588)] * 3 + [(200, 2048)]


def test_llima_plan_keeps_the_prefix_and_cache_on_the_device():
    plan = get_policy("pi05").build_runtime_plan()
    chains = [s for s in plan.steps if s.op == "run_device_chain"]
    assert len(chains) == 11  # one prefill, ten expert steps
    prefill = chains[0].args
    graphs = [c["graph"] for c in prefill["calls"]]
    assert graphs[:3] == ["vision"] * 3 and graphs.count("attn") == 17
    assert graphs[3:6] == ["pre_00", "attn", "post_00"] and graphs[-1] == "pre_17"
    assert "downloads" not in prefill  # nothing of the prefill returns to the host
    layout = rt.expert_layout(1024, rt.EXPERT_INPUTS)
    kv = 1024 * 256
    pre = [c for c in prefill["calls"] if c["graph"].startswith("pre_")]
    for layer, call in enumerate(pre):
        assert call["out"][1] == ["expert_in", layout["cache"][0] + 2 * layer * kv, kv]
        assert call["out"][2] == ["expert_in", layout["cache"][0] + (2 * layer + 1) * kv, kv]
        assert call["in"][0][0] == ("hidden_a", "hidden_b")[layer % 2]
    for k, chain in enumerate(chains[1:]):
        uploads = {u["dst_offset"]: u for u in chain.args["uploads"]}
        assert uploads[layout["cond"][0]]["src"] == f"cond_{k:02d}"
        assert uploads[layout["x"][0]]["src"] == "x" and not uploads[layout["x"][0]]["cache"]
        assert uploads[layout["rope"][0]]["cache"] and uploads[layout["bias"][0]]["cache"]
        assert chain.args["calls"][0]["in"] == [["expert_in", 0, sum(c for _, c in layout.values())]]
    mask = next(s for s in plan.steps if s.out == "mask")
    assert mask.op == "gather_strided" and mask.args["stride"] == 0 and mask.args["count"] == 1024
    assert len([s for s in plan.steps if s.op == "euler"]) == 10
    assert set(plan.sidecars) >= {"freq_real", "freq_imag", "cond_00"}
    from polima.bundle.pack import _plan_dict

    data = _plan_dict(plan)
    assert data["device_buffers"]["expert_in"] == sum(c for _, c in layout.values())
    for chain in chains:  # every span a chain binds is a declared device buffer
        for call in chain.args["calls"]:
            for buffer, offset, count in (*call["in"], *call["out"]):
                assert offset + count <= data["device_buffers"][buffer]


def test_expert_layout_follows_the_compiled_input_order():
    layout = rt.expert_layout(1024, ("bias", "cond", "x", "rope", "cache"))
    assert layout["bias"] == (0, 1074) and layout["cond"] == (1074, 1024)
    assert layout["cache"][0] == 1074 + 1024 + 1600 + 25600
    assert all(offset * 2 % 64 == 0 for name, (offset, _) in rt.expert_layout(1024, rt.EXPERT_INPUTS).items()
               if name == "cache")


def test_plan_runs_prefix_once_and_expert_ten_times():
    plan = build_spec(ONNX).build_runtime_plan()
    graphs = [s.args["graph"] for s in plan.steps if s.op == "run_elf"]
    assert graphs.count("vision") == 3 and graphs.count("expert") == 10
    assert all(graphs.count(name) == 1 for name in rt.block_names())
    eulers = [s for s in plan.steps if s.op == "euler"]
    assert len(eulers) == 10 and all(s.args["scalar"] == pytest.approx(0.1) for s in eulers)
    # The cache gathers every block's K/V, after its hidden state.
    cache = next(s for s in plan.steps if s.out == "cache")
    p = 968
    assert [part["src_offset"] for part in cache.args["parts"]] == [p * 2048] * 9
    assert plan.buffers["cache"] == 36 * p * 256
    assert plan.result == "x"


def test_six_joint_checkpoint_returns_six_actions(tmp_path):
    config = {"type": "pi05", "chunk_size": 50, "num_inference_steps": 10, "image_resolution": [224, 224],
              "tokenizer_max_length": 200, "use_relative_actions": True,
              "input_features": {"observation.state": {"type": "STATE", "shape": [6]},
                                 "observation.images.wrist_cam": {"type": "VISUAL", "shape": [3, 480, 640]},
                                 "observation.images.top_cam": {"type": "VISUAL", "shape": [3, 480, 640]}},
              "output_features": {"action": {"type": "ACTION", "shape": [6]}}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert geometry_from_config(tmp_path).prefix_tokens == 768  # LLiMa: 712 tokens in a 768-token group
    assert geometry_from_config(tmp_path).language_rows == 256
    geometry = geometry_from_config(tmp_path, backend="onnx")
    assert geometry.prefix_tokens == 712 and geometry.calibration_id == "so-arm101"
    spec = build_spec(geometry)
    plan = spec.build_runtime_plan()
    assert spec.wire.response_shape == (50, 6) and plan.result == "actions"
    gather = plan.steps[-1]
    assert gather.op == "gather_strided" and gather.args["take"] == 6 and gather.args["stride"] == 32


def test_unsupported_geometry_is_refused(tmp_path):
    config = {"type": "pi05", "chunk_size": 16, "input_features": {}, "output_features": {}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="chunk_size 50"):
        geometry_from_config(tmp_path)


def test_expert_inputs_follow_the_compiled_order():
    spec = build_spec(ONNX)
    from dataclasses import replace

    reordered = replace(spec, runtime_context={**spec.runtime_context,
                                               "expert_inputs": ["bias", "cond", "x", "rope", "cache"]})
    step = next(s for s in reordered.build_runtime_plan().steps if s.args.get("graph") == "expert")
    assert step.args["in"] == ["suffix_bias", "cond_00", "x", "suffix_rope", "cache"]


def test_prompt_matches_lerobot_discretization():
    state = np.array([-1.0, -0.999, 0.0, 0.5, 0.999, 1.0, 1.5, -2.0], np.float32)
    # np.digitize over linspace(-1, 1, 257)[:-1], minus one: -1 -> 0, 0 -> 128, 1 -> 255,
    # out-of-range values are not clipped below 0 or above 255 by the bins.
    assert client.prompt(" pick_up\nthe cup ", state) == (
        "Task: pick up the cup, State: 0 0 128 192 255 255 255 -1;\nAction: ")


def test_resize_with_pad_matches_lerobot_geometry():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    image = np.random.default_rng(0).random((480, 640, 3)).astype(np.float32)
    ours = client.resize_with_pad(image)
    x = torch.from_numpy(image).permute(2, 0, 1)[None]
    resized = F.interpolate(x, size=(168, 224), mode="bilinear", align_corners=False).clamp(0, 1)
    expected = F.pad(resized, (0, 0, 28, 28), value=0.0)[0] * 2 - 1
    assert ours.shape == (3, 224, 224)
    np.testing.assert_allclose(ours, expected.numpy(), atol=2e-4)  # float64 weights vs torch float32


def test_client_patches_match_the_vision_graph():
    torch = pytest.importorskip("torch")
    from polima.policies.pi05 import modules

    chw = np.random.default_rng(1).standard_normal((3, 224, 224)).astype(np.float32)
    np.testing.assert_array_equal(client.patchify(chw),
                                  modules.patchify(torch.from_numpy(chw)[None])[0].numpy())


def test_rope_rows_match_the_graph_tables():
    torch = pytest.importorskip("torch")
    from polima.policies.pi05 import modules

    cos, sin = modules.rope_tables(torch.arange(900, 950))
    rows = client.rope_rows(900)
    np.testing.assert_allclose(rows[:50], cos.numpy(), atol=1e-5)
    np.testing.assert_allclose(rows[50:], sin.numpy(), atol=1e-5)


def test_quantile_round_trip_and_identity_without_stats():
    stats = client.Stats(state_q01=np.zeros(2, np.float32), state_q99=np.array([2.0, 0.0], np.float32),
                         action_q01=np.array([-1.0, 0.0], np.float32), action_q99=np.array([3.0, 0.0], np.float32))
    np.testing.assert_allclose(stats.normalize_state(np.array([1.0, 0.0])), [0.0, -1.0])
    # A zero quantile span becomes 1e-8, as in LeRobot's normalizer.
    np.testing.assert_allclose(stats.unnormalize_actions(np.array([[0.0, 0.0]])), [[1.0, 0.0]], atol=1e-7)
    plain = client.Stats()
    np.testing.assert_array_equal(plain.normalize_state(np.array([0.25])), [0.25])


class FakeTokenizer:
    def encode(self, text):
        return type("E", (), {"ids": [2] + [ord(c) % 1000 + 3 for c in text]})()


def make_client(tmp_path, cameras=3, action_dim=32, relative=False):
    root = tmp_path / "pi05"
    root.mkdir()
    (root / "client.json").write_text(json.dumps({"cameras": cameras, "state_dim": 32, "action_dim": action_dim,
                                                  "relative_actions": relative}))
    table = np.arange(1100 * 2048, dtype=np.float32).reshape(1100, 2048) / 1e6
    table.astype(np.float16).tofile(root / "embedding.f16")
    return client.Pi05Client(root, tokenizer=FakeTokenizer()), table.astype(np.float16)


def test_request_has_the_wire_shapes_and_masks(tmp_path):
    pi05, table = make_client(tmp_path)
    images = [np.zeros((480, 640, 3), np.uint8)] * 3
    request = pi05.request(images, np.zeros(32), "pick", seed=3)
    spec = build_spec(ONNX)
    for tensor in spec.wire.request_tensors:
        assert request[tensor.name].shape == tensor.shape, tensor.name
    ids, valid = pi05.tokens(client.prompt("pick", np.zeros(32)))
    assert np.all(request["prefix_bias"][: 768 + valid] == 0)
    assert np.all(request["prefix_bias"][768 + valid:] == client.MASK)
    np.testing.assert_array_equal(request["language"][:valid], table[ids[:valid]].astype(np.float32))
    assert not request["language"][valid:].any()
    np.testing.assert_allclose(request["suffix_rope"], client.rope_rows(768 + valid))


def test_request_refuses_missing_cameras(tmp_path):
    pi05, _ = make_client(tmp_path)
    with pytest.raises(ValueError, match="all 3 cameras"):
        pi05.request([np.zeros((224, 224, 3), np.uint8)] * 2, np.zeros(32), "pick")


def test_relative_actions_add_state_except_gripper(tmp_path):
    pi05, _ = make_client(tmp_path, action_dim=2, relative=True)
    pi05.config["relative_mask"] = [True, False]
    out = pi05.actions(np.zeros((50, 2)), state=np.array([5.0, 7.0]))
    assert np.all(out[:, 0] == 5.0) and np.all(out[:, 1] == 0.0)


def test_static_modules_compose_like_the_plan():
    """Random tiny weights: the prefix block's packed output and the expert step
    have the shapes the plan slices, and padded keys do not change visible outputs."""
    torch = pytest.importorskip("torch")
    from polima.policies.pi05 import blocks

    torch.manual_seed(0)
    p, layers = 40, 2
    w = {}
    pre = "paligemma_with_expert.paligemma.model.language_model.layers"
    for i in range(layers):
        for name, shape in (("self_attn.q_proj", (2048, 2048)), ("self_attn.k_proj", (256, 2048)),
                            ("self_attn.v_proj", (256, 2048)), ("self_attn.o_proj", (2048, 2048)),
                            ("mlp.gate_proj", (64, 2048)), ("mlp.up_proj", (64, 2048)), ("mlp.down_proj", (2048, 64))):
            w[f"{pre}.{i}.{name}.weight"] = torch.randn(shape) * 0.02
        w[f"{pre}.{i}.input_layernorm.weight"] = torch.zeros(2048)
        w[f"{pre}.{i}.post_attention_layernorm.weight"] = torch.zeros(2048)
    block = blocks.PrefixBlock(w, 0, layers, p)
    hidden = torch.randn(1, 1, p, 2048)
    bias = torch.zeros(1, 1, 1, p)
    out = block(hidden, bias)
    assert out.shape == (1, 1, blocks.prefix_output_rows(p, layers), 256)
    bias_masked = bias.clone()
    bias_masked[..., 30:] = -1e4
    changed = hidden.clone()
    changed[..., 30:, :] = torch.randn(1, 1, p - 30, 2048)
    a = block(hidden, bias_masked).reshape(-1)[: p * 2048].reshape(p, 2048)[:30]
    b = block(changed, bias_masked).reshape(-1)[: p * 2048].reshape(p, 2048)[:30]
    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


def test_workbench_lists_and_compiles_a_pi05_checkpoint(tmp_path):
    from polima.workbench import app, jobs

    run = tmp_path / "pi05_so101" / "checkpoints" / "020000" / "pretrained_model"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"type": "pi05"}))
    (run / "model.safetensors").write_bytes(b"x")
    found = app.checkpoint(run)
    assert found["policy"] == "Pi0.5" and found["run"] == "pi05_so101" and found["step"] == 20000
    dataset = tmp_path / "data"
    dataset.mkdir()
    argv = jobs.compile_command(found, str(dataset), tmp_path / "out")
    assert argv[3:6] == ["compile", "--policy", "pi05"]
    assert argv[argv.index("--dataset") + 1] == str(dataset)
    with pytest.raises(ValueError, match="calibration dataset"):
        jobs.compile_command(found, "", tmp_path / "out")


def test_compile_cli_routes_pi05_to_the_staged_builder():
    from polima.cli.compile import _groot_build, _staged_policy

    assert _staged_policy(["--policy", "pi05", "--checkpoint", "c"]) == "pi05"
    assert _staged_policy(["--policy=pi05", "--build-dir", "b"]) is None
    assert not _groot_build(["--policy", "pi05"]) and _groot_build(["--policy", "groot"])


def test_builder_precisions_quantize_only_heavy_graphs(tmp_path):
    from polima.compile import pi05 as builder

    root = tmp_path / "build"
    root.mkdir()
    for precision, expected in (("bf16", ("bf16", "bf16")), ("w8a16", ("bf16", "int8")), ("w8a8", ("int8", "int8"))):
        (root / "build.json").write_text(json.dumps({"precision": precision}))
        jobs = {j["name"]: j for j in builder.graph_jobs(builder.Build(root))}
        assert len(jobs) == 11
        assert all((j["activation"], j["weights"]) == expected for j in jobs.values())
        assert all("--external-dram-layout" in j["flags"] for j in jobs.values())
        assert (jobs["expert"]["calibration"] is None) == (precision == "bf16")


def test_llima_client_sends_hwc_images_and_the_padded_group(tmp_path):
    pi05, _ = make_client(tmp_path)
    pi05.config["backend"] = "llima"
    pi05.backend, pi05.prefix, pi05.language_rows = "llima", 1024, 256
    image = np.random.default_rng(2).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    request = pi05.request([image] * 3, np.zeros(32), "pick", seed=1)
    for tensor in get_policy("pi05").wire.request_tensors:
        assert request[tensor.name].shape == tensor.shape, tensor.name
    np.testing.assert_array_equal(request["image0"], client.resize_with_pad(image).transpose(1, 2, 0))
    _, valid = pi05.tokens(client.prompt("pick", np.zeros(32)))
    assert np.all(request["prefix_bias"][768 + valid:] == client.MASK)  # prompt padding and group padding
    np.testing.assert_allclose(request["suffix_rope"], client.rope_rows(768 + valid))


def test_llima_request_converts_an_onnx_request():
    from polima.policies.pi05 import graphs

    chw = np.random.default_rng(0).standard_normal((3, 224, 224)).astype(np.float32)
    onnx = {"image0": client.patchify(chw), "language": np.ones((200, 2048), np.float32),
            "prefix_bias": np.zeros(456, np.float32), "suffix_rope": np.zeros((100, 256), np.float32),
            "noise": np.zeros((50, 32), np.float32)}
    request = graphs.llima_request(onnx, 1, 1024)
    np.testing.assert_array_equal(request["image0"], chw.transpose(1, 2, 0))
    assert request["language"].shape == (768, 2048) and not request["language"][200:].any()
    assert np.all(request["prefix_bias"][456:] == client.MASK) and not request["prefix_bias"][:456].any()


def test_padded_expert_row_masks_the_padding():
    from polima.policies.pi05 import graphs

    p = 40
    row = {"cache": np.ones((1, 1, 36 * p, 256), np.float32), "x": np.zeros((1, 1, 50, 32), np.float32),
           "cond": np.zeros((1, 1, 1, 1024), np.float32), "rope": np.zeros((1, 1, 100, 256), np.float32),
           "bias": np.zeros((1, 1, 1, p + 50), np.float32)}
    padded = graphs.pad_expert_row(row, 64)
    cache = padded["cache"].reshape(36, 64, 256)
    assert np.all(cache[:, :p] == 1) and not cache[:, p:].any()
    assert np.all(padded["bias"][..., p:64] == client.MASK) and not padded["bias"][..., 64:].any()


def _expert_weights(torch, layers=1):
    torch.manual_seed(0)
    w = {"action_in_proj.weight": torch.randn(1024, 32) * 0.05, "action_in_proj.bias": torch.zeros(1024),
         "action_out_proj.weight": torch.randn(32, 1024) * 0.05, "action_out_proj.bias": torch.zeros(32)}
    p = "paligemma_with_expert.gemma_expert.model"
    for i in range(layers):
        for name, shape in (("self_attn.q_proj", (2048, 1024)), ("self_attn.k_proj", (256, 1024)),
                            ("self_attn.v_proj", (256, 1024)), ("self_attn.o_proj", (1024, 2048)),
                            ("mlp.gate_proj", (4096, 1024)), ("mlp.up_proj", (4096, 1024)),
                            ("mlp.down_proj", (1024, 4096))):
            w[f"{p}.layers.{i}.{name}.weight"] = torch.randn(shape) * 0.02
        for norm in ("input_layernorm", "post_attention_layernorm"):
            w[f"{p}.layers.{i}.{norm}.dense.weight"] = torch.randn(3072, 1024) * 0.02
            w[f"{p}.layers.{i}.{norm}.dense.bias"] = torch.zeros(3072)
    w[f"{p}.norm.dense.weight"] = torch.randn(3072, 1024) * 0.02
    w[f"{p}.norm.dense.bias"] = torch.zeros(3072)
    return w


def test_nchw_expert_matches_the_expert_step():
    """GR00T's DiT rewrite applied to the expert computes the same velocity."""
    torch = pytest.importorskip("torch")
    from polima.policies.pi05 import blocks, modules

    p = 64
    step = blocks.ExpertStep(_expert_weights(torch), 1, p).eval()
    nchw = blocks.ExpertStepNchw(step).eval()
    cache = torch.randn(1, 1, 2 * p, 256)
    x, cond = torch.randn(1, 1, 50, 32), torch.randn(1, 1, 1, 1024)
    cos, sin = modules.rope_tables(torch.arange(40, 90))
    rope = torch.cat([cos, sin]).reshape(1, 1, 100, 256)
    bias = torch.zeros(1, 1, 1, p + 50)
    bias[..., 50:p] = modules.MASK
    with torch.no_grad():
        want = step(cache, x, cond, rope, bias).reshape(50, 32)
        got = nchw(*blocks.to_nchw_inputs(cache, x, cond, rope, bias, step, p)).reshape(32, 50).T
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4)


def test_newer_lerobot_checkpoints_strip_the_model_prefix(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    from polima.policies.pi05 import weights

    save_file({"model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight": torch.ones(4, 2048),
               "model.action_in_proj.weight": torch.zeros(2, 2)}, str(tmp_path / "model.safetensors"))
    assert set(weights.load(tmp_path)) == {"action_in_proj.weight"}
    assert weights.embedding_table(tmp_path)[0, 0] == pytest.approx(2048 ** 0.5)


def test_nchw_expert_spec_matches_its_plan_input():
    from dataclasses import replace

    from polima.bundle.pack import _plan_dict

    base = get_policy("pi05")
    spec = build_spec(Geometry.base(), {**base.runtime_context, "expert_form": "nchw"})
    graph = spec.compile.graph("expert")
    plan = _plan_dict(spec.build_runtime_plan())
    assert [t.name for t in graph.inputs] == list(rt.NCHW_EXPERT_INPUTS)
    assert sum(t.elements for t in graph.inputs) == plan["device_buffers"]["expert_in"]
    _ = replace


def test_device_denoise_runs_ten_euler_steps_in_one_device_program():
    """denoise='device': one program, x ping-pongs, each step binds its own adaRMS buffer,
    the prefill writes K/V into the buffer the expert binds, and x leaves the device once."""
    from polima.bundle.pack import _plan_dict

    base = get_policy("pi05")
    spec = build_spec(Geometry.base(), {**base.runtime_context, "expert_form": "nchw", "denoise": "device"})
    plan = _plan_dict(spec.build_runtime_plan())
    chains = [s for s in plan["steps"] if s["op"] == "run_device_chain"]
    assert [c["args"]["label"] for c in chains] == ["prefill", "denoise"]
    assert not any(s["op"] == "euler" for s in plan["steps"])
    denoise = chains[1]["args"]
    order = list(rt.NCHW_EXPERT_INPUTS)
    calls = denoise["calls"]
    assert len(calls) == rt.DENOISE_STEPS
    for k, call in enumerate(calls):
        inputs = dict(zip(order, (span[0] for span in call["in"]), strict=True))
        assert inputs["ada"] == f"ada_{k:02d}" and inputs["cache"] == "kv"
        assert inputs["x"] != call["out"][0][0]
        if k:
            assert inputs["x"] == calls[k - 1]["out"][0][0]
    assert denoise["downloads"] == [{"src": calls[-1]["out"][0][0], "count": rt.CHUNK * rt.ACTION_LANE, "dst": "x"}]
    assert {u["dst"] for u in denoise["uploads"] if u.get("cache")} >= {f"ada_{k:02d}" for k in range(10)}
    kv_writes = {span[0] for c in chains[0]["args"]["calls"] for span in c["out"][1:] if c["graph"].startswith("pre_")}
    assert kv_writes == {"kv"}
    devices = plan["device_buffers"]
    assert "expert_in" not in devices and devices["kv"] == rt.cache_rows(rt.llima_prefix(3)) * rt.ROW


def test_device_denoise_needs_the_nchw_expert():
    with pytest.raises(ValueError, match="NCHW"):
        rt.build_llima_plan(3, 32, rt.EXPERT_INPUTS, "nhwc", "device")


def test_euler_expert_returns_the_next_x():
    torch = pytest.importorskip("torch")
    from polima.policies.pi05 import blocks

    p = 64
    step = blocks.ExpertStepNchw(blocks.ExpertStep(_expert_weights(torch), 1, p).eval()).eval()
    euler = blocks.ExpertEulerNchw(step, rt.DT).eval()
    inputs = (torch.randn(1, 32, 1, 50), torch.randn(1, 3 * 3072, 1, 1), torch.randn(1, 256, 2, p),
              torch.randn(1, 512, 1, 50), torch.zeros(1, p + 50, 1, 1))
    ada = blocks.euler_ada(inputs[1])
    # Same HWC bytes: [1, C, H, W] HWC is H, W, C order.
    assert torch.equal(ada.permute(0, 2, 3, 1).reshape(-1), inputs[1].permute(0, 2, 3, 1).reshape(-1))
    with torch.no_grad():
        torch.testing.assert_close(euler(inputs[0], ada, *inputs[2:]), inputs[0] - rt.DT * step(*inputs))


def test_device_denoise_builds_quantize_then_compile_persistent(tmp_path):
    from polima.compile import pi05 as builder

    base = {"backend": "llima", "expert_form": "nchw", "precision": "w8a16"}
    assert builder.device_denoise({**base, "denoise": "device"})
    assert not builder.device_denoise(base)  # builds recorded before the option: APU loop
    assert not builder.device_denoise({**base, "expert_form": "nhwc", "denoise": "device"})

    class FakeBuild:
        def __init__(self, settings):
            self.settings = settings

        def path(self, *parts):
            return tmp_path.joinpath(*parts)

    (job,) = builder.graph_jobs(FakeBuild({**base, "denoise": "device"}))
    assert job["persistent"] and "--no-compile" in job["flags"] and "--no-simplify" in job["flags"]


def test_persistent_expert_maps_bindings_by_size(tmp_path):
    from polima.compile import pi05 as builder

    p = rt.llima_prefix(2)
    layout = rt.expert_layout(p, rt.NCHW_EXPERT_INPUTS)
    compiled = ("x", "ada", "rope", "cache", "mask")
    (tmp_path / "retained" / "x").mkdir(parents=True)
    (tmp_path / "retained" / "x" / "expert_stage1_mla.elf").write_bytes(b"\x7fELF")
    contract = {"inputs": [{"elements": layout[n][1]} for n in compiled], "outputs": [{"elements": 1600}]}
    (tmp_path / "contract.json").write_text(json.dumps(contract))
    found = builder.persistent_expert(tmp_path, p)
    assert found["order"] == list(compiled) and found["outputs"] == [1600]
    contract["outputs"] = [{"elements": 7}]
    (tmp_path / "contract.json").write_text(json.dumps(contract))
    with pytest.raises(builder.StageError, match="output x"):
        builder.persistent_expert(tmp_path, p)


def _onnx_fixture(where, rng, cameras=2):
    (where / "inputs").mkdir(parents=True)
    (where / "expected").mkdir(parents=True)
    p = rt.prefix_tokens(cameras)
    tensors = {**{f"image{k}": rng.standard_normal(256 * 588) for k in range(cameras)},
               "language": rng.standard_normal((200, 2048)), "prefix_bias": np.zeros(p),
               "suffix_rope": rng.standard_normal((100, 256)), "noise": rng.standard_normal((50, 32))}
    for name, value in tensors.items():
        value.astype("<f4").tofile(where / "inputs" / f"{name}.f32")
    rng.standard_normal((50, 6)).astype("<f4").tofile(where / "expected" / "normalized_actions.f32")


def test_llima_export_converts_every_fixture_in_the_set(tmp_path):
    from polima.policies.pi05 import graphs

    rng = np.random.default_rng(0)
    export = tmp_path / "export"
    _onnx_fixture(export / "fixtures", rng)
    for k in range(3):
        _onnx_fixture(export / "fixtures" / "set" / f"s{k:02d}", rng)
    (export / "constants").mkdir()
    graphs.write_llima_export(export, tmp_path / "derived", 2)
    for k in range(3):
        sample = tmp_path / "derived/fixtures/set" / f"s{k:02d}"
        bias = np.fromfile(sample / "inputs/prefix_bias.f32", "<f4")
        assert bias.size == rt.llima_prefix(2) and np.all(bias[rt.prefix_tokens(2):] == client.MASK)
        assert (sample / "expected/normalized_actions.f32").read_bytes() == \
            (export / "fixtures/set" / f"s{k:02d}" / "expected/normalized_actions.f32").read_bytes()


def test_check_judges_the_mean_of_the_fixture_set(tmp_path, monkeypatch):
    from polima.compile import pi05 as builder
    from polima.config import devkits

    bundle = tmp_path / "bundle"
    (bundle / "plan.json").parent.mkdir(parents=True)
    (bundle / "plan.json").write_text(json.dumps({"wire": {"response_shape": [50, 6]}}))
    rng = np.random.default_rng(1)
    want, got = {}, {}
    for k in range(4):  # three close fixtures and one hard one (alone it would fail w8a16's 0.03)
        name = f"s{k:02d}"
        (bundle / "fixtures/set" / name / "expected").mkdir(parents=True)
        want[name] = rng.standard_normal((50, 6)).astype("<f4")
        want[name].tofile(bundle / "fixtures/set" / name / "expected/normalized_actions.f32")
        got[name] = want[name] + (0.05 if k == 0 else 0.01)

    class FakeBuild:
        def __init__(self):
            self.settings, self.lines = {"bundle": str(bundle), "precision": "w8a16"}, []

        def path(self, *parts):
            return tmp_path.joinpath("build", *parts)

        def log(self, line):
            self.lines.append(line)

    commands = []

    def run(argv, check=False):
        commands.append(argv)
        if argv[0] == "rsync" and argv[-1].endswith("actions/"):
            for name, value in got.items():
                value.astype("<f4").tofile(Path(argv[-1]) / f"{name}.f32")
        return type("Done", (), {"returncode": 0})()

    monkeypatch.setattr(devkits, "apply", lambda target: None)
    monkeypatch.setattr(builder.subprocess, "run", run)
    build = FakeBuild()
    report = builder.board_check(build, "sima@devkit")
    assert report["fixtures"] == 4 and report["ok"]
    assert report["mean_abs"] == pytest.approx(0.02, abs=1e-6)
    assert report["worst"]["fixture"] == "s00" and report["worst"]["mean_abs"] == pytest.approx(0.05, abs=1e-6)
    assert sum(cmd.count("--input-dir") for cmd in (" ".join(c) for c in commands)) == 4
    assert "worst s00" in build.lines[-1]


def test_so101_bundles_get_their_rest_pose_on_every_backend(tmp_path, monkeypatch):
    """Both bundle paths (LLiMa and ONNX) collect the robot files through pi05_robot_files,
    which writes an SO-101 checkpoint's rest pose from its dataset."""
    from polima.compile import groot
    from polima.compile import pi05 as builder

    spec = build_spec(Geometry(**{**Geometry.base().__dict__, "cameras": ("observation.images.wrist_cam",
                                                                          "observation.images.top_cam"),
                                  "state_dim": 6, "action_dim": 6, "calibration_id": "so-arm101",
                                  "joint_names": ("a", "b", "c", "d", "e", "f"),
                                  "roles": (("wrist", "Wrist"), ("overhead", "Overhead"))}))
    (tmp_path / "client").mkdir()
    (tmp_path / "client" / "client.json").write_text("{}")
    written = []
    monkeypatch.setattr(groot, "write_rest_pose", lambda dataset, path, *rest: written.append((dataset, path)))
    monkeypatch.setattr("polima.bundle.importer.collect_robot_files", lambda spec, root: {})
    files = builder.pi05_robot_files(spec, tmp_path, tmp_path / "dataset")
    assert "pi05/client.json" in files and written[0][0] == tmp_path / "dataset"
    with pytest.raises(builder.StageError, match="rest pose"):
        builder.pi05_robot_files(spec, tmp_path, None)
    base = build_spec(Geometry.base())
    assert builder.pi05_robot_files(base, tmp_path, None) == {"pi05/client.json": tmp_path / "client/client.json"}
