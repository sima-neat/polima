"""run_device_chain: plan rewrite, reference executor, native Plan on a CPU double.

The miniature plans below copy the step structure of the production GR00T
N1.6 / N1.7 persistent-DiT plans (same graph names, packing, deepstack adds,
task-selected sidecars) at toy sizes. Hardware validation is separate: these
tests say nothing about the accelerator.
"""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from polima.bundle.device_chain import (
    N16_DIT,
    N16_QWEN,
    N17_LANGUAGE,
    N17_VISION,
    N17_VL,
    RewriteError,
    rewrite,
    verify_equivalence,
)

ROOT = Path(__file__).resolve().parents[2]
W = 32  # channels; keeps every device sub-span 64-byte aligned


def _graph(name, inputs, outputs, persistent=None, layout="HWC"):
    graph = {"name": name, "elf": f"models/{name}.elf", "input_elements": inputs,
             "output_elements": outputs, "dram_layout": "plain",
             "external_dram_layout": layout, "precision": "bf16", "sha256": "0" * 64}
    if persistent:
        graph["persistent_input_elements"] = persistent
    return graph


def _sidecar(constants: Path, name: str, count: int, rng) -> None:
    rng.standard_normal(count).astype(np.float32).tofile(constants / name)


def _pack_sidecar(out, src, count, stride=None):
    part = {"src": src, "dst_offset": 0, "count": count, "sidecar": True}
    if stride:
        part.update(source_stride=stride, selector_count=1)
    return {"op": "pack", "out": out, "args": {"parts": [part]}}


def _dit_tail(steps, hidden_in, temb, backbone, even, odd, out):
    steps.append({"op": "run_dit_chain", "out": out,
                  "args": {"graphs": N16_DIT, "in": [hidden_in, temb, backbone, even, odd]}})


def mini_n16(root: Path) -> Path:
    rng = np.random.default_rng(0)
    (root / "constants").mkdir(parents=True)
    tokens, hidden, mask = 8, 8 * W, 64
    slot = 2 * W
    dit_hidden, temb, dit_mask = 3 * W, W, tokens
    action = dit_hidden - W
    graphs = [_graph("eagle_vision_patch", 96, 128), _graph("eagle_vision_00_01", 128, 128),
              _graph("eagle_vision_connector", 128, slot, layout="compiler")]
    graphs += [_graph(n, hidden + mask, hidden) for n in N16_QWEN]
    graphs += [_graph("eagle_output_norm", hidden, hidden)]
    graphs += [_graph(n, dit_hidden + temb + hidden + dit_mask, dit_hidden,
                      [dit_hidden, temb, hidden, dit_mask]) for n in N16_DIT]
    graphs += [_graph("action_tail", dit_hidden + temb, action, layout="compiler"),
               _graph("state_project", W, W, layout="compiler"),
               _graph("action_project", 2 * action, action, layout="compiler")]
    c = root / "constants"
    _sidecar(c, "qwen_attention_masks", mask, rng)
    _sidecar(c, "prompt_embeddings", hidden, rng)
    _sidecar(c, "text_additive_masks", dit_mask, rng)
    _sidecar(c, "image_additive_mask", dit_mask, rng)
    for k in range(4):
        _sidecar(c, f"timestep_embedding_{k}", temb, rng)
    buffers = {"patches0": 96, "patches1": 96, "noise": action, "qwen_mask": mask,
               "text_mask": dit_mask, "image_mask": dit_mask, "language_embeddings": hidden,
               "backbone_features": hidden, "hidden_a": dit_hidden, "velocity": action,
               "actions": action, "state": 6, "state_lane": W, "state_features": W,
               "action_input": 2 * action, "action_features": action}
    steps = [_pack_sidecar("text_mask", "text_additive_masks", dit_mask, dit_mask),
             _pack_sidecar("qwen_mask", "qwen_attention_masks", mask, mask),
             _pack_sidecar("image_mask", "image_additive_mask", dit_mask)]
    for k in range(4):
        buffers[f"temb_0{k}"] = temb
        steps.append(_pack_sidecar(f"temb_0{k}", f"timestep_embedding_{k}", temb))
    for cam in range(2):
        buffers.update({f"vision_hidden_{cam}": 128, f"connector_input_{cam}": 128,
                        f"image_tokens_{cam}": slot})
        steps += [
            {"op": "run_elf_chain", "out": f"vision_hidden_{cam}",
             "args": {"graphs": ["eagle_vision_patch", "eagle_vision_00_01"], "in": [f"patches{cam}"]}},
            {"op": "pixel_unshuffle", "out": f"connector_input_{cam}",
             "args": {"src": f"vision_hidden_{cam}", "height": 4, "width": 4, "channels": 8, "factor": 2}},
            {"op": "run_elf", "out": f"image_tokens_{cam}",
             "args": {"graph": "eagle_vision_connector", "in": [f"connector_input_{cam}"]}}]
    steps.append({"op": "pack", "out": "language_embeddings", "args": {"parts": [
        {"src": "prompt_embeddings", "dst_offset": 0, "count": hidden, "sidecar": True,
         "source_stride": hidden, "selector_count": 1},
        {"src": "image_tokens_0", "dst_offset": W, "count": slot},
        {"src": "image_tokens_1", "dst_offset": 5 * W, "count": slot}]}})
    names = ["language_embeddings", "backbone_features"]
    for i, name in enumerate(N16_QWEN):
        steps.append({"op": "run_elf", "out": names[(i + 1) % 2],
                      "args": {"graph": name, "in": [names[i % 2], "qwen_mask"]}})
    steps.append({"op": "run_elf", "out": "backbone_features",
                  "args": {"graph": "eagle_output_norm", "in": ["language_embeddings"]}})
    steps += [{"op": "pack", "out": "state_lane", "args": {"parts": [
                  {"src": "state", "dst_offset": 0, "count": 6}]}},
              {"op": "run_elf", "out": "state_features",
               "args": {"graph": "state_project", "in": ["state_lane"]}},
              {"op": "slice", "out": "actions", "args": {"src": "noise", "count": action}}]
    for k in range(4):
        _sidecar(c, f"tau_embedding_{k}", action, rng)
        steps += [{"op": "pack", "out": "action_input", "args": {"parts": [
                      {"src": "actions", "dst_offset": 0, "count": action},
                      {"src": f"tau_embedding_{k}", "dst_offset": action, "count": action,
                       "sidecar": True}]}},
                  {"op": "run_elf", "out": "action_features",
                   "args": {"graph": "action_project", "in": ["action_input"]}},
                  {"op": "pack", "out": "hidden_a", "args": {"parts": [
                      {"src": "state_features", "dst_offset": 0, "count": W},
                      {"src": "action_features", "dst_offset": W, "count": action}]}}]
        _dit_tail(steps, "hidden_a", f"temb_0{k}", "backbone_features", "text_mask",
                  "image_mask", "hidden_a")
        steps += [{"op": "run_elf", "out": "velocity",
                   "args": {"graph": "action_tail", "in": ["hidden_a", f"temb_0{k}"]}},
                  {"op": "euler", "out": "actions", "args": {"src": "velocity", "scalar": -0.25}}]
    wire = {"request_tensors": [{"name": "patches0", "elements": 96},
                                {"name": "patches1", "elements": 96},
                                {"name": "state", "elements": 6},
                                {"name": "noise", "elements": action}]}
    return _write(root, graphs, {"buffers": buffers, "steps": steps, "result": "actions",
                                 "wire": wire}, rng)


def mini_n17(root: Path) -> Path:
    rng = np.random.default_rng(1)
    (root / "constants").mkdir(parents=True)
    hidden, rope, mask, vision, side, patch = 8 * W, W, 2 * W, 4 * W, 2 * W, 64
    dit_hidden, temb, dit_mask = 3 * W, W, 8
    action = dit_hidden - W
    packed = hidden + 2 * rope + mask
    graphs = [_graph("vision_patch", patch, vision)]
    for name in N17_VISION[1:]:
        graphs.append(_graph(name, vision, side if "merge" in name else vision))
    graphs += [_graph(n, packed, hidden) for n in N17_LANGUAGE]
    graphs += [_graph(n, hidden, hidden, layout="HWC" if n == "vlln" else "compiler") for n in N17_VL]
    graphs += [_graph(n, dit_hidden + temb + hidden + dit_mask, dit_hidden,
                      [dit_hidden, temb, hidden, dit_mask]) for n in N16_DIT]
    graphs += [_graph("action_tail", dit_hidden + temb, action + W, layout="compiler"),
               _graph("state_project", W, W, layout="compiler"),
               _graph("action_project", 2 * action, action, layout="compiler")]
    c = root / "constants"
    _sidecar(c, "text_embeddings", hidden, rng)
    for name in N17_LANGUAGE:
        for aux, count in (("cos", rope), ("sin", rope), ("attention_mask", mask)):
            _sidecar(c, f"{name}_{aux}", count, rng)
    buffers = {"patches0": patch, "patches1": patch, "noise": action, "velocity": action,
               "actions": action, "text_embeddings": hidden, "language_input": hidden,
               "vl_input": hidden, "state": W, "state_features": W}
    steps = []
    for cam in range(2):
        source = f"patches{cam}"
        for name in N17_VISION:
            if name.startswith("deepstack"):
                out = f"camera{cam}_deep{name[-1]}"
                buffers[out] = side
                steps.append({"op": "run_elf", "out": out, "args": {"graph": name, "in": [source]}})
                continue
            out = f"camera{cam}_tokens" if name == "vision_merge" else f"camera{cam}_{name}"
            buffers[out] = side if name == "vision_merge" else vision
            steps.append({"op": "run_elf", "out": out, "args": {"graph": name, "in": [source]}})
            source = out
    slots = [W, 5 * W]
    steps.append(_pack_sidecar("text_embeddings", "text_embeddings", hidden))
    steps.append({"op": "pack", "out": "language_input", "args": {"parts": [
        {"src": "text_embeddings", "dst_offset": 0, "count": hidden, "src_offset": 0},
        *({"src": f"camera{cam}_tokens", "dst_offset": slots[cam], "count": side, "src_offset": 0}
          for cam in range(2))]}})
    source = "language_input"
    for layer, name in enumerate(N17_LANGUAGE):
        for aux, count in (("attention_mask", mask), ("cos", rope), ("sin", rope)):
            buffers[f"{name}_{aux}"] = count
            steps.append(_pack_sidecar(f"{name}_{aux}", f"{name}_{aux}", count))
        buffers[name] = hidden
        steps.append({"op": "run_elf", "out": name, "args": {"graph": name, "in": [
            source, f"{name}_cos", f"{name}_sin", f"{name}_attention_mask"]}})
        if layer < 3:
            buffers[f"deepstack_{layer}"] = hidden
            steps += [{"op": "pack", "out": f"deepstack_{layer}", "args": {"parts": [
                {"src": f"camera{cam}_deep{layer}", "dst_offset": slots[cam], "count": side,
                 "src_offset": 0} for cam in range(2)]}},
                {"op": "euler", "out": name, "args": {"src": f"deepstack_{layer}", "scalar": -1.0}}]
        source = name
    for name in N17_VL:
        out = "vl_input" if name == "vlln" else name
        buffers[out] = hidden
        steps.append({"op": "run_elf", "out": out, "args": {"graph": name, "in": [source]}})
        source = out
    steps += [{"op": "run_elf", "out": "state_features",
               "args": {"graph": "state_project", "in": ["state"]}},
              {"op": "pack", "out": "actions", "args": {"parts": [
                  {"src": "noise", "dst_offset": 0, "count": action, "src_offset": 0}]}}]
    for k in range(4):
        _sidecar(c, f"temb_{k}", temb, rng)
        _sidecar(c, f"tau_{k}", action, rng)
        for j in range(2):
            _sidecar(c, f"mask_{k}_{j}", dit_mask, np.random.default_rng(10 + j))
        buffers.update({f"temb_{k}": temb, f"hidden_{k}": dit_hidden, f"mask_{k}_0": dit_mask,
                        f"mask_{k}_1": dit_mask, f"dit_out_{k}": dit_hidden, f"tau_{k}": action,
                        f"action_features_{k}": action, f"velocity_padded_{k}": action + W})
        steps += [_pack_sidecar(f"tau_{k}", f"tau_{k}", action),
                  _pack_sidecar(f"temb_{k}", f"temb_{k}", temb),
                  {"op": "run_elf", "out": f"action_features_{k}",
                   "args": {"graph": "action_project", "in": ["actions", f"tau_{k}"]}},
                  {"op": "pack", "out": f"hidden_{k}", "args": {"parts": [
                      {"src": "state_features", "dst_offset": 0, "count": W, "src_offset": 0},
                      {"src": f"action_features_{k}", "dst_offset": W, "count": action,
                       "src_offset": 0}]}},
                  _pack_sidecar(f"mask_{k}_0", f"mask_{k}_0", dit_mask),
                  _pack_sidecar(f"mask_{k}_1", f"mask_{k}_1", dit_mask)]
        _dit_tail(steps, f"hidden_{k}", f"temb_{k}", "vl_03", f"mask_{k}_0", f"mask_{k}_1",
                  f"dit_out_{k}")
        steps += [{"op": "run_elf", "out": f"velocity_padded_{k}",
                   "args": {"graph": "action_tail", "in": [f"dit_out_{k}", f"temb_{k}"]}},
                  {"op": "gather_strided", "out": "velocity", "args": {
                      "src": f"velocity_padded_{k}", "stride": 48, "take": 32, "count": 2}},
                  {"op": "euler", "out": "actions", "args": {"src": "velocity", "scalar": -0.25}}]
    wire = {"request_tensors": [{"name": "patches0", "elements": patch},
                                {"name": "patches1", "elements": patch},
                                {"name": "state", "elements": W},
                                {"name": "noise", "elements": action}]}
    return _write(root, graphs, {"buffers": buffers, "steps": steps, "result": "actions",
                                 "wire": wire}, rng)


def _write(root: Path, graphs, plan, rng) -> Path:
    (root / "models").mkdir()
    for graph in graphs:
        (root / graph["elf"]).touch()
    (root / "bundle.json").write_text(json.dumps(
        {"bundle_id": "groot-mini-0-00000000", "policy": "groot", "graphs": graphs}))
    (root / "plan.json").write_text(json.dumps(plan))
    inputs = root / "fixtures" / "inputs"
    inputs.mkdir(parents=True)
    for tensor in plan["wire"]["request_tensors"]:
        rng.standard_normal(tensor["elements"]).astype(np.float32).tofile(
            inputs / f"{tensor['name']}.f32")
    return root


CASES = [(mini_n16, ["lm"]), (mini_n16, ["lm", "connector"]), (mini_n16, ["lm", "connector", "dit"]),
         (mini_n16, ["dit"]), (mini_n16, ["lm", "connector", "dit", "denoise"]),
         (mini_n16, ["dit", "denoise"]), (mini_n17, ["lm"]), (mini_n17, ["vision", "lm"]),
         (mini_n17, ["vision", "lm", "dit"]), (mini_n17, ["dit"]),
         (mini_n17, ["vision", "lm", "dit", "denoise"])]


def _load(bundle: Path):
    return (json.loads((bundle / "plan.json").read_text()),
            json.loads((bundle / "bundle.json").read_text())["graphs"])


@pytest.mark.parametrize("make,levels", CASES)
def test_rewrite_is_byte_identical_on_the_reference_executor(tmp_path, make, levels):
    bundle = make(tmp_path / "b")
    plan, graphs = _load(bundle)
    new = rewrite(plan, graphs, levels)
    assert any(s["op"] == "run_device_chain" for s in new["steps"])
    report = verify_equivalence(bundle, new, trials=2)
    assert report["byte_identical"] == [True] * 3


def test_reference_check_detects_misbound_slots_and_masks(tmp_path):
    bundle = mini_n17(tmp_path / "b")
    plan, graphs = _load(bundle)
    good = rewrite(plan, graphs, ["vision", "lm", "dit"])
    swapped = copy.deepcopy(good)
    vision = next(s for s in swapped["steps"] if s["args"].get("label") == "vision")
    merges = [c for c in vision["args"]["calls"] if c["graph"] == "vision_merge"]
    merges[0]["out"][0][1], merges[1]["out"][0][1] = merges[1]["out"][0][1], merges[0]["out"][0][1]
    assert verify_equivalence(bundle, swapped, trials=1)["byte_identical"] == [False, False]
    masks = copy.deepcopy(good)
    for step in masks["steps"]:
        if step["args"].get("label", "").startswith("dit"):
            for call in step["args"]["calls"]:
                call["in"][3] = ["dit_mask_even" if call["in"][3] == ["dit_mask_odd"] else "dit_mask_odd"]
    assert verify_equivalence(bundle, masks, trials=1)["byte_identical"] == [False, False]


def test_rewrite_refuses_int8_graphs_and_unexpected_structure(tmp_path):
    bundle = mini_n16(tmp_path / "b")
    plan, graphs = _load(bundle)
    int8 = copy.deepcopy(graphs)
    for graph in int8:
        if graph["name"] == N16_QWEN[3]:
            graph["int8_inputs"] = [{"elements": 1, "multiplier": 1.0, "zero_point": 0}]
    with pytest.raises(RewriteError, match="INT8"):
        rewrite(plan, int8, ["lm"])
    broken = copy.deepcopy(plan)
    qwen = next(s for s in broken["steps"] if s["args"].get("graph") == N16_QWEN[4])
    qwen["args"]["in"][1] = "text_mask"
    with pytest.raises(RewriteError):
        rewrite(broken, graphs, ["lm"])
    with pytest.raises(RewriteError, match="already"):
        rewrite(rewrite(plan, graphs, ["lm"]), graphs, ["lm"])


# ------------------------------------------------------------- native Plan


@pytest.fixture(scope="module")
def run_plan(tmp_path_factory):
    compiler = shutil.which("g++")
    if not compiler or not Path("/usr/include/nlohmann/json.hpp").is_file():
        pytest.skip("needs g++ and nlohmann/json")
    binary = tmp_path_factory.mktemp("native") / "run_plan"
    native = ROOT / "tests" / "native"
    subprocess.run([compiler, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror",
                    f"-I{native / 'llima_mock'}", f"-I{ROOT / 'native/include'}",
                    str(native / "run_plan.cpp"), str(ROOT / "native/src/plan.cpp"),
                    "-o", str(binary)], check=True, capture_output=True, text=True)
    return binary


def _native(binary: Path, bundle: Path, out: Path, env=None, counters=None):
    proc = subprocess.run([str(binary), str(bundle), str(bundle / "fixtures" / "inputs"), str(out), "3"],
                          capture_output=True, text=True, env=env, check=False)
    if counters is not None:
        for line in proc.stdout.splitlines():
            key, _, value = line.partition("=")
            if value.isdigit():
                counters[key] = int(value)
    return proc.returncode, proc.stderr, out.read_bytes() if out.exists() else b""


def _variant(source: Path, target: Path, plan: dict) -> Path:
    shutil.copytree(source, target)
    (target / "plan.json").write_text(json.dumps(plan))
    return target


@pytest.mark.parametrize("make,levels", CASES)
def test_native_plan_matches_the_unchained_plan(tmp_path, run_plan, make, levels):
    bundle = make(tmp_path / "b")
    plan, graphs = _load(bundle)
    code, err, before = _native(run_plan, bundle, tmp_path / "before.f32")
    assert code == 0, err
    chained = _variant(bundle, tmp_path / "chained", rewrite(plan, graphs, levels))
    # Default: queued submissions (the mock refuses one ELF twice in one).
    code, err, after = _native(run_plan, chained, tmp_path / "after.f32")
    assert code == 0, err
    assert after == before
    unqueued_env = {"POLIMA_DISABLE_MLA_QUEUE": "1", "PATH": "/usr/bin:/bin"}
    code, err, unqueued = _native(run_plan, chained, tmp_path / "unqueued.f32", env=unqueued_env)
    assert code == 0 and unqueued == before, err


def test_repeated_elf_starts_a_new_queue_submission(tmp_path, run_plan):
    # N1.6 SO-101 LM program: two eagle_vision_connector calls. N1.7 vision:
    # every vision ELF once per camera. Both must split, never fall back.
    for make, levels, label in ((mini_n16, ["lm", "connector"], "lm"),
                                (mini_n17, ["vision", "lm"], "vision")):
        bundle = make(tmp_path / label)
        plan, graphs = _load(bundle)
        new = rewrite(plan, graphs, levels)
        program = next(s for s in new["steps"] if s["args"].get("label") == label)
        calls = [c["graph"] for c in program["args"]["calls"]]
        assert len(set(calls)) < len(calls)
        chained = _variant(bundle, tmp_path / f"{label}_chained", new)
        counters = {}
        code, err, _ = _native(run_plan, chained, tmp_path / f"{label}.f32", counters=counters)
        assert code == 0, err
        assert counters["queued_runs"] > 0 and counters["queue_submissions"] > 0


def test_native_plan_rejects_stale_reads_and_cache_hazards(tmp_path, run_plan):
    bundle = mini_n17(tmp_path / "b")
    plan, graphs = _load(bundle)
    good = rewrite(plan, graphs, ["vision", "lm", "dit"])

    stale = copy.deepcopy(good)  # the LM entry upload removed: layer 03_04 reads nothing
    lm = next(s for s in stale["steps"] if s["args"].get("label") == "lm")
    lm["args"]["uploads"] = [u for u in lm["args"]["uploads"] if u.get("src") != "language_02_02"]
    code, err, _ = _native(run_plan, _variant(bundle, tmp_path / "stale", stale), tmp_path / "x.f32")
    assert code != 0 and "not written earlier" in err

    hazard = copy.deepcopy(good)  # a cached cos tail moved under the hidden state a graph writes
    lm = next(s for s in hazard["steps"] if s["args"].get("label") == "lm")
    tail = next(u for u in lm["args"]["uploads"] if u.get("dst") == "lang_05_06" and u.get("cache"))
    tail["dst_offset"] = 0
    code, err, _ = _native(run_plan, _variant(bundle, tmp_path / "hazard", hazard), tmp_path / "y.f32")
    assert code != 0 and "cached device upload region" in err

    misaligned = copy.deepcopy(good)
    vision = next(s for s in misaligned["steps"] if s["args"].get("label") == "vision")
    merge = next(c for c in vision["args"]["calls"] if c["graph"] == "vision_merge")
    merge["out"][0][1] += 1
    code, err, _ = _native(run_plan, _variant(bundle, tmp_path / "align", misaligned), tmp_path / "z.f32")
    assert code != 0 and "aligned" in err


# --------------------------------------------------------- run_bound_chain


def _bound_variant(bundle: Path, target: Path, *, scramble: bool = False) -> Path:
    """mini_n16 with every run_dit_chain replaced by the equivalent run_bound_chain."""
    plan, graphs = _load(bundle)
    binds = ["hidden", "temb", "backbone_features", "text_mask", "image_mask"]
    for p, graph in enumerate(g for g in graphs if g["name"] in N16_DIT):
        counts = graph["persistent_input_elements"]
        names = ["hidden", "temb", "backbone_features", binds[3 + p % 2]]
        inputs = [{"name": n, "elements": c, "binding": f"input_{i}"}
                  for i, (n, c) in enumerate(zip(names, counts))]
        if scramble:  # numeric order where the runtime uses binding-name order
            inputs = [dict(x, binding=f"input_{9 - i}") for i, x in enumerate(inputs)]
        graph["persistent_bindings"] = {
            "inputs": inputs,
            "outputs": [{"name": "hidden_out", "elements": graph["output_elements"],
                         "binding": "output_0"}]}
    for step in plan["steps"]:
        if step["op"] == "run_dit_chain":
            step["op"] = "run_bound_chain"
            step["args"] = {"graphs": step["args"]["graphs"], "in": step["args"]["in"],
                            "bind": binds, "carry": "hidden"}
    out = _variant(bundle, target, plan)
    manifest = json.loads((out / "bundle.json").read_text())
    manifest["graphs"] = graphs
    (out / "bundle.json").write_text(json.dumps(manifest))
    return out


def test_bound_chain_matches_the_dit_chain_and_checks_binding_order(tmp_path, run_plan):
    bundle = mini_n16(tmp_path / "b")
    code, err, before = _native(run_plan, bundle, tmp_path / "before.f32")
    assert code == 0, err
    bound = _bound_variant(bundle, tmp_path / "bound")
    counters = {}
    code, err, after = _native(run_plan, bound, tmp_path / "after.f32", counters=counters)
    assert code == 0 and after == before, err
    assert counters["queued_runs"] > 0
    env = {"POLIMA_DISABLE_MLA_QUEUE": "1", "PATH": "/usr/bin:/bin"}
    code, err, unqueued = _native(run_plan, bound, tmp_path / "unqueued.f32", env=env)
    assert code == 0 and unqueued == before, err
    scrambled = _bound_variant(bundle, tmp_path / "scrambled", scramble=True)
    code, err, _ = _native(run_plan, scrambled, tmp_path / "x.f32")
    assert code != 0 and "binding-name order" in err
