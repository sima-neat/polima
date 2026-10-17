"""SmolVLA's denoising loop as one device program (`run_device_chain`).

The host plan (runtime.build_steps) runs the prefix, then ten times: sincos time
features, two strided gathers into the suffix input, the suffix graph, a pack of
the 2.47M-element K/V cache with the suffix into the denoise input, the denoise
graph and a host Euler update. The K/V cache never changes across the ten steps,
yet is re-packed and re-uploaded every time.

`device_plan` rewrites that plan so that one device program does it all:

  uploads   prefix_embeddings -> prefix_in, noise -> x_a, tbias_00..09 (cached)
  calls     prefix(prefix_in) -> cache
            denoise_step(x_a, tbias_00, cache) -> x_b
            denoise_step(x_b, tbias_01, cache) -> x_a      ... ten steps, ping-pong
  download  x

`denoise_step` (graphs.DenoiseStep) folds the suffix graph and the Euler update
into the expert: its time MLP's per-step term is the cached `tbias_XX` constant
(graphs.step_time_bias), and it returns x - dt * velocity. It is compiled with one
persistent binding per input, in compiler order (`step_inputs`).
"""

from __future__ import annotations

import copy

from polima.policies.smolvla.runtime import (
    ACTION_LANE,
    CACHE_ELEMENTS,
    CHUNK,
    DENOISE_STEPS,
    EXPERT_HIDDEN,
    PREFIX_ELEMENTS,
)

STEP_GRAPH = "denoise_step"
STEP_INPUTS = ("x", "tbias", "cache")
X_ELEMENTS = CHUNK * ACTION_LANE
LOOP_GRAPHS = ("suffix", "denoise")
LOOP_OPS = ("sincos_time", "gather_strided", "euler")


def tbias_sidecar(step: int) -> str:
    return f"tbias_{step:02d}"


def device_plan(plan: dict, step_inputs: tuple[str, ...] = STEP_INPUTS) -> dict:
    """Rewrite a SmolVLA host plan (plan.json dict) into the device-loop form."""
    if sorted(step_inputs) != sorted(STEP_INPUTS):
        raise ValueError(f"denoise_step inputs must be {STEP_INPUTS} in some order, got {step_inputs}")
    if "device_buffers" in plan:
        raise ValueError("plan already uses device programs")
    steps = plan["steps"]
    prefix = next(i for i, s in enumerate(steps) if s["op"] == "run_elf" and s["args"].get("graph") == "prefix")
    unpad = next(i for i, s in enumerate(steps)
                 if i > prefix and s["op"] == "gather_strided" and s["out"] == "actions_lane")
    loop = steps[prefix + 1:unpad]
    unexpected = [s for s in loop if not (s["op"] in ("slice", "pack", *LOOP_OPS)
                                          or s["args"].get("graph") in LOOP_GRAPHS)]
    if unexpected or sum(1 for s in loop if s["op"] == "euler") != DENOISE_STEPS:
        raise ValueError(f"not a SmolVLA host denoising loop: {unexpected[:2]}")

    device = {"prefix_in": PREFIX_ELEMENTS, "cache": CACHE_ELEMENTS, "x_a": X_ELEMENTS, "x_b": X_ELEMENTS}
    device.update({tbias_sidecar(k): EXPERT_HIDDEN for k in range(DENOISE_STEPS)})
    uploads = [{"dst": "prefix_in", "count": PREFIX_ELEMENTS, "src": "prefix_embeddings"},
               {"dst": "x_a", "count": X_ELEMENTS, "src": "noise"}]
    uploads += [{"dst": tbias_sidecar(k), "count": EXPERT_HIDDEN, "sidecar": tbias_sidecar(k), "cache": True}
                for k in range(DENOISE_STEPS)]
    calls = [{"graph": "prefix", "in": [["prefix_in", 0, PREFIX_ELEMENTS]], "out": [["cache", 0, CACHE_ELEMENTS]]}]
    for k in range(DENOISE_STEPS):
        current, following = ("x_a", "x_b") if k % 2 == 0 else ("x_b", "x_a")
        spans = {"x": [current, 0, X_ELEMENTS], "tbias": [tbias_sidecar(k), 0, EXPERT_HIDDEN],
                 "cache": ["cache", 0, CACHE_ELEMENTS]}
        calls.append({"graph": STEP_GRAPH, "in": [spans[name] for name in step_inputs],
                      "out": [[following, 0, X_ELEMENTS]]})
    last = "x_a" if DENOISE_STEPS % 2 == 0 else "x_b"
    chain = {"op": "run_device_chain", "out": "x",
             "args": {"label": "prefix+denoise", "uploads": uploads, "calls": calls,
                      "downloads": [{"src": last, "count": X_ELEMENTS, "dst": "x"}]}}

    result = copy.deepcopy(plan)
    result["steps"] = [*copy.deepcopy(steps[:prefix]), chain, *copy.deepcopy(steps[unpad:])]
    # The loop's host buffers (suffix_input, denoise_input, ...) are now unused; they
    # cost only load-time memory, so the buffer table is kept as is.
    result["device_buffers"] = device
    result["sidecars"] = [*plan.get("sidecars", []), *(tbias_sidecar(k) for k in range(DENOISE_STEPS))]
    for graph in LOOP_GRAPHS:
        result.get("graph_outputs", {}).pop(graph, None)
    return result


#: SigLIP's post-LayerNorm tokens per camera, what the LLiMa vision graph returns (1024 x 768).
VISION_TOKENS = 1024 * 768


def split_vision(plan: dict, tokens: int = VISION_TOKENS) -> dict:
    """Host plan with SmolVLA's vision split in two graphs: `vision` (image -> SigLIP tokens,
    compiled by LLiMa) and `connector` (graphs.VisionConnector: tokens -> the camera's 64
    prefix rows, image scale folded in), replacing `run_elf vision` + `scale` per camera."""
    steps = []
    for step in plan["steps"]:
        if step["op"] == "run_elf" and step["args"].get("graph") == "vision":
            camera = step["out"].removeprefix("vision")
            steps.append({"op": "run_elf", "out": f"tokens{camera}", "args": {"graph": "vision",
                                                                             "in": list(step["args"]["in"])}})
            steps.append({"op": "run_elf", "out": f"vision{camera}_scaled",
                          "args": {"graph": "connector", "in": [f"tokens{camera}"]}})
        elif step["op"] == "scale" and step["out"].startswith("vision"):
            continue  # folded into the connector
        else:
            steps.append(copy.deepcopy(step))
    if not any(s["args"].get("graph") == "connector" for s in steps):
        raise ValueError("plan has no vision graph to split")
    result = copy.deepcopy(plan)
    result["steps"] = steps
    cameras = [s["out"].removeprefix("tokens") for s in steps if s["out"].startswith("tokens")]
    result["buffers"] = {**plan["buffers"], **{f"tokens{c}": tokens for c in cameras}}
    outputs = result.setdefault("graph_outputs", {})
    outputs["vision"] = [{"name": "tokens", "shape": [1, 1, tokens // 768, 768], "dtype": "float32",
                          "dram_layout": "plain", "logical_width": None, "logical_channels": None}]
    outputs["connector"] = [{"name": "image_tokens", "shape": [1, 1, 64, 960], "dtype": "float32",
                             "dram_layout": "plain", "logical_width": None, "logical_channels": None}]
    return result


def optimized_plan(plan: dict, step_inputs: tuple[str, ...] = STEP_INPUTS) -> dict:
    """The optimized build's plan: split vision, then one device program for vision,
    connector, prefix and the ten denoising steps."""
    return vision_on_device(device_plan(split_vision(plan), step_inputs))


def vision_on_device(plan: dict) -> dict:
    """Move the camera path of a `device_plan` into its device program.

    Expects the split vision form, per camera `run_elf vision` (image -> SigLIP tokens) then
    `run_elf connector` (tokens -> the camera's 64 prefix rows, image scale included). The
    connector then writes straight into `prefix_in`; the constant empty-camera and language
    rows become cached sidecar uploads into it, and the host's state embedding is uploaded
    into its row. The host prefix pack and the token round trip disappear."""
    steps = plan["steps"]
    chain_index = next(i for i, s in enumerate(steps) if s["op"] == "run_device_chain")
    chain = copy.deepcopy(steps[chain_index])
    vision = [s for s in steps[:chain_index] if s["op"] == "run_elf" and s["args"].get("graph") == "vision"]
    connector = [s for s in steps[:chain_index] if s["op"] == "run_elf" and s["args"].get("graph") == "connector"]
    pack = next(s for s in steps[:chain_index] if s["op"] == "pack" and s["out"] == "prefix_embeddings")
    if not vision or len(vision) != len(connector):
        raise ValueError("vision_on_device needs per-camera vision -> connector graphs")
    images = [s["args"]["in"][0] for s in vision]
    tokens = plan["buffers"][vision[0]["out"]]
    rows = {part["src"]: part for part in pack["args"]["parts"]}

    uploads = [u for u in chain["args"]["uploads"] if u["dst"] != "prefix_in"]
    calls = []
    for camera, (v, c) in enumerate(zip(vision, connector)):
        part = rows[c["out"]]
        uploads.insert(camera, {"dst": f"pixels{camera}", "count": plan["buffers"][images[camera]],
                                "src": images[camera]})
        calls.append({"graph": "vision", "in": [[f"pixels{camera}", 0, plan["buffers"][images[camera]]]],
                      "out": [["tokens", 0, tokens]]})
        calls.append({"graph": "connector", "in": [["tokens", 0, tokens]],
                      "out": [["prefix_in", part["dst_offset"], part["count"]]]})
    moved = {c["out"] for c in connector}
    for part in pack["args"]["parts"]:
        if part["src"] in moved:
            continue
        upload = {"dst": "prefix_in", "dst_offset": part["dst_offset"], "count": part["count"]}
        upload.update({"sidecar": part["src"], "cache": True} if part.get("sidecar") else {"src": part["src"]})
        uploads.append(upload)
    chain["args"]["uploads"] = uploads
    chain["args"]["calls"] = calls + chain["args"]["calls"]
    chain["args"]["label"] = "vision+prefix+denoise"

    gone = {id(s) for s in (*vision, *connector, pack)}
    result = copy.deepcopy(plan)
    result["steps"] = [copy.deepcopy(s) for s in steps[:chain_index] if id(s) not in gone]
    result["steps"] += [chain, *copy.deepcopy(steps[chain_index + 1:])]
    result["device_buffers"] = {**plan["device_buffers"], "tokens": tokens,
                                **{f"pixels{k}": plan["buffers"][image] for k, image in enumerate(images)}}
    for graph in ("vision", "connector"):
        result.get("graph_outputs", {}).pop(graph, None)
    return result
