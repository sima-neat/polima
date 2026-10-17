"""Pi0.5's on-device pipeline, as plan.json.

    vision(image_k) -> 256 x 2048 tokens               once per camera (image_k: 256 patches x 588)
    hidden = [vision_0 | ... | vision_{C-1} | language]  P = 256*C + 200 tokens
    for each Gemma-2B block (2 layers):                 nine blocks, once
        out = gemma_AA_BB(hidden, prefix_bias)          -> [hidden' | K,V of its 2 layers]
        hidden = out[: P*2048]
    cache = [K_0 | V_0 | K_1 | ... | V_17]              gathered from the nine outputs
    x = noise                                           50 x 32
    repeat 10 times, t = 1.0, 0.9, ... 0.1:
        v = expert(cache, x, cond_t, suffix_rope, suffix_bias)
        x -= 0.1 * v
    result = x                                          normalized actions, 50 x 32

What the client sends, and why it is not computed here:

* `language` (200 x 2048): Pi0.5 writes the robot state into its prompt as 256-bin
  tokens, so the prompt changes every request. The client tokenizes it and looks
  the rows up in the bundled (scaled) embedding table; a 1 GB table and a
  SentencePiece tokenizer do not belong in the plan interpreter.
* `prefix_bias` (P): 0 for each visible key, MASK for padded prompt tokens and
  for the 256 tokens of a camera the robot does not have.
* `suffix_rope` (100 x 256): cos then sin rows for the action tokens' positions
  n..n+49, where n is the number of visible prefix tokens; n changes with the
  prompt length.
* `noise` (50 x 32): the flow-matching seed, so a request is reproducible.
* `image_k` (256 x 588): the [-1, 1] image cut into 14x14 patches. The vision
  graph then starts with a dense layer: ModelSDK cannot detessellate a 3-channel
  image input.

`cond_t` is the adaRMS conditioning silu(time_mlp_out(silu(time_mlp_in(emb(t))))).
It depends only on the fixed Euler timestep, so export precomputes the ten
vectors as `constants/cond_XX.f32`; the expert graph applies each layer's
modulation itself.
"""
from __future__ import annotations

from typing import Any

from polima.policies.base import PolicySpec, RuntimePlan, Step

IMAGE_SIZE = 224
IMAGE_TOKENS = 256
PATCH_VALUES = 3 * 14 * 14  # one 14x14 RGB patch, laid out (c, y, x)
IMAGE_ELEMENTS = IMAGE_TOKENS * PATCH_VALUES
WIDTH = 2048
ROW = 256
LANGUAGE_TOKENS = 200
PREFIX_LAYERS = 18
BLOCK_LAYERS = 2
EXPERT_LAYERS = 18
CHUNK = 50
ACTION_LANE = 32
COND = 1024
DENOISE_STEPS = 10
DT = 1.0 / DENOISE_STEPS
WIRE_MAGIC = int.from_bytes(b"PI05", "little")
DEFAULT_PORT = 8094


def prefix_tokens(cameras: int) -> int:
    return IMAGE_TOKENS * cameras + LANGUAGE_TOKENS


def block_names() -> list[str]:
    return [f"gemma_{i:02d}_{i + BLOCK_LAYERS - 1:02d}" for i in range(0, PREFIX_LAYERS, BLOCK_LAYERS)]


def block_rows(tokens: int) -> int:
    """Rows of 256 in one prefix block's output: 8 per token of hidden, then K and V
    (one row per token) for each of its layers."""
    return tokens * (WIDTH // ROW) + 2 * BLOCK_LAYERS * tokens


def cache_rows(tokens: int) -> int:
    return 2 * PREFIX_LAYERS * tokens


def buffers(cameras: int, action_dim: int) -> dict[str, int]:
    p = prefix_tokens(cameras)
    sizes = {f"image{k}": IMAGE_ELEMENTS for k in range(cameras)}
    sizes.update({
        "language": LANGUAGE_TOKENS * WIDTH,
        "prefix_bias": p,
        "suffix_rope": 2 * CHUNK * ROW,
        "noise": CHUNK * ACTION_LANE,
        "suffix_bias": p + CHUNK,
        "cache": cache_rows(p) * ROW,
        "x": CHUNK * ACTION_LANE,
        "velocity": CHUNK * ACTION_LANE,
    })
    sizes.update({f"vision{k}": IMAGE_TOKENS * WIDTH for k in range(cameras)})
    for index, name in enumerate(block_names()):
        sizes[f"hidden{index}"] = p * WIDTH
        sizes[f"{name}_out"] = block_rows(p) * ROW
    sizes.update({f"cond_{k:02d}": COND for k in range(DENOISE_STEPS)})
    if action_dim != ACTION_LANE:
        sizes["actions"] = CHUNK * action_dim
    return sizes


def _step(op: str, out: str, **args) -> Step:
    return Step(op, out, args)


def build_steps(cameras: int, action_dim: int, expert_inputs: tuple[str, ...] | None = None) -> list[Step]:
    """`expert_inputs` is the compiled expert ELF's MLA input order (names from
    `EXPERT_INPUTS`); the runner concatenates the step's buffers in that order."""
    p = prefix_tokens(cameras)
    steps: list[Step] = []
    for k in range(cameras):
        steps.append(_step("run_elf", f"vision{k}", graph="vision", **{"in": [f"image{k}"]}))
    parts = [{"src": f"vision{k}", "dst_offset": k * IMAGE_TOKENS * WIDTH, "count": IMAGE_TOKENS * WIDTH}
             for k in range(cameras)]
    parts.append({"src": "language", "dst_offset": cameras * IMAGE_TOKENS * WIDTH,
                  "count": LANGUAGE_TOKENS * WIDTH})
    steps.append(_step("pack", "hidden0", parts=parts))
    names = block_names()
    hidden_elements = p * WIDTH
    kv_elements = 2 * BLOCK_LAYERS * p * ROW
    for index, name in enumerate(names):
        steps.append(_step("run_elf", f"{name}_out", graph=name, **{"in": [f"hidden{index}", "prefix_bias"]}))
        if index + 1 < len(names):
            steps.append(_step("slice", f"hidden{index + 1}", src=f"{name}_out", offset=0, count=hidden_elements))
    steps.append(_step("pack", "cache", parts=[
        {"src": f"{name}_out", "src_offset": hidden_elements, "dst_offset": index * kv_elements,
         "count": kv_elements} for index, name in enumerate(names)]))
    steps.append(_step("pack", "suffix_bias", parts=[{"src": "prefix_bias", "dst_offset": 0, "count": p}]))
    for k in range(DENOISE_STEPS):
        steps.append(_step("pack", f"cond_{k:02d}",
                           parts=[{"src": f"cond_{k:02d}", "dst_offset": 0, "count": COND, "sidecar": True}]))
    steps.append(_step("slice", "x", src="noise", count=CHUNK * ACTION_LANE))
    feeds = {"cache": "cache", "x": "x", "rope": "suffix_rope", "bias": "suffix_bias"}
    order = expert_inputs or EXPERT_INPUTS
    for k in range(DENOISE_STEPS):
        step_feeds = {**feeds, "cond": f"cond_{k:02d}"}
        steps.append(_step("run_elf", "velocity", graph="expert", **{"in": [step_feeds[i] for i in order]}))
        steps.append(_step("euler", "x", src="velocity", scalar=DT))
    if action_dim != ACTION_LANE:
        steps.append(_step("gather_strided", "actions", src="x", stride=ACTION_LANE, take=action_dim, count=CHUNK))
    return steps


#: The expert graph's inputs in export order.
EXPERT_INPUTS = ("cache", "x", "cond", "rope", "bias")


def sidecars() -> tuple[str, ...]:
    return tuple(f"cond_{k:02d}" for k in range(DENOISE_STEPS))


def build_plan(spec: PolicySpec, context: Any = None) -> RuntimePlan:
    runtime = dict(spec.runtime_context)
    cameras = int(runtime.get("cameras", 3))
    action_dim = spec.dataset.action_dim
    order = tuple(runtime.get("expert_inputs") or EXPERT_INPUTS)
    if runtime.get("backend") == "llima":
        form = runtime.get("expert_form", "nhwc")
        if form == "nchw" and not runtime.get("expert_inputs"):
            order = NCHW_EXPERT_INPUTS
        return build_llima_plan(cameras, action_dim, order, form, runtime.get("denoise", "host"))
    return RuntimePlan(
        buffers=buffers(cameras, action_dim),
        steps=tuple(build_steps(cameras, action_dim, order)),
        result="x" if action_dim == ACTION_LANE else "actions",
        sidecars=sidecars(),
    )


# ------------------------------------------------------------------ LLiMa backend
#
# The PaliGemma half compiled by LLiMa (sima_lmm): one vision ELF taking the
# [-1, 1] image as 224x224x3 HWC, and per Gemma layer a `pre` ELF (RMSNorm, QKV,
# RoPE -> q, K, V), one shared `attn` ELF (masked attention over the 1024-row
# group) and a `post` ELF (o_proj, residual, MLP). The prefix is padded to one
# 1024-token group; padded keys are masked, so the result is unchanged.
#
# Everything stays on the device: vision writes the image rows of the hidden
# state, each `pre` writes its layer's K and V straight into the expert's packed
# input, and the denoise loop uploads only x, the step's conditioning and the
# (cached) RoPE rows and key mask.

LLIMA_TOKENS = 1024
FREQS = 128  # rotary frequencies (head_dim / 2), cos table then sin table


LLIMA_GROUP_ALIGN = 128


def llima_prefix(cameras: int) -> int:
    """The LLiMa prefill group: the prefix rounded up to a multiple of 128 tokens
    (1024 for three cameras, 768 for two). Padded tokens are masked keys."""
    needed = IMAGE_TOKENS * cameras + LANGUAGE_TOKENS
    group = -(-needed // LLIMA_GROUP_ALIGN) * LLIMA_GROUP_ALIGN
    if group > LLIMA_TOKENS:
        raise ValueError(f"{cameras} cameras plus a {LANGUAGE_TOKENS}-token prompt exceed one "
                         f"{LLIMA_TOKENS}-token LLiMa group")
    return group


def llima_pre(layer: int) -> str:
    return f"pre_{layer:02d}"


def llima_post(layer: int) -> str:
    return f"post_{layer:02d}"


#: Per-step adaRMS modulation of the NCHW expert: [scale|shift|gate] of every layer's
#: two norms and the final norm (blocks.ada_vectors).
ADA = (2 * EXPERT_LAYERS + 1) * 3 * 1024
NCHW_EXPERT_INPUTS = ("x", "ada", "cache", "rope", "mask")


def expert_layout(tokens: int, order: tuple[str, ...]) -> dict[str, tuple[int, int]]:
    """(offset, count) of each expert input inside its packed MLA input, in `order`.
    Names cover both forms: (cache, x, cond, rope, bias) and NCHW (x, ada, cache, rope, mask)."""
    sizes = {"cache": cache_rows(tokens) * ROW, "x": CHUNK * ACTION_LANE, "cond": COND,
             "rope": 2 * CHUNK * ROW, "bias": tokens + CHUNK, "mask": tokens + CHUNK, "ada": ADA}
    layout, offset = {}, 0
    for name in order:
        layout[name] = (offset, sizes[name])
        offset += sizes[name]
    return layout


def _device_denoise(expert_inputs: tuple[str, ...], layout: dict[str, tuple[int, int]]) -> Step:
    """The ten Euler steps as one device program over the NCHW expert's persistent bindings
    (in compiled order). Step k reads x from one ping-pong buffer and writes x - dt * v to
    the other; adaRMS, RoPE and the key mask are cached uploads, K/V is the prefill's."""
    n = CHUNK * ACTION_LANE
    uploads = [{"dst": "x_a", "count": n, "src": "noise"},
               {"dst": "rope", "count": layout["rope"][1], "src": "rope_tokens", "cache": True},
               {"dst": "suffix_mask", "count": layout["mask"][1], "src": "suffix_bias", "cache": True}]
    uploads += [{"dst": f"ada_{k:02d}", "count": ADA, "sidecar": f"ada_{k:02d}", "cache": True}
                for k in range(DENOISE_STEPS)]
    calls = []
    for k in range(DENOISE_STEPS):
        current, following = ("x_a", "x_b") if k % 2 == 0 else ("x_b", "x_a")
        spans = {"x": [current], "ada": [f"ada_{k:02d}"], "rope": ["rope"], "cache": ["kv"], "mask": ["suffix_mask"]}
        calls.append({"graph": "expert", "in": [spans[name] for name in expert_inputs], "out": [[following]]})
    last = "x_a" if DENOISE_STEPS % 2 == 0 else "x_b"
    return Step("run_device_chain", "x", {"label": "denoise", "uploads": uploads, "calls": calls,
                                          "downloads": [{"src": last, "count": n, "dst": "x"}]})


def build_llima_plan(cameras: int, action_dim: int, expert_inputs: tuple[str, ...] = EXPERT_INPUTS,
                     expert_form: str = "nhwc", denoise: str = "host") -> RuntimePlan:
    """`expert_form` "nchw": the GR00T-DiT-form expert (blocks.ExpertStepNchw) taking the
    precomputed per-step adaRMS modulation (sidecars ada_XX) and RoPE as [50][cos|sin].

    `denoise` "device" (NCHW only): the expert ELF is blocks.ExpertEulerNchw with one
    persistent binding per input, so the ten steps are one device program -- x ping-pongs
    between two device buffers, each step binds its own (cached) adaRMS buffer and the
    prefill's K/V buffer, and the APU uploads the noise and downloads x once."""
    nchw = expert_form == "nchw"
    on_device = denoise == "device"
    if on_device and not nchw:
        raise ValueError("denoise='device' needs the NCHW expert form")
    p = llima_prefix(cameras)
    text = p - IMAGE_TOKENS * cameras
    image = IMAGE_SIZE * IMAGE_SIZE * 3
    layout = expert_layout(p, tuple(expert_inputs))
    expert_elements = sum(count for _, count in layout.values())
    kv_offset, _ = layout["cache"]
    host = {f"image{k}": image for k in range(cameras)}
    host.update({"language": text * WIDTH, "prefix_bias": p, "suffix_rope": 2 * CHUNK * ROW,
                 "noise": CHUNK * ACTION_LANE, "mask": p * p, "suffix_bias": p + CHUNK,
                 "x": CHUNK * ACTION_LANE, "velocity": CHUNK * ACTION_LANE, "prefill_done": 1})
    if on_device:
        del host["velocity"]
    if nchw:
        host.update({"rope_sin": CHUNK * ROW, "rope_tokens": 2 * CHUNK * ROW})
    else:
        host.update({f"cond_{k:02d}": COND for k in range(DENOISE_STEPS)})
    if action_dim != ACTION_LANE:
        host["actions"] = CHUNK * action_dim
    device = {"hidden_a": p * WIDTH, "hidden_b": p * WIDTH, "freq_real": p * FREQS, "freq_imag": p * FREQS,
              "q": 8 * p * ROW, "attn": p * WIDTH, "mask": p * p}
    if on_device:
        device.update({"kv": layout["cache"][1], "x_a": CHUNK * ACTION_LANE, "x_b": CHUNK * ACTION_LANE,
                       "rope": layout["rope"][1], "suffix_mask": layout["mask"][1]})
        device.update({f"ada_{k:02d}": ADA for k in range(DENOISE_STEPS)})
        kv_buffer, kv_offset = "kv", 0
    else:
        device.update({"expert_in": expert_elements, "velocity": CHUNK * ACTION_LANE})
        kv_buffer = "expert_in"
    device.update({f"pixels{k}": image for k in range(cameras)})

    steps: list[Step] = [
        # Key padding row broadcast to every query row of the group.
        _step("gather_strided", "mask", src="prefix_bias", stride=0, take=p, dst_stride=p, count=p),
        _step("pack", "suffix_bias", parts=[{"src": "prefix_bias", "dst_offset": 0, "count": p}]),
    ]
    uploads = [{"dst": f"pixels{k}", "count": image, "src": f"image{k}"} for k in range(cameras)]
    uploads += [
        {"dst": "hidden_a", "dst_offset": IMAGE_TOKENS * cameras * WIDTH, "count": text * WIDTH, "src": "language"},
        {"dst": "freq_real", "count": p * FREQS, "sidecar": "freq_real", "cache": True},
        {"dst": "freq_imag", "count": p * FREQS, "sidecar": "freq_imag", "cache": True},
        {"dst": "mask", "count": p * p, "src": "mask"},
    ]
    calls = [{"graph": "vision", "in": [[f"pixels{k}", 0, image]],
              "out": [["hidden_a", k * IMAGE_TOKENS * WIDTH, IMAGE_TOKENS * WIDTH]]} for k in range(cameras)]
    hidden = ["hidden_a", "hidden_b"]
    kv = p * ROW
    for layer in range(PREFIX_LAYERS):
        current = hidden[layer % 2]
        k_span = [kv_buffer, kv_offset + 2 * layer * kv, kv]
        v_span = [kv_buffer, kv_offset + (2 * layer + 1) * kv, kv]
        calls.append({"graph": llima_pre(layer), "in": [[current, 0, p * WIDTH], ["freq_real", 0, p * FREQS],
                                                        ["freq_imag", 0, p * FREQS]],
                      "out": [["q", 0, 8 * p * ROW], k_span, v_span]})
        if layer + 1 < PREFIX_LAYERS:  # the last layer contributes only its K and V
            calls.append({"graph": "attn", "in": [["q", 0, 8 * p * ROW], k_span, ["mask", 0, p * p], v_span],
                          "out": [["attn", 0, p * WIDTH]]})
            calls.append({"graph": llima_post(layer), "in": [[current, 0, p * WIDTH], ["attn", 0, p * WIDTH]],
                          "out": [[hidden[(layer + 1) % 2], 0, p * WIDTH]]})
    steps.append(Step("run_device_chain", "prefill_done", {"label": "prefill", "uploads": uploads, "calls": calls}))
    if nchw:  # RoPE rows token-major for the NCHW graph: [cos_t | sin_t] per action token
        steps.append(_step("slice", "rope_sin", src="suffix_rope", offset=CHUNK * ROW, count=CHUNK * ROW))
        steps.append(_step("gather_strided", "rope_tokens", src="suffix_rope", stride=ROW, take=ROW,
                           dst_stride=2 * ROW, dst_offset=0, count=CHUNK))
        steps.append(_step("gather_strided", "rope_tokens", src="rope_sin", stride=ROW, take=ROW,
                           dst_stride=2 * ROW, dst_offset=ROW, count=CHUNK))
    else:
        for k in range(DENOISE_STEPS):
            steps.append(_step("pack", f"cond_{k:02d}",
                               parts=[{"src": f"cond_{k:02d}", "dst_offset": 0, "count": COND, "sidecar": True}]))
    if on_device:
        steps.append(_device_denoise(tuple(expert_inputs), layout))
        if action_dim != ACTION_LANE:
            steps.append(_step("gather_strided", "actions", src="x", stride=ACTION_LANE, take=action_dim,
                               count=CHUNK))
        return RuntimePlan(buffers=host, steps=tuple(steps), result="x" if action_dim == ACTION_LANE else "actions",
                           sidecars=(*(f"ada_{k:02d}" for k in range(DENOISE_STEPS)), "freq_real", "freq_imag"),
                           device_buffers=device)
    steps.append(_step("slice", "x", src="noise", count=CHUNK * ACTION_LANE))
    feeds = ({"x": "x", "rope": "rope_tokens", "mask": "suffix_bias"} if nchw
             else {"x": "x", "rope": "suffix_rope", "bias": "suffix_bias"})
    for k in range(DENOISE_STEPS):
        step_uploads = [{"dst": "expert_in", "dst_offset": layout[name][0], "count": layout[name][1],
                         "src": source, "cache": name in ("rope", "bias", "mask")}
                        for name, source in feeds.items()]
        if nchw:
            step_uploads.append({"dst": "expert_in", "dst_offset": layout["ada"][0], "count": ADA,
                                 "sidecar": f"ada_{k:02d}"})
        else:
            step_uploads.append({"dst": "expert_in", "dst_offset": layout["cond"][0], "count": COND,
                                 "src": f"cond_{k:02d}", "cache": False})
        steps.append(Step("run_device_chain", "velocity", {
            "label": f"expert_{k:02d}", "uploads": step_uploads,
            "calls": [{"graph": "expert", "in": [["expert_in", 0, expert_elements]],
                       "out": [["velocity", 0, CHUNK * ACTION_LANE]]}],
            "downloads": [{"src": "velocity", "count": CHUNK * ACTION_LANE, "dst": "velocity"}]}))
        steps.append(_step("euler", "x", src="velocity", scalar=DT))
    if action_dim != ACTION_LANE:
        steps.append(_step("gather_strided", "actions", src="x", stride=ACTION_LANE, take=action_dim, count=CHUNK))
    constants = tuple(f"ada_{k:02d}" for k in range(DENOISE_STEPS)) if nchw else sidecars()
    return RuntimePlan(buffers=host, steps=tuple(steps), result="x" if action_dim == ACTION_LANE else "actions",
                       sidecars=(*constants, "freq_real", "freq_imag"), device_buffers=device)
