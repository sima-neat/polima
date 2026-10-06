"""SmolVLA's on-device pipeline, as plan.json.

Every constant here is transcribed from the working
`SmolVLA/devkit/smolvla_som_server/smolvla_som_server.cpp`, which is the binary
that currently serves SmolVLA on the board. This module emits the same sequence
of operations as data, so the one `polima_server` runs it without SmolVLA being
compiled into the binary.

## The pipeline

    vision(image0) -> 64x960 tokens          } run twice, same ELF
    vision(image1) -> 64x960 tokens          }
    state -> normalize(6) -> pad to 32 -> matvec(960x32) -> state embedding
    pack 241x960 prefix:  [ img0*sqrt(960) | img1*sqrt(960) | empty | language | state ]
    prefix(...) -> 2,467,840 packed KV cache
    x = noise (50x32)
    repeat 10 times, t = 1.0, 0.9, ... 0.1:
        suffix( pack per token [ x(32) | sincos(t)(720) ] ) -> 50x720
        denoise( pack [ cache | suffix at 1205*2048 ] )     -> velocity 50x32
        x -= 0.1 * velocity
    actions = denormalize(gather x[t*32 .. +6])             -> 50x6

Three things that look arbitrary and are not:

* **`sqrt(960)` on the image tokens.** SmolVLM scales image embeddings by
  sqrt(hidden) before they enter the language model. Dropping it does not error;
  it just puts the vision tokens at the wrong magnitude relative to the language
  ones.
* **The 32-wide action lane carrying 6 real joints.** Like ACT's 16-wide action
  head, this is MLA channel alignment. Positions 6..31 are zero going in and
  ignored coming out.
* **The denoise input is 1223x2048 with the suffix written at token 1205.** The
  cache occupies the first 1205 tokens; the rest is zero except that window.

## Why the loop is unrolled

Ten iterations are emitted as ~40 explicit steps rather than a loop opcode. The
interpreter then needs no control flow at all, each iteration's timestep is a
literal, and `--verbose` reports per-step timings that show which of the twenty
MLA calls is slow. The plan is generated, so the repetition costs nothing.
"""

from __future__ import annotations

import math
from typing import Any

from polima.policies.base import PolicySpec, RuntimePlan, Step

# ------------------------------------------------------------------ geometry

IMAGE_HEIGHT = IMAGE_WIDTH = 512
IMAGE_ELEMENTS = 3 * IMAGE_HEIGHT * IMAGE_WIDTH          # 786432, NCHW

HIDDEN = 960                     # SmolVLM2 hidden width
IMAGE_TOKENS = 64                # tokens per camera
LANGUAGE_TOKENS = 48
PREFIX_TOKENS = 241              # 64 + 64 + 64 (empty) + 48 + 1 (state)
STATE_TOKEN_INDEX = 240

VISION_ELEMENTS = IMAGE_TOKENS * HIDDEN                  # 61440
LANGUAGE_ELEMENTS = LANGUAGE_TOKENS * HIDDEN             # 46080
PREFIX_ELEMENTS = PREFIX_TOKENS * HIDDEN                 # 231360

#: The packed KV cache the prefix graph emits: 16 layers x (key, value).
CACHE_ELEMENTS = 2467840

CHUNK = 50                       # action timesteps per inference
ACTION_DIM = 6                   # real SO-101 joints
ACTION_LANE = 32                 # padded lane width on the MLA
STATE_DIM = 6

EXPERT_HIDDEN = 720              # action-expert width
SUFFIX_TOKEN = ACTION_LANE + EXPERT_HIDDEN               # 752
SUFFIX_IN_ELEMENTS = CHUNK * SUFFIX_TOKEN                # 37600
SUFFIX_OUT_ELEMENTS = CHUNK * EXPERT_HIDDEN              # 36000

DENOISE_TOKENS = 1223
DENOISE_WIDTH = 2048
DENOISE_IN_ELEMENTS = DENOISE_TOKENS * DENOISE_WIDTH     # 2504704
DENOISE_SUFFIX_TOKEN = 1205                              # where the suffix lands
DENOISE_SUFFIX_OFFSET = DENOISE_SUFFIX_TOKEN * DENOISE_WIDTH
NOISE_ELEMENTS = CHUNK * ACTION_LANE                     # 1600

#: Flow-matching integration: 10 Euler steps from t=1 down to t=0.1.
DENOISE_STEPS = 10
DT = 1.0 / DENOISE_STEPS

#: Time embedding periods, from smolvla_som_server.cpp::make_suffix_input.
#: period = MIN * (MAX/MIN)^(i/(half-1)), angle = t * 2pi/period.
TIME_MIN_PERIOD = 0.004
TIME_MAX_PERIOD = 4.0

#: sqrt(hidden), applied to image tokens before they join the prefix.
IMAGE_SCALE = math.sqrt(float(HIDDEN))

RESPONSE_ELEMENTS = CHUNK * ACTION_DIM                   # 300

WIRE_MAGIC = 0x534D4F4C          # "SMOL"
DEFAULT_PORT = 8081


# -------------------------------------------------------------------- buffers


def buffers(state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM) -> dict[str, int]:
    """Every buffer the plan touches, allocated once at load."""
    if not 0 < state_dim <= ACTION_LANE or not 0 < action_dim <= ACTION_LANE:
        raise ValueError(
            f"SmolVLA state/action dimensions must fit the {ACTION_LANE}-wide MLA lane"
        )
    response_elements = CHUNK * action_dim
    sizes = {
        # wire inputs
        "image0": IMAGE_ELEMENTS,
        "image1": IMAGE_ELEMENTS,
        "state": state_dim,
        "noise": NOISE_ELEMENTS,
        # vision
        "vision0": VISION_ELEMENTS,
        "vision1": VISION_ELEMENTS,
        "vision0_scaled": VISION_ELEMENTS,
        "vision1_scaled": VISION_ELEMENTS,
        # state path
        "state_normalized": state_dim,
        "state_lane": ACTION_LANE,
        "state_embedding": HIDDEN,
        # prefix
        "prefix_embeddings": PREFIX_ELEMENTS,
        "cache": CACHE_ELEMENTS,
        # denoise loop
        "x": NOISE_ELEMENTS,
        "suffix_input": SUFFIX_IN_ELEMENTS,
        "suffix_output": SUFFIX_OUT_ELEMENTS,
        "denoise_input": DENOISE_IN_ELEMENTS,
        "velocity": NOISE_ELEMENTS,
        # output
        "actions_lane": response_elements,
        "actions": response_elements,
    }
    for step in range(DENOISE_STEPS):
        sizes[f"time_{step:02d}"] = EXPERT_HIDDEN
    return sizes


# ---------------------------------------------------------------------- steps


def _step(op: str, out: str, **args) -> Step:
    return Step(op, out, args)


def build_steps(state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM) -> list[Step]:
    steps: list[Step] = []

    # --- vision, the same ELF run once per camera --------------------------
    steps.append(_step("run_elf", "vision0", graph="vision", **{"in": ["image0"]}))
    steps.append(_step("run_elf", "vision1", graph="vision", **{"in": ["image1"]}))
    steps.append(_step("scale", "vision0_scaled", src="vision0", scalar=IMAGE_SCALE))
    steps.append(_step("scale", "vision1_scaled", src="vision1", scalar=IMAGE_SCALE))

    # --- state: normalize real channels, widen to 32, project to 960 --------
    # Normalizing into a narrow buffer and packing separately is deliberate.
    # Normalizing a 32-wide buffer directly would apply the statistics to the
    # padding slots too, turning structural zeros into -mean/std.
    steps.append(_step("normalize", "state_normalized", src="state",
                       mean="state_mean", std="state_std"))
    steps.append(_step("pack", "state_lane",
                       parts=[{"src": "state_normalized", "dst_offset": 0, "count": state_dim}]))
    steps.append(_step("matvec", "state_embedding", src="state_lane",
                       weights="state_project_weight", bias="state_project_bias",
                       rows=HIDDEN, cols=ACTION_LANE))

    # --- the 241-token prefix ---------------------------------------------
    steps.append(_step(
        "pack", "prefix_embeddings",
        parts=[
            {"src": "vision0_scaled", "dst_offset": 0, "count": VISION_ELEMENTS},
            {"src": "vision1_scaled", "dst_offset": VISION_ELEMENTS, "count": VISION_ELEMENTS},
            {"src": "empty_image_embedding", "dst_offset": 2 * VISION_ELEMENTS,
             "count": VISION_ELEMENTS, "sidecar": True},
            {"src": "language_embedding", "dst_offset": 3 * VISION_ELEMENTS,
             "count": LANGUAGE_ELEMENTS, "sidecar": True},
            {"src": "state_embedding", "dst_offset": STATE_TOKEN_INDEX * HIDDEN,
             "count": HIDDEN},
        ],
    ))
    steps.append(_step("run_elf", "cache", graph="prefix", **{"in": ["prefix_embeddings"]}))

    # --- flow matching, unrolled -------------------------------------------
    steps.append(_step("slice", "x", src="noise", count=NOISE_ELEMENTS))
    for index in range(DENOISE_STEPS):
        timestep = 1.0 - index / DENOISE_STEPS
        time_buffer = f"time_{index:02d}"
        steps.append(_step("sincos_time", time_buffer, scalar=timestep,
                           min_period=TIME_MIN_PERIOD, max_period=TIME_MAX_PERIOD))
        # Per token: 32 action values, then the shared 720-wide time embedding.
        steps.append(_step("gather_strided", "suffix_input", src="x",
                           stride=ACTION_LANE, take=ACTION_LANE,
                           dst_stride=SUFFIX_TOKEN, dst_offset=0, count=CHUNK, clear=True))
        steps.append(_step("gather_strided", "suffix_input", src=time_buffer,
                           stride=0, take=EXPERT_HIDDEN,
                           dst_stride=SUFFIX_TOKEN, dst_offset=ACTION_LANE, count=CHUNK))
        steps.append(_step("run_elf", "suffix_output", graph="suffix",
                           **{"in": ["suffix_input"]}))
        steps.append(_step("pack", "denoise_input", parts=[
            {"src": "cache", "dst_offset": 0, "count": CACHE_ELEMENTS},
            {"src": "suffix_output", "dst_offset": DENOISE_SUFFIX_OFFSET,
             "count": SUFFIX_OUT_ELEMENTS},
        ]))
        steps.append(_step("run_elf", "velocity", graph="denoise",
                           **{"in": ["denoise_input"]}))
        steps.append(_step("euler", "x", src="velocity", scalar=DT))

    # --- 32-wide lane -> real robot channels, then denormalize --------------
    steps.append(_step("gather_strided", "actions_lane", src="x",
                       stride=ACTION_LANE, take=action_dim, count=CHUNK))
    steps.append(_step("denormalize", "actions", src="actions_lane",
                       mean="action_mean", std="action_std"))
    return steps


#: Constants shipped in the bundle's constants/ directory and read by name.
SIDECARS = (
    "empty_image_embedding",
    "language_embedding",
    "state_project_weight",
    "state_project_bias",
    "state_mean",
    "state_std",
    "action_mean",
    "action_std",
)


def build_plan(spec: PolicySpec, context: Any = None) -> RuntimePlan:
    state_dim = spec.dataset.state_dim
    action_dim = spec.dataset.action_dim
    return RuntimePlan(
        buffers=buffers(state_dim, action_dim),
        steps=tuple(build_steps(state_dim, action_dim)),
        result="actions",
        sidecars=SIDECARS,
    )
