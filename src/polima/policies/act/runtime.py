"""ACT's on-device execution plan.

This function emits, as data, exactly what
ACT/devkit/act_llima/act_llima.cpp::ActModel::predict() does in hand-written C++
(lines 124-147):

    camera0 = vision.run(image0)          -> run_elf
    camera1 = vision.run(image1)          -> run_elf
    packed  = zeros(601 * 512)            -> pack
      packed[0:state_dim]      = state
      packed[512:512+153600]   = camera0
      packed[301*512 : ...]    = camera1
    hidden  = stem.run(packed)            -> run_elf
    hidden  = encoder{1,2,3}.run(hidden)  -> run_elf x3
    padded  = decoder.run(hidden)         -> run_elf
    for step in 0..99:                    -> gather_strided
        actions[step*D : +D] = padded[step*16 : +D]

The offsets are not arbitrary: the stem graph reshapes its input to (1, 601, 512)
and slices the profile-selected 6 or 7 state lanes, `[:, 1:301]` for camera 0,
and `[:, 301:601]`
for camera 1 (see EncoderStemPacked in ACT/scripts/export_act_modalix.py:125-129),
so token 0 holds the state and each camera occupies 300 tokens of 512.

The 16-wide action head is also deliberate: DecoderActionRank4 widens the
profile's 6- or 7-DoF head for MLA channel alignment and zero-fills the rest
(export_act_modalix.py:145-149), so the host strides past 10 pad values per step.
"""

from __future__ import annotations

from typing import Any

from polima.policies.base import PolicySpec, RuntimePlan, Step

HIDDEN = 512
CAMERA_TOKENS = 300
STEM_TOKENS = 601                    # latent + state + 2 x 300 camera tokens
ENCODER_TOKENS = 602                 # stem output adds the latent token back
CHUNK = 100
ACTION_DIM = 6
PADDED_ACTION_DIM = 16               # widened for MLA channel alignment

IMAGE_ELEMENTS = 3 * 480 * 640       # 921600
CAMERA_ELEMENTS = CAMERA_TOKENS * HIDDEN     # 153600
STEM_ELEMENTS = STEM_TOKENS * HIDDEN         # 307712
HIDDEN_ELEMENTS = ENCODER_TOKENS * HIDDEN    # 308224
PADDED_ELEMENTS = CHUNK * PADDED_ACTION_DIM  # 1600
ACTION_ELEMENTS = CHUNK * ACTION_DIM         # 600 for the default SO-101 profile

CAMERA0_OFFSET = HIDDEN              # token 1
CAMERA1_OFFSET = 301 * HIDDEN        # token 301

ENCODER_LAYERS = ("encoder_layer_01", "encoder_layer_02", "encoder_layer_03")


def build_plan(spec: PolicySpec, context: Any = None) -> RuntimePlan:
    state_dim = spec.dataset.state_dim
    action_dim = spec.dataset.action_dim
    steps = [
        Step("run_elf", "cam0", {"graph": "vision_backbone", "in": ["image0"]}),
        Step("run_elf", "cam1", {"graph": "vision_backbone", "in": ["image1"]}),
        Step(
            "pack",
            "packed",
            {
                "size": STEM_ELEMENTS,
                "parts": [
                    {"src": "state", "dst_offset": 0, "count": state_dim},
                    {"src": "cam0", "dst_offset": CAMERA0_OFFSET, "count": CAMERA_ELEMENTS},
                    {"src": "cam1", "dst_offset": CAMERA1_OFFSET, "count": CAMERA_ELEMENTS},
                ],
            },
        ),
        Step("run_elf", "hidden", {"graph": "encoder_layer_00_stem", "in": ["packed"]}),
    ]
    steps += [
        Step("run_elf", "hidden", {"graph": name, "in": ["hidden"]}) for name in ENCODER_LAYERS
    ]
    steps += [
        Step("run_elf", "padded", {"graph": "decoder_action_tail", "in": ["hidden"]}),
        Step(
            "gather_strided",
            "actions",
            {
                "src": "padded",
                "stride": PADDED_ACTION_DIM,
                "take": action_dim,
                "count": CHUNK,
            },
        ),
    ]

    return RuntimePlan(
        buffers={
            "image0": IMAGE_ELEMENTS,
            "image1": IMAGE_ELEMENTS,
            "state": state_dim,
            "cam0": CAMERA_ELEMENTS,
            "cam1": CAMERA_ELEMENTS,
            "packed": STEM_ELEMENTS,
            "hidden": HIDDEN_ELEMENTS,
            "padded": PADDED_ELEMENTS,
            "actions": CHUNK * action_dim,
        },
        steps=tuple(steps),
        result="actions",
        # ACT normalizes on the client from normalization_stats.npz, so the
        # server needs no constants. (SmolVLA currently bakes 24 float literals
        # into its binary; Phase 4 moves those to constants/normalization.json.)
        sidecars=(),
    )
