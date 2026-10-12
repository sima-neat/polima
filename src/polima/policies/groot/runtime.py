"""GR00T N1.6's on-device pipeline, as plan.json.

Every constant here is transcribed from the working DevKit binaries in the
GR00T-N1.6 tree, which are what currently run this policy on Modalix:

    GR00T-N1.6/devkit/groot_eagle_llima/groot_eagle_llima.cpp    Eagle geometry
    GR00T-N1.6/devkit/groot_action_llima/groot_action_llima.cpp  action geometry
    GR00T-N1.6/devkit/groot_full_llima/groot_full_llima.cpp      the joined pipeline
    GR00T-N1.6/scripts/export_groot_modalix_eagle.py             the Eagle cut
    GR00T-N1.6/scripts/export_groot_modalix_action.py            the action cut

## The pipeline

    patches(324x588)                                     } one device-resident
      -> eagle_vision_patch                              } chain: 16 ELFs, one
      -> eagle_vision_00_01 .. 26_26   (14 pairs)        } upload, one download
      -> eagle_vision_post_norm                          }
    pixel_unshuffle 18x18x1152 -> 81x4608
      -> eagle_vision_connector                          -> 81x2048 image tokens
    pack 116x2048: prompt embedding, image tokens at token 32
      -> eagle_qwen_00_01 .. 14_15 (8 pairs)             } second chain: 9 ELFs
      -> eagle_output_norm                               } -> backbone features

    state(6) -> normalize -> pad to 128 -> state_project -> 1x1536
    x = noise (50x128)
    repeat 4 times, tau = 0, 250, 500, 750:
        action_project( [x | tau_embedding] )  -> 50x1536
        hidden = [state_features | action_features]      -> 51x1536
        for pair in 0..15:
            dit_blocks_2p_2p+1( hidden, temb, backbone, mask )
        action_tail( hidden, temb )                      -> velocity 50x128
        x += 0.25 * velocity
    actions = denormalize(gather x[t*128 .. +6])         -> 50x6

Four things that look arbitrary and are not:

* **The alternating attention mask.** Even pairs attend to text, odd pairs to
  images. `export_groot_modalix_action.py` selects `text_additive if start % 4
  == 0 else image_additive` over block starts, which is pair parity. Feeding one
  mask to every block does not error; it silently attends to the wrong half of
  the prompt.
* **The 128-wide state and action lanes carrying 6 real joints.** GR00T pads
  every embodiment to its maximum state/action width. Positions 6..127 are zero
  going in and ignored coming out.
* **The 81 image tokens land at sequence position 32.** The GR1 prompt places a
  contiguous run of `<image>` tokens there, which is what lets the language
  embedding be a `pack` of one constant plus one live buffer instead of a
  gather against a 151,680-row vocabulary table.
* **The Euler update adds where SmolVLA's subtracts.** GR00T integrates
  `x += (1/N) * velocity`; the `euler` opcode computes `out - scalar * src`, so
  the plan carries a negative DT. See `EULER_SCALAR`.

## Why the loop is unrolled

Four denoise steps x (1 + 16 + 1) ELF calls are emitted as explicit steps rather
than a loop opcode, exactly as SmolVLA's ten are: the interpreter needs no
control flow, each step's tau is a literal, and `--verbose` reports which of the
seventy-odd MLA calls is slow.
"""

from __future__ import annotations

from typing import Any

from polima.policies.base import PolicySpec, RuntimePlan, Step

# ------------------------------------------------------- Eagle geometry

#: GR1's fixed image contract: one 252x252 frame, 14-pixel patches, 18x18 grid.
IMAGE_SIDE = 252
PATCH = 14
PATCH_SIDE = IMAGE_SIDE // PATCH                         # 18
VISION_WIDTH = PATCH_SIDE * PATCH_SIDE                   # 324 patch tokens
PATCH_CHANNELS = PATCH * PATCH * 3                       # 588
VISION_CHANNELS = 1152                                   # SigLIP hidden width
VISION_PAIRS = 14                                        # 27 layers, paired

#: pixel_unshuffle(2) folds the 18x18 grid into 9x9 and quadruples the channels.
UNSHUFFLE = 2
CONNECTOR_WIDTH = (PATCH_SIDE // UNSHUFFLE) ** 2         # 81 image tokens
CONNECTOR_CHANNELS = VISION_CHANNELS * UNSHUFFLE ** 2    # 4608

SEQUENCE = 116                                           # prompt + image tokens
LANGUAGE_CHANNELS = 2048                                 # Qwen hidden width
QWEN_PAIRS = 8                                           # 16 layers, paired

#: Where the contiguous run of 81 <image> tokens begins in the GR1 prompt.
#: Verified against eagle_fixture.npz: input_ids == 151669 at 32..112.
IMAGE_TOKEN_START = 32

PATCH_ELEMENTS = VISION_WIDTH * PATCH_CHANNELS           # 190512
VISION_ELEMENTS = VISION_WIDTH * VISION_CHANNELS         # 373248
CONNECTOR_ELEMENTS = CONNECTOR_WIDTH * CONNECTOR_CHANNELS        # 373248
IMAGE_TOKEN_ELEMENTS = CONNECTOR_WIDTH * LANGUAGE_CHANNELS       # 165888
BACKBONE_ELEMENTS = SEQUENCE * LANGUAGE_CHANNELS                 # 237568
MASK_ELEMENTS = SEQUENCE                                         # 116

# ------------------------------------------------------ action geometry

STATE_DIM = 6                                            # real SO-101 joints
ACTION_DIM = 6
STATE_LANE = 128                                         # GR00T's padded width
ACTION_LANE = 128
CHUNK = 50                                               # action horizon

HIDDEN_WIDTH = 1 + CHUNK                                 # 51: state + actions
HIDDEN_CHANNELS = 1536                                   # DiT width
TEMB_ELEMENTS = HIDDEN_CHANNELS                          # 1536
BLOCK_PAIRS = 16                                         # 32 DiT blocks, paired

STATE_ELEMENTS = 1 * STATE_LANE                          # 128
ACTION_ELEMENTS = CHUNK * ACTION_LANE                    # 6400
TAU_ELEMENTS = CHUNK * HIDDEN_CHANNELS                   # 76800
ACTION_INPUT_ELEMENTS = ACTION_ELEMENTS + TAU_ELEMENTS   # 83200
HIDDEN_ELEMENTS = HIDDEN_WIDTH * HIDDEN_CHANNELS         # 78336

#: The fixed four-step Euler schedule. GR00T buckets tau at 0/250/500/750 and
#: the host precomputes both sinusoidal encodings, so nothing on the board has
#: to cast or expand a scalar -- Model Compiler cannot place those on Modalix.
DENOISE_STEPS = 4
TIMESTEP_BUCKETS = (0, 250, 500, 750)
DT = 1.0 / DENOISE_STEPS                                 # 0.25

#: `euler` computes `out - scalar * src`. GR00T's update is `x += DT * v`, so
#: the scalar is negated here rather than a second opcode being added.
EULER_SCALAR = -DT

RESPONSE_ELEMENTS = CHUNK * ACTION_DIM                   # 300

WIRE_MAGIC = 0x54555247                                  # "GRUT"
DEFAULT_PORT = 8093


def two_digits(value: int) -> str:
    return f"{value:02d}"


def vision_stage_names() -> tuple[str, ...]:
    """The 27 SigLIP layers, paired -- the last one is odd and stands alone.

    `export_groot_modalix_eagle.py` names a pair by its first and last layer,
    so the tail stage is `eagle_vision_26_26`, not `eagle_vision_26_27`.
    """
    names = []
    for start in range(0, 2 * VISION_PAIRS, 2):
        end = min(start + 2, 27)
        names.append(f"eagle_vision_{two_digits(start)}_{two_digits(end - 1)}")
    return tuple(names)


def qwen_stage_names() -> tuple[str, ...]:
    return tuple(
        f"eagle_qwen_{two_digits(start)}_{two_digits(start + 1)}"
        for start in range(0, 2 * QWEN_PAIRS, 2)
    )


def block_names() -> tuple[str, ...]:
    return tuple(
        f"dit_blocks_{two_digits(start)}_{two_digits(start + 1)}"
        for start in range(0, 2 * BLOCK_PAIRS, 2)
    )


#: The two device-resident chains. One upload and one download each, with every
#: intermediate staying in MLA DRAM -- this is what took Eagle from 362 ms to
#: 169 ms in the GR00T-N1.6 build tree's own validation runs.
VISION_CHAIN = ("eagle_vision_patch", *vision_stage_names(), "eagle_vision_post_norm")
QWEN_CHAIN = (*qwen_stage_names(), "eagle_output_norm")


# -------------------------------------------------------------- buffers


def _profile(spec: PolicySpec | None):
    """Return real robot/output dimensions and traced Eagle geometry."""
    if spec is None:
        return STATE_DIM, ACTION_DIM, CHUNK, SEQUENCE, (IMAGE_TOKEN_START,), 0, ("patches",)
    state_dim = spec.dataset.state_dim
    action_dim = spec.dataset.action_dim
    horizon = spec.robot.actions_per_chunk
    context = dict(spec.runtime_context)
    sequence = int(context.get("sequence", SEQUENCE))
    starts = tuple(int(value) for value in context.get("image_token_starts", (IMAGE_TOKEN_START,)))
    relative_dim = int(context.get("relative_action_dim", 0))
    patch_names = tuple(
        tensor.name for tensor in spec.wire.request_tensors
        if tensor.name.startswith("patches")
    )
    if len(starts) != len(patch_names):
        raise ValueError(
            f"GR00T runtime has {len(patch_names)} camera tensors but token windows {starts}"
        )
    return state_dim, action_dim, horizon, sequence, starts, relative_dim, patch_names


def _task_count(spec: PolicySpec | None) -> int:
    if spec is None:
        return 0
    catalog = list(dict(spec.runtime_context).get("task_catalog") or ())
    if catalog and [int(task.get("id", -1)) for task in catalog] != list(range(len(catalog))):
        raise ValueError(f"invalid GR00T runtime task catalog: {catalog}")
    return len(catalog)


def _vision_geometry(spec: PolicySpec | None) -> tuple[int, int, int, int]:
    """Processed patch rows/columns and their pre/post-unshuffle widths."""
    context = {} if spec is None else dict(spec.runtime_context)
    patch_grid = tuple(
        int(value) for value in context.get("patch_grid", (PATCH_SIDE, PATCH_SIDE))
    )
    if len(patch_grid) != 2:
        raise ValueError(f"invalid GR00T patch grid {patch_grid}")
    rows, columns = patch_grid
    if (
        rows <= 0
        or columns <= 0
        or rows % UNSHUFFLE
        or columns % UNSHUFFLE
    ):
        raise ValueError(
            f"GR00T patch grid {patch_grid} must be positive and divisible by "
            f"{UNSHUFFLE}"
        )
    vision_width = rows * columns
    connector_width = (rows // UNSHUFFLE) * (columns // UNSHUFFLE)
    recorded = int(context.get("connector_width", connector_width))
    if recorded != connector_width:
        raise ValueError(
            f"GR00T connector width {recorded} disagrees with patch grid {patch_grid}"
        )
    return rows, columns, vision_width, connector_width


def _vision_slots(patch_names: tuple[str, ...]):
    if patch_names == ("patches",):
        return (("patches", "vision_hidden", "connector_input", "image_tokens"),)
    return tuple(
        (name, f"vision_hidden_{index}", f"connector_input_{index}", f"image_tokens_{index}")
        for index, name in enumerate(patch_names)
    )


def buffers(spec: PolicySpec | None = None) -> dict[str, int]:
    """Every buffer the plan touches, allocated once at load."""
    state_dim, action_dim, horizon, sequence, _, relative_dim, patch_names = _profile(spec)
    backbone_elements = sequence * LANGUAGE_CHANNELS
    response_elements = horizon * action_dim
    _, _, vision_width, connector_width = _vision_geometry(spec)
    patch_elements = vision_width * PATCH_CHANNELS
    vision_elements = vision_width * VISION_CHANNELS
    connector_elements = connector_width * CONNECTOR_CHANNELS
    image_token_elements = connector_width * LANGUAGE_CHANNELS
    sizes = {
        "state": state_dim,
        "noise": ACTION_ELEMENTS,
        "language_embeddings": backbone_elements,
        "backbone_features": backbone_elements,
        "text_mask": sequence,
        "image_mask": sequence,
        "qwen_mask": sequence * sequence,
        "state_normalized": state_dim,
        "state_clipped": state_dim,
        "state_lane": STATE_LANE,
        "state_features": HIDDEN_CHANNELS,
        "actions": ACTION_ELEMENTS,
        "action_input": ACTION_INPUT_ELEMENTS,
        "action_features": TAU_ELEMENTS,
        "hidden_a": HIDDEN_ELEMENTS,
        "hidden_b": HIDDEN_ELEMENTS,
        "velocity": ACTION_ELEMENTS,
        "actions_lane": response_elements,
        "actions_clipped": response_elements,
        "action_chunk": response_elements,
    }
    for patches, vision, connector, tokens in _vision_slots(patch_names):
        sizes[patches] = patch_elements
        sizes[vision] = vision_elements
        sizes[connector] = connector_elements
        sizes[tokens] = image_token_elements
    if relative_dim:
        sizes["reference_chunk"] = response_elements
    for step in range(DENOISE_STEPS):
        sizes[f"temb_{step:02d}"] = TEMB_ELEMENTS
    return sizes


# ---------------------------------------------------------------- steps


def _step(op: str, out: str, **args) -> Step:
    return Step(op, out, args)


def _from_sidecar(name: str, sidecar: str, count: int) -> Step:
    """Materialise one constant into a buffer.

    `run_elf` reads buffers, not sidecars, and the masks and time embeddings are
    each read 64 times per inference. Copying them once at the top of the plan
    keeps the hot path free of special cases.
    """
    return _step("pack", name,
                 parts=[{"src": sidecar, "dst_offset": 0, "count": count, "sidecar": True}])


def build_steps(spec: PolicySpec | None = None) -> list[Step]:
    state_dim, action_dim, horizon, sequence, starts, relative_dim, patch_names = _profile(spec)
    backbone_elements = sequence * LANGUAGE_CHANNELS
    patch_rows, patch_columns, _, connector_width = _vision_geometry(spec)
    image_token_elements = connector_width * LANGUAGE_CHANNELS
    steps: list[Step] = []

    # --- constants into buffers -------------------------------------------
    task_count = _task_count(spec)
    if task_count:
        steps.append(_step("pack", "text_mask", parts=[{
            "src": "text_additive_masks",
            "dst_offset": 0,
            "count": sequence,
            "sidecar": True,
            "source_stride": sequence,
            "selector_count": task_count,
        }]))
        steps.append(_step("pack", "qwen_mask", parts=[{
            "src": "qwen_attention_masks",
            "dst_offset": 0,
            "count": sequence * sequence,
            "sidecar": True,
            "source_stride": sequence * sequence,
            "selector_count": task_count,
        }]))
    else:
        steps.append(_from_sidecar("text_mask", "text_additive_mask", sequence))
        steps.append(_from_sidecar(
            "qwen_mask", "qwen_attention_mask", sequence * sequence
        ))
    steps.append(_from_sidecar("image_mask", "image_additive_mask", sequence))
    for index in range(DENOISE_STEPS):
        steps.append(_from_sidecar(
            f"temb_{index:02d}", f"timestep_embedding_{index}", TEMB_ELEMENTS))

    # Eagle's weights are shared: run the same compiled vision chain once per
    # camera, preserving processor camera order.
    slots = _vision_slots(patch_names)
    for patches, vision, connector, tokens in slots:
        steps.append(_step("run_elf_chain", vision,
                           graphs=list(VISION_CHAIN), **{"in": [patches]}))
        steps.append(_step("pixel_unshuffle", connector, src=vision,
                           height=patch_rows, width=patch_columns,
                           channels=VISION_CHANNELS, factor=UNSHUFFLE))
        steps.append(_step("run_elf", tokens, graph="eagle_vision_connector",
                           **{"in": [connector]}))

    prompt_part = {
        "src": "prompt_embeddings" if task_count else "prompt_embedding",
        "dst_offset": 0,
        "count": backbone_elements,
        "sidecar": True,
    }
    if task_count:
        # The server assigns request-header flags to the catalog id. Selection
        # happens while packing, without another tensor or host-side copy.
        prompt_part.update({
            "source_stride": backbone_elements,
            "selector_count": task_count,
        })
    parts = [prompt_part]
    for start, (_, _, _, tokens) in zip(starts, slots, strict=True):
        parts.append({"src": tokens, "dst_offset": start * LANGUAGE_CHANNELS,
                      "count": image_token_elements})
    steps.append(_step("pack", "language_embeddings", parts=parts))
    qwen_input = "language_embeddings"
    qwen_output = "backbone_features"
    for graph in qwen_stage_names():
        steps.append(_step(
            "run_elf", qwen_output, graph=graph,
            **{"in": [qwen_input, "qwen_mask"]},
        ))
        qwen_input, qwen_output = qwen_output, qwen_input
    steps.append(_step(
        "run_elf", "backbone_features", graph="eagle_output_norm",
        **{"in": [qwen_input]},
    ))

    state_encoding = spec.runtime_context.get("state_encoding") if spec else None
    if state_encoding:
        widths = state_encoding["group_widths"]
        if (state_encoding["kind"] != "grouped_sincos" or not widths
                or any(w <= 0 for w in widths) or sum(widths) != state_dim
                or 2 * state_dim > 128):
            raise ValueError("unsupported GR00T state encoding geometry")
        steps.append(_step("sincos_state", "state_lane", src="state", group_widths=widths))
    else:
        steps.append(_step("normalize", "state_normalized", src="state",
                           mean="state_mean", std="state_std"))
        steps.append(_step("clip_unit", "state_clipped", src="state_normalized"))
        steps.append(_step("pack", "state_lane",
                           parts=[{"src": "state_clipped", "dst_offset": 0,
                                   "count": state_dim}]))
    steps.append(_step("run_elf", "state_features", graph="state_project",
                       **{"in": ["state_lane"]}))

    # --- flow matching, unrolled ------------------------------------------
    steps.append(_step("slice", "actions", src="noise", count=ACTION_ELEMENTS))
    for index in range(DENOISE_STEPS):
        temb = f"temb_{index:02d}"
        steps.append(_step("pack", "action_input", parts=[
            {"src": "actions", "dst_offset": 0, "count": ACTION_ELEMENTS},
            {"src": f"tau_embedding_{index}", "dst_offset": ACTION_ELEMENTS,
             "count": TAU_ELEMENTS, "sidecar": True},
        ]))
        steps.append(_step("run_elf", "action_features", graph="action_project",
                           **{"in": ["action_input"]}))
        steps.append(_step("pack", "hidden_a", parts=[
            {"src": "state_features", "dst_offset": 0, "count": HIDDEN_CHANNELS},
            {"src": "action_features", "dst_offset": HIDDEN_CHANNELS,
             "count": TAU_ELEMENTS},
        ]))
        # Ping-pong so no ELF reads and writes the same buffer. Sixteen pairs is
        # even, so the last block lands back in hidden_a for the tail.
        for pair, name in enumerate(block_names()):
            source = "hidden_a" if pair % 2 == 0 else "hidden_b"
            target = "hidden_b" if pair % 2 == 0 else "hidden_a"
            mask = "text_mask" if pair % 2 == 0 else "image_mask"
            steps.append(_step("run_elf", target, graph=name,
                               **{"in": [source, temb, "backbone_features", mask]}))
        steps.append(_step("run_elf", "velocity", graph="action_tail",
                           **{"in": ["hidden_a", temb]}))
        steps.append(_step("euler", "actions", src="velocity", scalar=EULER_SCALAR))

    # The model lane remains 50x128, but the saved processor decodes only its
    # first horizon x real-action-width block (16x7 for MyCobot).
    steps.append(_step("gather_strided", "actions_lane", src="actions",
                       stride=ACTION_LANE, take=action_dim, count=horizon))
    steps.append(_step("clip_unit", "actions_clipped", src="actions_lane"))
    steps.append(_step("denormalize", "action_chunk", src="actions_clipped",
                       mean="action_mean", std="action_std"))
    if relative_dim:
        steps.append(_step("gather_strided", "reference_chunk", src="state",
                           stride=0, take=relative_dim, count=horizon,
                           dst_stride=action_dim, clear=True))
        # euler computes out - scalar*src; -1 adds the raw state reference.
        steps.append(_step("euler", "action_chunk", src="reference_chunk", scalar=-1.0))
    return steps


#: Constants shipped in the bundle's constants/ directory and read by name.
SIDECARS = (
    "prompt_embedding",
    "image_additive_mask",
    "text_additive_mask",
    "qwen_attention_mask",
    *(f"tau_embedding_{index}" for index in range(DENOISE_STEPS)),
    *(f"timestep_embedding_{index}" for index in range(DENOISE_STEPS)),
    "state_mean",
    "state_std",
    "action_mean",
    "action_std",
)


def build_plan(spec: PolicySpec, context: Any = None) -> RuntimePlan:
    sidecars = SIDECARS
    if _task_count(spec):
        sidecars = (
            "prompt_embeddings",
            "image_additive_mask",
            "text_additive_masks",
            "qwen_attention_masks",
            *SIDECARS[4:],
        )
    return RuntimePlan(
        buffers=buffers(spec),
        steps=tuple(build_steps(spec)),
        result="action_chunk",
        sidecars=sidecars,
    )
