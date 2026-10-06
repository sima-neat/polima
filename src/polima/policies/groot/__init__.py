"""GR00T N1.6 -- Eagle VLM + flow-matching DiT, third port into PoLiMa.

Every constant is transcribed from the working GR00T-N1.6 stack, not invented:

  GR00T-N1.6/devkit/groot_full_llima/groot_full_llima.cpp       pipeline, sizes
  GR00T-N1.6/scripts/export_groot_modalix_eagle.py              the Eagle cut
  GR00T-N1.6/scripts/export_groot_modalix_action.py             the action cut
  GR00T-N1.6/scripts/groot_modalix_compile_controller.py        per-graph flags
  GR00T-N1.6/train_groot_local.sh                               fine-tune, cameras
  GR00T-N1.6/config/modality.json                               SO-101 modalities

## How it differs from SmolVLA, structurally

SmolVLA cuts into four graphs. GR00T cuts into forty-five, because neither the
Eagle backbone nor the 32-block DiT fits the compiler in one piece: the VLM is
split into 26 stages and the action head into 19. That is the whole reason this
spec is generated rather than typed out.

Forty-five graphs would also mean forty-five upload/download round trips, which
is what the two `run_elf_chain` steps exist to avoid -- sixteen vision ELFs and
nine language ELFs each run device-resident behind a single transfer. The build
tree's own validation records what that is worth: Eagle fell from 362 ms to
169 ms when the chains landed, taking the full image-to-action pipeline from
652 ms to 457 ms.

## Two horizons, and which one this spec declares

The compiled artifacts in `GR00T-N1.6/outputs/modalix_groot_n1d6_base_gr1/` are
the *base* GR1 checkpoint: 50-step action horizon, embodiment id 20, 128-wide
state and action lanes. `train_groot_local.sh` fine-tunes the same architecture
on the local two-camera SO-101 dataset under `NEW_EMBODIMENT` with a 16-step
horizon (`config/so101_config.py`).

This spec declares the 50-step geometry, because that is the only decomposition
that has been exported, compiled, and measured end to end. A 16-step bundle is
the same graph shapes with `CHUNK = 16`; `runtime.CHUNK` is the single place
that changes, and every dependent size is derived from it.

## Language conditioning

Like SmolVLA, GR00T is language-conditioned, so `single_task` is False.
Export discovers every trained instruction from the dataset and precomputes one
prompt row in `constants/prompt_embeddings`. The request header's flags select
that row, so all trained tasks share one set of compiled ELFs without shipping
the 151,680-row vocabulary table. Export rejects catalogs whose processor
layouts differ; it never silently truncates or pads a prompt.
"""

from __future__ import annotations

from polima.policies.base import (
    CalibrationSource,
    CompilePlan,
    DatasetContract,
    DatasetConverter,
    GraphSpec,
    PolicySpec,
    RobotSpec,
    TensorSpec,
    TrainSpec,
    WireSpec,
)
from polima.policies.groot.runtime import (
    ACTION_DIM,
    ACTION_LANE,
    CHUNK,
    CONNECTOR_CHANNELS,
    CONNECTOR_WIDTH,
    DEFAULT_PORT,
    DENOISE_STEPS,
    HIDDEN_CHANNELS,
    HIDDEN_WIDTH,
    LANGUAGE_CHANNELS,
    MASK_ELEMENTS,
    PATCH_CHANNELS,
    SEQUENCE,
    STATE_DIM,
    STATE_LANE,
    TEMB_ELEMENTS,
    VISION_CHANNELS,
    VISION_WIDTH,
    WIRE_MAGIC,
    block_names,
    qwen_stage_names,
    vision_stage_names,
)
from polima.policies.registry import register_policy

#: SO-101 follower joints, in dataset order. Same arm as ACT and SmolVLA.
JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

#: Cameras are captured at 480x640 and resized to 252x252 by the client, which
#: also patchifies: Eagle's first ELF consumes 324 rows of 588, not an image.
CAPTURE_HEIGHT, CAPTURE_WIDTH = 480, 640


def _token_tensor(name: str, width: int, channels: int) -> TensorSpec:
    return TensorSpec(name, (1, width, channels))


def _eagle_stage(name: str, in_channels: int, out_channels: int, width: int,
                 *, samples: int = 1) -> GraphSpec:
    """One device-resident Eagle stage.

    Chained stages must expose a flat HWC boundary at both ends so one ELF's
    output buffer binds directly to the next ELF's input with no download,
    detessellation and upload in between. `promote_rank3_hwc` is what makes that
    legal for these [N, W, C] token graphs: ModelSDK 2.1's MPK path assumes four
    dimensions once an explicit HWC layout is selected.
    """
    return GraphSpec(
        name=name,
        builder=f"polima.policies.groot.graphs:{name}",
        inputs=(_token_tensor("hidden", width, in_channels),),
        outputs=(_token_tensor("output", width, out_channels),),
        layout="NHWC",
        precision="bf16",
        calibration=CalibrationSource("npz", samples=samples),
        mla_tessellation=False,
        external_dram_layout="HWC",
        promote_rank3_hwc=True,
        exit_on_stable_elf=True,
    )


def _qwen_stage(name: str) -> GraphSpec:
    """Qwen pair with explicit task-selected causal/key-padding mask."""

    return GraphSpec(
        name=name,
        builder=f"polima.policies.groot.graphs:{name}",
        inputs=(
            TensorSpec("hidden", (1, 1, SEQUENCE, LANGUAGE_CHANNELS)),
            TensorSpec("attention_mask", (1, SEQUENCE, SEQUENCE, 1)),
        ),
        outputs=(TensorSpec("output", (1, 1, SEQUENCE, LANGUAGE_CHANNELS)),),
        layout="NHWC",
        precision="bf16",
        calibration=CalibrationSource("npz", samples=1),
        mla_tessellation=False,
        external_dram_layout="HWC",
        exit_on_stable_elf=True,
    )


def _eagle_graphs() -> tuple[GraphSpec, ...]:
    graphs = [
        # Patchified pixels in, SigLIP embeddings out. The 588 input channels
        # are 14x14x3, which is why the compiler warns about Preproc here.
        _eagle_stage("eagle_vision_patch", PATCH_CHANNELS, VISION_CHANNELS,
                     VISION_WIDTH),
    ]
    graphs += [
        _eagle_stage(name, VISION_CHANNELS, VISION_CHANNELS, VISION_WIDTH)
        for name in vision_stage_names()
    ]
    graphs.append(
        _eagle_stage("eagle_vision_post_norm", VISION_CHANNELS, VISION_CHANNELS,
                     VISION_WIDTH)
    )
    # The connector breaks the chain: the host folds the vision grid into
    # C*4 channels between post_norm and here. ModelSDK 2.1.3 accepts the
    # tessellated input boundary but the raw connector ELF returns ordinary
    # contiguous BF16, not HWC16. This was confirmed on Modalix with the
    # MyCobot fixture: plain decoding matched the ONNX range while HWC16
    # byte-plane decoding manufactured non-finite values.
    graphs.append(GraphSpec(
        name="eagle_vision_connector",
        builder="polima.policies.groot.graphs:eagle_vision_connector",
        inputs=(_token_tensor("hidden", CONNECTOR_WIDTH, CONNECTOR_CHANNELS),),
        outputs=(_token_tensor("output", CONNECTOR_WIDTH, LANGUAGE_CHANNELS),),
        layout="NHWC",
        precision="bf16",
        calibration=CalibrationSource("npz", samples=1),
        exit_on_stable_elf=True,
    ))
    graphs += [_qwen_stage(name) for name in qwen_stage_names()]
    graphs.append(
        _eagle_stage("eagle_output_norm", LANGUAGE_CHANNELS, LANGUAGE_CHANNELS,
                     SEQUENCE)
    )
    return tuple(graphs)


def _action_graphs() -> tuple[GraphSpec, ...]:
    #: Calibration covers the four denoise steps, which is the whole range these
    #: graphs ever see: tau is bucketed at 0/250/500/750 and nothing else.
    calibration = CalibrationSource("npz", samples=DENOISE_STEPS)
    graphs = [
        # Padded, normalized joint positions -> the DiT's first token.
        GraphSpec(
            name="state_project",
            builder="polima.policies.groot.graphs:state_project",
            inputs=(TensorSpec("state", (1, 1, STATE_LANE)),),
            outputs=(_token_tensor("state_features", 1, HIDDEN_CHANNELS),),
            layout="NHWC",
            precision="bf16",
            calibration=CalibrationSource("npz", samples=1),
        ),
        # The action lane plus this step's tau encoding. Runs 4x.
        GraphSpec(
            name="action_project",
            builder="polima.policies.groot.graphs:action_project",
            inputs=(
                TensorSpec("actions", (1, CHUNK, ACTION_LANE)),
                TensorSpec("tau_embedding", (1, CHUNK, HIDDEN_CHANNELS)),
            ),
            outputs=(_token_tensor("action_features", CHUNK, HIDDEN_CHANNELS),),
            layout="NHWC",
            precision="bf16",
            calibration=calibration,
        ),
    ]
    # Sixteen cross/self block pairs. Each reads the full backbone and one of
    # the two attention masks; the runtime alternates them by pair parity.
    graphs += [
        GraphSpec(
            name=name,
            builder=f"polima.policies.groot.graphs:{name}",
            inputs=(
                TensorSpec("hidden", (1, HIDDEN_WIDTH, HIDDEN_CHANNELS)),
                TensorSpec("temb", (1, TEMB_ELEMENTS)),
                TensorSpec("backbone_features", (1, SEQUENCE, LANGUAGE_CHANNELS)),
                TensorSpec("additive_mask", (1, MASK_ELEMENTS)),
            ),
            outputs=(_token_tensor("hidden_out", HIDDEN_WIDTH, HIDDEN_CHANNELS),),
            layout="NHWC",
            precision="bf16",
            calibration=calibration,
            exit_on_stable_elf=True,
        )
        for name in block_names()
    ]
    graphs.append(GraphSpec(
        name="action_tail",
        builder="polima.policies.groot.graphs:action_tail",
        inputs=(
            TensorSpec("hidden", (1, HIDDEN_WIDTH, HIDDEN_CHANNELS)),
            TensorSpec("temb", (1, TEMB_ELEMENTS)),
        ),
        outputs=(_token_tensor("velocity", CHUNK, ACTION_LANE),),
        layout="NHWC",
        precision="bf16",
        calibration=calibration,
    ))
    return tuple(graphs)


GROOT_SPEC = PolicySpec(
    name="groot",
    display_name="GR00T N1.6 (Eagle VLM + flow-matching DiT)",
    # --------------------------------------------------------------- dataset --
    dataset=DatasetContract(
        state_names=JOINT_NAMES,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        camera_keys=("observation.images.overhead", "observation.images.wrist"),
        camera_shape=(CAPTURE_HEIGHT, CAPTURE_WIDTH, 3),
        fps=30,
        codebase_version="v3.0",
        single_task=False,
        task_canonicalizer="polima.data.contract:canonical_task",
        # GR00T addresses cameras by modality name, not by LeRobot feature key.
        # config/modality.json is what performs this rename during conversion.
        rename_map={
            "observation.images.overhead": "front",
            "observation.images.wrist": "wrist",
        },
        # GR00T trains on LeRobot v2.1, which v3.0 datasets have to be
        # downgraded into -- in a second environment, because the converter and
        # the trainer need incompatible lerobot versions.
        converter=DatasetConverter(
            name="lerobot-v3-to-v2.1",
            conda_env="groot-n1.6-convert",
            entrypoint=("scripts/lerobot_conversion/convert_v3_to_v2.py",),
            target_codebase_version="v2.1",
            # modality.json lands in the *converted* v2.1 tree, which is why it
            # is post_copy here and not extra_meta_files: requiring it of the
            # v3.0 source would reject every dataset before conversion runs.
            post_copy=(("config/mycobot_modality.json", "meta/modality.json"),),
        ),
    ),
    # ------------------------------------------------- checkpoint provenance --
    train=TrainSpec(
        backend="groot-launch-finetune",
        conda_env="groot-n1.6",
        repo_dir_hint="GR00T-N1.6/Isaac-GR00T",
        entrypoint=("gr00t/experiment/launch_finetune.py",),
        build_args="polima.policies.groot.train:build_args",
        defaults={
            "steps": 20_000,
            "batch_size": 8,
            "embodiment_tag": "NEW_EMBODIMENT",
            "action_horizon": 16,
            "tune_projector": True,
            "tune_diffusion_model": True,
            "tune_llm": False,
            "tune_visual": False,
        },
        augmentation_tfs=(
            '{"brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08}'
        ),
        checkpoint_glob="checkpoint-*",
    ),
    # --------------------------------------------------------------- compile --
    compile=CompilePlan(
        export_entry="polima.policies.groot.graphs:export_all",
        verify_entry="polima.policies.groot.graphs:verify_chain",
        fixture_entry="polima.policies.groot.graphs:write_fixtures",
        normalization_entry="polima.policies.groot.graphs:write_normalization",
        fixture_file="groot_fixture.npz",
        # The Eagle stack accumulates BF16 error over 26 stages; the build
        # tree's own report records cosine 0.9977 against the BF16 reference,
        # so the chain tolerance is looser than SmolVLA's.
        verify_atol=1e-2,
        verify_rtol=1e-2,
        graphs=(*_eagle_graphs(), *_action_graphs()),
    ),
    # ------------------------------------------------------------------ wire --
    wire=WireSpec(
        magic=WIRE_MAGIC,                 # "GRUT" little-endian
        version=1,
        default_port=DEFAULT_PORT,
        request_header="<IIII",
        response_header="<IIIIfI",
        request_tensors=(
            # Resized to 252x252, normalized, and patchified by the live client:
            # 324 patches of 14x14x3. Doing it host-side keeps an image reshape
            # off the board and out of the opcode set.
            TensorSpec("patches", (VISION_WIDTH, PATCH_CHANNELS)),
            TensorSpec("state", (STATE_DIM,)),
            # The flow-matching seed. Supplied by the client rather than drawn
            # on the board so a run is reproducible from its request alone.
            TensorSpec("noise", (CHUNK, ACTION_LANE)),
        ),
        response_shape=(CHUNK, ACTION_DIM),
        # Server-side, as for SmolVLA: the board already holds the statistics
        # because the state lane is normalized before it is projected.
        normalization_side="server",
        stats_file="normalization_stats.npz",
    ),
    # ----------------------------------------------------------------- robot --
    robot=RobotSpec(
        camera_roles=(("overhead", "Overhead"), ("wrist", "Wrist")),
        camera_hints={"overhead": "C920", "wrist": "Sonix"},
        joint_names=JOINT_NAMES,
        actions_per_chunk=CHUNK,
        default_fps=30,
        max_relative_target=12,
        aggregate_fn="weighted_average",
        # Short 16-action plans: always request the next one (20 trials, 11 picks).
        chunk_threshold=1.0,
        image_preprocessor="polima.policies.groot.graphs:patchify_for_wire",
    ),
    runtime_plan_builder="polima.policies.groot.runtime:build_plan",
    checkpoint_validator="polima.policies.groot.graphs:validate_checkpoint",
    profile_entry="polima.policies.groot.profile:specialize_spec",
)

register_policy(GROOT_SPEC)

__all__ = ["GROOT_SPEC", "JOINT_NAMES"]
