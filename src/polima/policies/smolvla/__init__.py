"""SmolVLA -- vision-language-action policy, second port into PoLiMa.

Every constant is transcribed from the working legacy stack, not invented:

  SmolVLA/devkit/smolvla_som_server/smolvla_som_server.cpp   pipeline, wire, sizes
  SmolVLA/scripts/compile_deploy_smolvla_som.sh              per-graph compile flags
  SmolVLA/scripts/smolvla_som_client.py                      client-side protocol
  SmolVLA/scripts/extract_smolvla_som_constants.py           the constants/ sidecars
  SmolVLA/inference_smolvla.sh                               robot launch, task string

## How it differs from ACT, structurally

ACT is a fixed feed-forward chain: six graphs, one pass, done. SmolVLA is a
*flow-matching* policy, so the action is produced by integrating a learned
velocity field over ten Euler steps. Two of its four graphs therefore run ten
times per inference, not once -- twenty MLA calls in total against ACT's six.

That is why PoLiMa's runtime is a plan interpreter rather than a fixed pipeline:
the loop, the time embedding and the Euler update are all data in plan.json.
See `polima.policies.smolvla.runtime`.

## Language conditioning

SmolVLA is language-conditioned, so `single_task` is False -- unlike ACT, a
dataset with several task strings is legitimate. The language embedding itself is
*precomputed* into `constants/language_embedding.f32` at export: the task string
is fixed for a deployed checkpoint, so tokenizing and embedding it on every
inference would burn latency on a constant. Changing the task means re-exporting.
"""

from __future__ import annotations

from polima.policies.base import (
    CalibrationSource,
    CompilePlan,
    DatasetContract,
    GraphSpec,
    PolicySpec,
    RobotSpec,
    SmokeSpec,
    TensorSpec,
    WireSpec,
)
from polima.policies.registry import register_policy
from polima.policies.smolvla.runtime import (
    ACTION_DIM,
    CACHE_ELEMENTS,
    CHUNK,
    DEFAULT_PORT,
    DENOISE_TOKENS,
    DENOISE_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    PREFIX_TOKENS,
    STATE_DIM,
    SUFFIX_TOKEN,
    WIRE_MAGIC,
)

#: SO-101 follower joints, in dataset order. Same arm as ACT.
JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

#: Cameras are captured at 480x640 and letterboxed to 512x512 by the client
#: before the wire, matching SmolVLM2's expected input geometry.
CAPTURE_HEIGHT, CAPTURE_WIDTH = 480, 640


SMOLVLA_SPEC = PolicySpec(
    name="smolvla",
    display_name="SmolVLA (SmolVLM2 + flow-matching action expert)",
    # --------------------------------------------------------------- dataset --
    dataset=DatasetContract(
        state_names=JOINT_NAMES,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        camera_keys=("observation.images.overhead", "observation.images.wrist"),
        camera_shape=(CAPTURE_HEIGHT, CAPTURE_WIDTH, 3),
        fps=30,
        codebase_version="v3.0",
        # Language-conditioned: several task strings in one dataset is fine.
        single_task=False,
        task_canonicalizer="polima.data.contract:canonical_task",
    ),
    # --------------------------------------------------------------- compile --
    compile=CompilePlan(
        export_entry="polima.policies.smolvla.graphs:export_all",
        verify_entry="polima.policies.smolvla.graphs:verify_chain",
        fixture_entry="polima.policies.smolvla.graphs:write_fixtures",
        normalization_entry="polima.policies.smolvla.graphs:write_normalization",
        fixture_file="smolvla_fixture.npz",
        verify_atol=1e-3,
        verify_rtol=1e-2,
        graphs=(
            # The SmolVLM2 vision tower, and the only NCHW graph.
            #
            # LLiMa produces its ONNX (llima_compile_smolvlm2.py) -- it does NOT
            # compile it. compile_deploy_smolvla_som.sh runs the same afe wrapper
            # over this graph as over the others, just with NCHW:
            #
            #     compile_model "$VISION_ONNX" ... --model-layout NCHW
            #
            # The `_llima_` in the legacy directory name refers to where the ONNX
            # came from, which is what misled this spec into declaring
            # compiler="llima" and disabling tessellation.
            #
            # Run twice per inference, once per camera.
            GraphSpec(
                name="vision",
                aliases=("vision_llima_bf16", "vision_bf16"),
                builder="polima.policies.smolvla.graphs:VisionTower",
                inputs=(TensorSpec("image", (1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)),),
                outputs=(TensorSpec("image_tokens", (1, 64, 960)),),
                layout="NCHW",
                precision="bf16",
                calibration=CalibrationSource("random"),
                # Measured on Modalix: the tessellated HWC16 output contract
                # costs an order of magnitude of fixture accuracy (cosine 0.978
                # against 0.9996) to save 2 ms of the 300 ms chunk. Compile the
                # token output as plain HWC and let the runtime read it directly.
                mla_tessellation=False,
                external_dram_layout="HWC",
                promote_rank3_hwc=True,
                elf_from="retained",
                exit_on_stable_elf=True,
                # The exported SmolVLM graph already carries the exact static
                # boundary and token output. onnxsim rewrites that 390 MiB graph
                # for minutes and can change unsupported attention patterns;
                # the proven SiMa recipe compiles it as emitted.
                llima_args=("--no-simplify",),
            ),
            # Prefix: 241 tokens of vision + language + state -> the packed KV
            # cache every denoise step reads. Runs once.
            GraphSpec(
                name="prefix",
                aliases=("prefix_llima_bf16", "prefix_llima_nhwc_bf16"),
                builder="polima.policies.smolvla.graphs:PrefixCache",
                inputs=(TensorSpec("prefix_embeddings", (1, 1, PREFIX_TOKENS, 960)),),
                outputs=(TensorSpec("cache", (1, 1, 32, CACHE_ELEMENTS // 32)),),
                layout="NHWC",
                precision="bf16",
                calibration=CalibrationSource("random"),
                # The ELF-only runtime does not run MPK tessellation plugins.
                # Keep both boundaries flat HWC and preserve the attention
                # weights in BF16; INT8 weight drift is amplified enough by
                # the prefix attention stack to produce non-finite caches.
                mla_tessellation=False,
                external_dram_layout="HWC",
                exit_on_stable_elf=True,
                # SmolVLA's proven compiler path runs ONNX shape inference.
                # Omitting it can still produce an ELF, but the action-side
                # graphs then have the wrong MLA boundary. --no-simplify keeps
                # the RMSNorms in the pattern ModelSDK fuses (graphs.fusible_rms);
                # onnxsim would drop their Mul(weight) and break the fusion.
                llima_args=("--no-simplify", "--infer-shapes"),
            ),
            # Action expert input projection + time conditioning. Runs 10x.
            GraphSpec(
                name="suffix",
                aliases=("suffix_llima_bf16", "suffix_host_time_llima_bf16"),
                builder="polima.policies.smolvla.graphs:SuffixProjection",
                inputs=(TensorSpec("suffix_input", (1, 1, CHUNK, SUFFIX_TOKEN)),),
                outputs=(TensorSpec("suffix_output", (1, 1, CHUNK, 720)),),
                layout="NHWC",
                precision="bf16",
                calibration=CalibrationSource("random"),
                llima_args=("--infer-shapes",),
            ),
            # The velocity field. Reads the cache plus this step's suffix and
            # emits dx/dt over the 50x32 action lane. Runs 10x -- the single
            # most expensive thing in the pipeline.
            GraphSpec(
                name="denoise",
                aliases=("denoise_single_bf16", "denoise_core_llima_nhwc_bf16"),
                builder="polima.policies.smolvla.graphs:DenoiseExpert",
                inputs=(TensorSpec("denoise_input", (1, 1, DENOISE_TOKENS, DENOISE_WIDTH)),),
                # Modalix hardware proof (2026-09-10): a bare denoise ELF
                # downloads ordinary contiguous BF16. Interpreting those bytes
                # as HWC16 created two non-finite values (first at index 320),
                # while the same raw words decoded as plain BF16 were all finite
                # with a plausible [-3.28125, 2.65625] range.
                outputs=(TensorSpec("velocity", (1, 50, 32)),),
                layout="NHWC",
                precision="bf16",
                # ModelSDK 2.1's all-BF16 codegen is numerically unstable for
                # this checkpoint's denoiser even though ONNX Runtime is finite.
                # BF16 activations preserve its dynamic range; calibrated INT8
                # weights avoid the broken all-BF16 kernel selection.
                # Re-confirmed 2026-08-20: an all-BF16 rebuild of this graph
                # returns a non-finite action at index 0 on the first request.
                activation_precision="bf16",
                weight_precision="int8",
                calibration=CalibrationSource("raw_f32"),
                # --no-simplify: keep the fusible RMSNorm pattern (see prefix).
                # Unsimplified, the graph ends in a reshape that AFE leaves in
                # the segment IR while the MLA boundary is its producer;
                # --groot-boundaries detessellates the producer instead (with
                # BF16 activations it changes nothing else).
                llima_args=("--no-simplify", "--infer-shapes", "--groot-boundaries"),
            ),
        ),
    ),
    # ------------------------------------------------------------------ wire --
    wire=WireSpec(
        magic=WIRE_MAGIC,                 # "SMOL" little-endian
        version=1,
        default_port=DEFAULT_PORT,
        request_header="<IIII",
        response_header="<IIIIfI",
        request_tensors=(
            # Letterboxed, normalized, and serialized HWC by the live client.
            TensorSpec("image0", (IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
            TensorSpec("image1", (IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
            TensorSpec("state", (STATE_DIM,)),
            # The flow-matching seed. Supplied by the client rather than drawn
            # on the board so a run is reproducible from its request alone.
            TensorSpec("noise", (CHUNK, 32)),
        ),
        response_shape=(CHUNK, ACTION_DIM),
        # Unlike ACT, normalization is server-side: the board already holds the
        # statistics as sidecars because the state projection needs them there.
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
    ),
    runtime_plan_builder="polima.policies.smolvla.runtime:build_plan",
    checkpoint_validator="polima.policies.smolvla.graphs:validate_checkpoint",
    profile_entry="polima.policies.smolvla.profile:specialize_spec",
    # ----------------------------------------------------------------- smoke --
    # ACT's 0.01 is the wrong bar here. Measured on Modalix 2026-08-20, the
    # deployed chain agrees with PyTorch to cosine 0.99957 but a mean of 1.85
    # degrees per joint, because vision and denoise are quantized and the error
    # compounds along the 50-action chunk (0.24 deg over the first ten steps,
    # 3.79 over the last ten). 3.0 accepts that and still rejects every bad
    # build we have seen: the mis-calibrated denoise scored 10.1 and 10.5, and
    # an all-BF16 denoise returns non-finite, which fails on any threshold.
    smoke=SmokeSpec(mean_abs_max=3.0),
)

register_policy(SMOLVLA_SPEC)

__all__ = ["JOINT_NAMES", "SMOLVLA_SPEC"]
