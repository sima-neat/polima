# Reference performance

These results describe the published SO-101 pick-and-place bundles on one
Modalix DevKit. They are reference measurements for the linked artifacts and
software versions, not performance guarantees for other checkpoints or setups.

## Inference latency

Each model uses 30 measured plans after five warm-up plans and runs by itself.
Model latency was remeasured on eLxr 26.04.02 with Palette 3.0.0, LLiMa 0.6.0
and PoLiMa runtime `7ca990f`. Live observation-to-plan latency comes from the
reference robot setup with two 640 × 480 cameras at 30 FPS.

| Model | Precision | Actions per plan | Model p50 / p95 / p99 | Live observation to plan, median |
|---|---|---:|---:|---:|
| ACT | BF16 | 100 | 13.5 / 13.5 / 13.5 ms | 25 ms |
| GR00T N1.6 | Mixed INT8/BF16 | 32 | 309.7 / 309.9 / 310.0 ms | 361 ms |
| GR00T N1.7 | Mixed INT8/BF16 | 32 | 226.6 / 227.1 / 227.4 ms | 279 ms |
| Pi0.5 | W8A16 with BF16 attention | 50 | 345.7 / 345.9 / 346.0 ms | 438 ms |

**Model latency** measures policy execution on the DevKit after request
decoding. **Live observation to plan** starts after the two-camera observation
is captured and ends when the action plan enters the robot client's queue. The
arm can continue executing the current plan while the next plan is computed.

## Model configuration

| Model | Visual/language backbone | Action head | Denoising steps | Compiled graphs |
|---|---|---|---:|---:|
| ACT | ResNet-18 + transformer encoder | ACT transformer decoder | not applicable | 6 |
| GR00T N1.6 | Eagle 2.5 | Flow-matching transformer | 4 | 39 |
| GR00T N1.7 | Cosmos-Reason2 / Qwen3-VL | Flow-matching transformer | 4 | 45 |
| Pi0.5 | PaliGemma: SigLIP So400m/14 + 18-layer Gemma-2B | 18-layer Gemma-300M expert | 10 | 38 |

Graph counts are compiled artifacts, not neural-network layer counts. Exact
graph groups and per-section precision are listed below.

## Numerical fidelity

Compiled outputs were compared with the original PyTorch policies on held-out
dataset frames. Every frame used its own input and, for GR00T and
Pi0.5, its own noise tensor.

| Model | Validation inputs | Cosine, mean / worst | Normalized mean error, mean / worst | Normalized max error, mean / worst |
|---|---:|---:|---:|---:|
| ACT | 16 held-out frames | 0.999992 / 0.999984 | 0.0030 / 0.0042 | 0.0190 / 0.0295 |
| GR00T N1.6 | 6 held-out frames | 0.998941 / 0.995766 | 0.0115 / 0.0292 | 0.0545 / 0.1343 |
| GR00T N1.7 | 6 held-out frames | 0.998489 / 0.996070 | 0.0115 / 0.0237 | 0.0680 / 0.1289 |

ACT passes when cosine similarity is at least 0.999 and normalized mean
absolute error is at most 0.01. The mixed-precision GR00T bundles pass at 0.995
and 0.03 respectively. Pi0.5 passed its eight-fixture comparison with an
overall action mean absolute error of 0.016 and a worst-timestep error of 0.037.
Maximum error is informational rather than a separate pass condition.

These checks establish numerical agreement for the tested inputs. They do not
measure physical task success, which depends on the robot and workspace setup.

## Reference artifacts

| Model | Published bundle | Training checkpoint |
|---|---|---|
| ACT | [`ACT-SO101-PickPlace@307e6dbc`](https://huggingface.co/simaai/ACT-SO101-PickPlace/tree/307e6dbccc32d8e52ea0a4a983dd9c024c157643) | step 100,000 |
| GR00T N1.6 | [`GR00T-N1.6-SO101-PickPlace@97b01c04`](https://huggingface.co/simaai/GR00T-N1.6-SO101-PickPlace/tree/97b01c04d4c1751476184d13105df4f6d5670ec9) | step 50,000 from [`nvidia/GR00T-N1.6-3B`](https://huggingface.co/nvidia/GR00T-N1.6-3B) |
| GR00T N1.7 | [`GR00T-N1.7-SO101-PickPlace@f0b5a251`](https://huggingface.co/simaai/GR00T-N1.7-SO101-PickPlace/tree/f0b5a25148f1d41d625043dcf8191d8ff7658feb) | step 50,000 from [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) |
| Pi0.5 | [`Pi0.5-SO101-PickPlace@308b18cf`](https://huggingface.co/simaai/Pi0.5-SO101-PickPlace/tree/308b18cf2d32e28004a8ce2ca1b60ac4fa4c952d) | step 20,000 from [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base) |

All four models use the two-camera
[`SO101-GreyEraser-PickPlace`](https://huggingface.co/datasets/Priyakarna/SO101-GreyEraser-PickPlace/tree/bfd7c1831092adfb0f02273e390d86bad193d419)
dataset at revision `bfd7c183`. Each fine-tune uses its 769-demonstration
training split; the complete dataset contains 961 teleoperated demonstrations
and 272,853 frames. The reference task is placing a grey eraser in a white
basket.

The published repositories contain deployable compiled bundles. They do not
contain the original training checkpoints or dataset.

## Compilation and package size

The bundles were compiled on Linux x86_64 with an Intel Core Ultra 7 265KF,
249 GiB RAM and an NVIDIA GeForce RTX 5090. GR00T used three concurrent graph
jobs; ACT used six. Pi0.5 used three for its ModelSDK expert build.

| Model | Reference build time | Download size | Free space for installation |
|---|---:|---:|---:|
| ACT | 2 min 46 s | 142 MB | 213 MB |
| GR00T N1.6 | 2 h 52 min | 3.44 GB | 5.16 GB |
| GR00T N1.7 | 2 h 0 min | 3.64 GB | 5.46 GB |
| Pi0.5 | 41 min 52 s | 5.04 GB | 7.56 GB |

Build time depends on host hardware and compiler load. Installation space is
1.5 times the downloaded artifact so PoLiMa can validate a replacement before
activation.

<details>
<summary>Build commands and package contents</summary>

ACT was compiled with:

```bash
polima compile --policy act \
  --checkpoint "$CHECKPOINT" \
  --build-dir "$BUILD_DIR" \
  --output-root "$OUTPUT_ROOT" \
  --jobs 6
```

GR00T was compiled with its version-specific Isaac-GR00T environment:

```bash
polima compile --policy groot \
  --checkpoint "$CHECKPOINT" \
  --dataset "$LEROBOT_V2_DATASET" \
  --output "$BUILD_DIR" \
  --precision mixed \
  --jobs 3 \
  --check-devkit sima@<devkit-ip>
```

Pi0.5 was compiled from its LeRobot checkpoint and v3 dataset:

```bash
polima compile --policy pi05 \
  --checkpoint "$CHECKPOINT" \
  --dataset "$LEROBOT_V3_DATASET" \
  --precision w8a16 \
  --output "$BUILD_DIR" \
  --jobs 3 \
  --check-devkit sima@<devkit-ip>
```

| Model | Compiled ELFs | Constants | Fixtures | Robot client | Largest ELF |
|---|---:|---:|---:|---:|---:|
| ACT | 112.8 MB | none | 28.9 MB | 0.03 MB | 31.0 MB |
| GR00T N1.6 | 3,431.2 MB | 5.2 MB | 2.1 MB | 0.03 MB | 131.1 MB |
| GR00T N1.7 | 3,626.7 MB | 7.7 MB | 4.4 MB | 0.03 MB | 128.3 MB |
| Pi0.5 | 3,932.7 MB | 5.4 MB | 30.7 MB | 1,070.9 MB | 722.2 MB |

</details>

## Precision

ACT uses BF16 weights and activations throughout. In the mixed GR00T bundles,
transformer linear weights use symmetric INT8 quantization per output channel.
Their BF16 inputs are dynamically quantized to INT8 per token inside the
accelerator operation. Graph boundaries and the remaining graphs stay in BF16.
LayerNorm, RMSNorm, attention and softmax remain compiled operations. The
quantized build descriptions contain no `RsqrtOp` or
`QuantizationTransformOp` fallback. Pi0.5 uses dynamic W8A16 for its vision,
Gemma and action-expert weight layers; the shared attention graph remains BF16.

| Model | Model section | Graph count | Weights | Linear inputs | Graph boundaries |
|---|---|---:|---|---|---|
| ACT | Vision, encoder and decoder | 6 | BF16 | BF16 | BF16 |
| GR00T N1.6 | Vision transformer | 14 | INT8 | INT8 per token | BF16 |
| GR00T N1.6 | Language transformer | 8 | INT8 | INT8 per token | BF16 |
| GR00T N1.6 | Action transformer | 10 | INT8 | INT8 per token | BF16 |
| GR00T N1.6 | Adapters, norms and projectors | 7 | BF16 | BF16 | BF16 |
| GR00T N1.7 | Vision transformer | 12 | INT8 | INT8 per token | BF16 |
| GR00T N1.7 | Language transformer | 10 | INT8 | INT8 per token | BF16 |
| GR00T N1.7 | Action transformer | 10 | INT8 | INT8 per token | BF16 |
| GR00T N1.7 | Adapters, merges and projectors | 13 | BF16 | BF16 | BF16 |
| Pi0.5 | Vision | 1 | INT8 | INT8 per token | BF16 |
| Pi0.5 | Gemma prefix pre/post | 35 | INT8 | INT8 per token | BF16 |
| Pi0.5 | Shared attention | 1 | BF16 | BF16 | BF16 |
| Pi0.5 | Action expert | 1 | INT8 | INT8 per token | BF16 |

<details>
<summary>Compiled graph groups</summary>

- **ACT:** `vision_backbone`, `encoder_layer_00_stem`,
  `encoder_layer_01..03`, and `decoder_action_tail`.
- **GR00T N1.6 vision:** `eagle_vision_00_01` through
  `eagle_vision_26_26`.
- **GR00T N1.6 language:** `eagle_qwen_00_01` through
  `eagle_qwen_14_15`.
- **GR00T N1.6 action:** `dit_kv_00_14`, `dit_kv_16_30`, and
  `ditkv_blocks_00_03` through `ditkv_blocks_28_31`.
- **GR00T N1.6 BF16:** `eagle_vision_patch`,
  `eagle_vision_post_norm`, `eagle_vision_connector`,
  `eagle_output_norm`, `state_project`, `action_project`, and `action_tail`.
- **GR00T N1.7 vision:** `vision_00_01` through `vision_22_23`.
- **GR00T N1.7 language:** `language_00_00` through `language_15_15`.
- **GR00T N1.7 action:** `dit_kv_00_14`, `dit_kv_16_30`, and
  `ditkv_00_03` through `ditkv_28_31`.
- **GR00T N1.7 BF16:** `vision_patch`, `vision_merge`,
  `deepstack_merge_0..2`, `vlln`, `vl_00..03`, `state_project`,
  `action_project`, and `action_tail`.
- **Pi0.5:** `vision`, `pre_00..17`, shared `attn`, `post_00..16`, and
  `expert`. The prefix graphs run once; `expert` runs ten denoising steps.

The bundle plans have no static INT8 graph inputs. Fused device chains are
validated end to end using the numerical-fidelity checks above.

</details>

## Execution-stage profile

These values are medians over ten diagnostic runs after five warm-ups, using
the same bundles and SDK 3 runtime as the model-latency measurements above.
Stage profiling runs each compiled graph separately and therefore disables the
normal accelerator queue. The values identify where accelerator execution time
is spent; do not add them to or directly compare them with the queued p50/p95/p99
table. Host preparation and transport are not assigned to a neural-network
stage.

| Model | Vision | Language | Action policy |
|---|---:|---:|---:|
| ACT | 5.85 ms | not applicable | 5.52 ms |
| GR00T N1.6 | 155.27 ms | 42.61 ms | 108.95 ms |
| GR00T N1.7 | 70.40 ms | 27.45 ms | 123.78 ms |
| Pi0.5 | 197.65 ms combined prefill | included in prefill | 148.49 ms |

Vision includes both camera views. ACT has no language stage; its action-policy
time includes its transformer encoder and decoder. GR00T action time includes
the action-head adapters, K/V projection and four denoising passes. Pi0.5
exposes one combined vision-language prefill rather than independently timed
vision and language stages; its action time includes all ten expert denoising
steps.

## Reproduce on a DevKit

Download the exact validated revision, then deploy its `bundle/` directory:

```bash
hf download simaai/GR00T-N1.7-SO101-PickPlace \
  --revision f0b5a25148f1d41d625043dcf8191d8ff7658feb \
  --local-dir GR00T-N1.7-SO101-PickPlace

polima deploy \
  --bundle ./GR00T-N1.7-SO101-PickPlace/bundle \
  --devkit sima@<devkit-ip> \
  --start

polima run \
  --bundle ./GR00T-N1.7-SO101-PickPlace/bundle \
  --devkit sima@<devkit-ip> \
  --fixture \
  --repeat 30
```

Replace the repository and revision with the ACT, N1.6 or Pi0.5 value in the
artifact table. These commands validate inference without moving a robot.
Studio **Performance** provides the same hardware-free benchmark workflow.

For native p50/p95/p99, open the runtime CLI on the DevKit:

```bash
polima_cli --models-dir /media/nvme/polima/models \
  --interactive --bundle <bundle-id> --exclusive-control
bench 30 5
check
quit
```

Run diagnostic stage profiling in a separate process so it cannot change the
queued latency measurement. At the prompt, run `stages` 15 times and discard
the first five profiles:

```bash
POLIMA_PROFILE_PHASES=1 POLIMA_PROFILE_STAGES=1 \
  polima_cli --models-dir /media/nvme/polima/models \
  --interactive --bundle <bundle-id> --exclusive-control
stages
quit
```

Raw samples, reference outputs, seeds, memory snapshots and SHA-256 manifests
are retained with SiMa.ai's release-validation records.

## Interpretation

Compilation, numerical validation and robot operation are separate results:

| Result | What it establishes |
|---|---|
| Compilation | ModelSDK produced the required graphs and PoLiMa packaged them. |
| Fixture check | The compiled model matches its packaged reference input within the stated tolerance. |
| Held-out comparison | Compiled outputs match PyTorch on the tested dataset frames. |
| Physical run | The stated robot and setup completed the observed task trials. |

Use Studio **Performance** to benchmark a bundle on its deployed hardware.
