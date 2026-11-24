# Reference performance

These results describe the published SO-101 pick-and-place bundles on one
Modalix DevKit. They are reference measurements for the linked artifacts and
software versions, not performance guarantees for other checkpoints or setups.

## Inference latency

Each model uses 30 measured plans after five warm-up plans and runs by itself.
Model latency was measured on eLxr 26.04.02 / platform 3.0.1 with Palette 3.0.0.2,
LLiMa 0.6.2 and PoLiMa runtime `faa710b`. Live observation-to-plan latency comes
from the earlier ModelSDK 2.1.3 reference bundles on a robot setup with two
640 × 480 cameras at 30 FPS. It was not repeated with the current 3.0.0.2
bundles.

| Model | Precision | Actions per plan | Model p50 / p95 / p99 | Live observation to plan, median |
|---|---|---:|---:|---:|
| ACT | BF16 | 100 | 13.4 / 13.5 / 13.5 ms | 25 ms |
| GR00T N1.6 | Mixed INT8/BF16 | 32 | 254.8 / 254.9 / 255.0 ms | 361 ms |
| GR00T N1.7 | Mixed INT8/BF16 | 32 | 212.3 / 212.5 / 219.9 ms | 279 ms |
| Pi0.5 | W8A16 with BF16 attention | 50 | 334.1 / 334.3 / 334.4 ms | 438 ms |

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
| Pi0.5 | PaliGemma: SigLIP So400m/14 + 18-layer Gemma-2B | 18-layer Gemma-300M expert | 10 | 64 |

Graph counts are compiled artifacts, not neural-network layer counts. Exact
graph groups and per-section precision are listed below.

## Numerical fidelity

Each exact release bundle passed its packaged on-device comparison against the
reference output produced by the source policy.

| Model | Cosine similarity | Normalized mean absolute error | Normalized maximum absolute error |
|---|---:|---:|---:|
| ACT | 0.999992 | 0.0030 | 0.0147 |
| GR00T N1.6 | 0.999742 | 0.0083 | 0.0351 |
| GR00T N1.7 | 0.998700 | 0.0114 | 0.1069 |
| Pi0.5 | 0.999920 | 0.0263 | 0.1176 |

ACT passes when cosine similarity is at least 0.999 and normalized mean
absolute error is at most 0.01. The mixed-precision GR00T and Pi0.5 bundles
pass at 0.995 and 0.03 respectively. Maximum error is informational rather
than a separate pass condition.

These checks establish numerical agreement for the tested inputs. They do not
measure physical task success, which depends on the robot and workspace setup.

## Reference artifacts

| Model | Published bundle | Training checkpoint |
|---|---|---|
| ACT | [`ACT-SO101-PickPlace@36f6e012`](https://huggingface.co/simaai/ACT-SO101-PickPlace/tree/36f6e012b20d7c486600207ad10df62fa9ffeb8d) | step 100,000 |
| GR00T N1.6 | [`GR00T-N1.6-SO101-PickPlace@7279862b`](https://huggingface.co/simaai/GR00T-N1.6-SO101-PickPlace/tree/7279862b80b1ed906546d33eed74c5e6ac781a7d) | step 50,000 from [`nvidia/GR00T-N1.6-3B`](https://huggingface.co/nvidia/GR00T-N1.6-3B) |
| GR00T N1.7 | [`GR00T-N1.7-SO101-PickPlace@af7076f7`](https://huggingface.co/simaai/GR00T-N1.7-SO101-PickPlace/tree/af7076f74e189bd05b1a6200a94fe27d31ff8b11) | step 50,000 from [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) |
| Pi0.5 | [`Pi0.5-SO101-PickPlace@fcb19a41`](https://huggingface.co/simaai/Pi0.5-SO101-PickPlace/tree/fcb19a410a00c0c00566a83cb5e278b80cf43b79) | step 20,000 from [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base) |

All four models use the two-camera
[`SO101-GreyEraser-PickPlace`](https://huggingface.co/datasets/Priyakarna/SO101-GreyEraser-PickPlace/tree/bfd7c1831092adfb0f02273e390d86bad193d419)
dataset at revision `bfd7c183`. Each fine-tune uses its 769-demonstration
training split; the complete dataset contains 961 teleoperated demonstrations
and 272,853 frames. The reference task is placing a grey eraser in a white
basket.

The published repositories contain deployable compiled bundles. They do not
contain the original training checkpoints or dataset.

## Compilation and package size

The bundles were compiled with Model Compiler 3.0.0.2 on Linux x86_64. Bundle
size is the deployable `bundle/` directory; allow 1.5 times that space so
PoLiMa can stage and validate a replacement before activation.

| Model | Bundle size | Free space for installation |
|---|---:|---:|
| ACT | 132 MB | 198 MB |
| GR00T N1.6 | 2.67 GB | 4.00 GB |
| GR00T N1.7 | 3.01 GB | 4.52 GB |
| Pi0.5 | 4.28 GB | 6.42 GB |

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
| ACT | 102.7 MB | none | 28.9 MB | 0.04 MB | 27.0 MB |
| GR00T N1.6 | 2,646.9 MB | 5.2 MB | 15.5 MB | 0.04 MB | 113.8 MB |
| GR00T N1.7 | 2,998.0 MB | 7.7 MB | 4.4 MB | 0.04 MB | 112.8 MB |
| Pi0.5 | 3,169.5 MB | 5.4 MB | 30.7 MB | 1,070.9 MB | 375.9 MB |

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
| Pi0.5 | Vision | 27 | INT8 | INT8 per token | BF16 |
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
- **Pi0.5:** `vision_00..26`, `pre_00..17`, shared `attn`, `post_00..16`, and
  `expert`. The prefix graphs run once; `expert` runs ten denoising steps.

The bundle plans have no static INT8 graph inputs. Fused device chains are
validated end to end using the numerical-fidelity checks above.

</details>

## Execution-stage profile

These values are medians over a separate diagnostic series of 30 plans after
five warm-ups, using the exact bundles and runtime from the model-latency
table. They sum accelerator-stage execution by model section. Host preparation,
transport and unassigned graph overhead are excluded, so the sections should
not be added to or directly compared with end-to-end model latency.

| Model | Vision | Language | Action policy |
|---|---:|---:|---:|
| ACT | 5.95 ms | not applicable | 5.40 ms |
| GR00T N1.6 | 124.43 ms | 28.30 ms | 100.07 ms |
| GR00T N1.7 | 67.97 ms | 24.00 ms | 115.92 ms |
| Pi0.5 | 48.77 ms | 144.58 ms | 143.32 ms |

Vision includes both camera views. ACT has no language stage; its action-policy
time includes its transformer encoder and decoder. GR00T action time includes
the action-head adapters, K/V projection and four denoising passes. Pi0.5
language time is the Gemma prefix work after its two vision paths; action time
includes all ten expert denoising steps.

## Reproduce on a DevKit

Download the exact validated revision, then deploy its `bundle/` directory:

```bash
hf download simaai/GR00T-N1.7-SO101-PickPlace \
  --revision af7076f74e189bd05b1a6200a94fe27d31ff8b11 \
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
polima_cli --models-dir /media/nvme/polima/models
use <n>
bench 30 5
check
quit
```

Enable diagnostic stage output in a separate process. Select the bundle shown
by the initial model list, run five warm-ups and 30 measured plans, then group
the emitted `POLIMA_PHASE` stage times by model section:

```bash
POLIMA_PROFILE_PHASES=1 POLIMA_PROFILE_STAGES=1 \
  polima_cli --models-dir /media/nvme/polima/models
use <n>
bench 30 5
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
