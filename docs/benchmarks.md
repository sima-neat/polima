# Reference performance

These results describe the published SO-101 pick-and-place bundles on one
Modalix DevKit. They are reference measurements for the linked artifacts and
software versions, not performance guarantees for other checkpoints or setups.

## Inference latency

ACT and GR00T use 30 measured plans after five warm-up plans. Pi0.5 uses 20
after three. Each model ran by itself. The camera-to-plan values come from the
reference robot setup with two 640 × 480 cameras at 30 FPS.

| Model | Precision | Actions per plan | Model p50 / p95 / p99 | Camera to plan, median |
|---|---|---:|---:|---:|
| ACT | BF16 | 100 | 13.23 / 13.39 / 13.45 ms | 25 ms |
| GR00T N1.6 | Mixed INT8/BF16 | 32 | 315.40 / 315.62 / 315.68 ms | 361 ms |
| GR00T N1.7 | Mixed INT8/BF16 | 32 | 231.65 / 232.17 / 232.57 ms | 279 ms |
| Pi0.5 | W8A16 with BF16 attention | 50 | 358.5 / 359.1 / 360.3 ms | not measured |

**Model latency** measures policy execution on the DevKit after request
decoding. **Camera to plan** starts at the newer timestamp of the two camera
frames and ends when the action plan is ready. The live-camera series retained
its medians but not individual samples, so p95 and p99 are reported only for
model execution. Pi0.5 camera-to-plan samples were not retained, so that
value is left unreported. The arm can continue executing the current plan while
the next plan is computed.

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
| Pi0.5 | 8 fixtures | not retained | 0.0160 overall; 0.0370 worst timestep | not retained |

ACT passes when cosine similarity is at least 0.999 and normalized mean
absolute error is at most 0.01. The mixed-precision GR00T bundles pass at 0.995
and 0.03 respectively. Pi0.5's eight-fixture check passes when the mean across
the set meets the 0.995 cosine and 0.03 mean-error gates. Its retained summary
reports overall action error and the worst individual timestep, rather than
per-frame cosine and maximum error.
Maximum error is reported where retained but is not a separate pass condition.

These checks establish numerical agreement for the tested inputs. They do not
measure physical task success. A controlled robot success rate was not recorded
for this benchmark series.

## Reference artifacts

| Model | Published bundle | Checkpoint | Dataset split | Build software |
|---|---|---|---|---|
| ACT | [`ACT-SO101-PickPlace@307e6dbc`](https://huggingface.co/simaai/ACT-SO101-PickPlace/tree/307e6dbccc32d8e52ea0a4a983dd9c024c157643) | step 100,000 | 769 of 961 demonstrations | PoLiMa `e997878`, ModelSDK 2.1.3 |
| GR00T N1.6 | [`GR00T-N1.6-SO101-PickPlace@97b01c04`](https://huggingface.co/simaai/GR00T-N1.6-SO101-PickPlace/tree/97b01c04d4c1751476184d13105df4f6d5670ec9) | step 50,000 from [`nvidia/GR00T-N1.6-3B`](https://huggingface.co/nvidia/GR00T-N1.6-3B) | 769 of 961 demonstrations | PoLiMa `19c052e`, ModelSDK 2.1.3 |
| GR00T N1.7 | [`GR00T-N1.7-SO101-PickPlace@f0b5a251`](https://huggingface.co/simaai/GR00T-N1.7-SO101-PickPlace/tree/f0b5a25148f1d41d625043dcf8191d8ff7658feb) | step 50,000 from [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) | 769 of 961 demonstrations | PoLiMa `19c052e`, ModelSDK 2.1.3 |
| Pi0.5 | [`Pi0.5-SO101-PickPlace@308b18cf`](https://huggingface.co/simaai/Pi0.5-SO101-PickPlace/tree/308b18cf2d32e28004a8ce2ca1b60ac4fa4c952d) | step 20,000 from [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base) | 769 of 961 demonstrations | PoLiMa `d31a58a`, ModelSDK 2.1.3, LLiMa 0.4.0 |

All four models use the two-camera
[`SO101-GreyEraser-PickPlace`](https://huggingface.co/datasets/Priyakarna/SO101-GreyEraser-PickPlace/tree/bfd7c1831092adfb0f02273e390d86bad193d419)
dataset at revision `bfd7c183`. It contains 961 teleoperated demonstrations and
272,853 frames. The reference task is placing a grey eraser in a white basket.

The published repositories contain deployable compiled bundles. They do not
contain the original training checkpoints or dataset.

<details>
<summary>Exact bundle identifiers and source revisions</summary>

- ACT: `act-so101-grey-eraser-pickplace-100k-bf16-1879ca7c`
- GR00T N1.6: `groot-so101-grey-eraser-pickplace-v1.6-50k-mixed-28ce4ebf`,
  Isaac-GR00T `9b37aa1ce69c73c6d165233fa88128283bba4508`
- GR00T N1.7: `groot-so101-grey-eraser-pickplace-v1.7-50k-mixed-d32a6801`,
  Isaac-GR00T `51d4c89f72fda44cbf77285c6a8114b52676b8a1`
- Pi0.5: `pi05-so101-grey-eraser-pickplace-20k-w8a16-2ae183ea`, based on `lerobot/pi05_base`

</details>

## Compilation and package size

The bundles were compiled on Linux x86_64 with an Intel Core Ultra 7 265KF,
249 GiB RAM and an NVIDIA GeForce RTX 5090. GR00T used three concurrent graph
jobs; ACT used six. Pi0.5 used three for its ModelSDK expert build.

| Model | Recorded build time | Peak build RAM | Retained build tree | Downloaded artifact | Free space for installation |
|---|---:|---:|---:|---:|---:|
| ACT | 2 min 46 s | 2.12 GiB | 1.62 GB | 142 MB | 213 MB |
| GR00T N1.6 | 2 h 52 min | not recorded | 91.4 GB | 3.44 GB | 5.16 GB |
| GR00T N1.7 | 2 h 0 min | not recorded | 83.4 GB | 3.64 GB | 5.46 GB |
| Pi0.5 | 41 min 52 s | not recorded | 49.0 GB | 5.04 GB | 7.56 GB |

The GR00T and Pi0.5 retained-tree values include exports and compiler
intermediates; they are not minimum disk requirements. Installation space is 1.5 times the final
artifact so PoLiMa can stage and validate a replacement before activation.

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

The bundle plans have no static INT8 graph inputs. Per-graph compiled accuracy
was not measured because the fused device chains do not expose every
intermediate output. End-to-end compiled accuracy is reported above.

</details>

## DevKit resources

The measurements used eLxr 12, kernel `6.18.3-modalix`, Neat 0.5.0,
`sima_lmm` 0.5.0 and PoLiMa runtime `7342f909`. Studio, the robot client and
other inference workloads were stopped. Each model ran alone.

| Model | ELF-cache-cold load | Immediate reload | Server peak RSS | Fixture client peak RSS | Free CMA during requests |
|---|---:|---:|---:|---:|---:|
| ACT | 1.86 s | 1.37 s | 79 MiB | 38 MiB | 142–144 MiB |
| GR00T N1.6 | 8.24 s | 5.85 s | 99 MiB | 40 MiB | 27–36 MiB |
| GR00T N1.7 | 9.23 s | 9.52 s | 179 MiB | 47 MiB | 12–21 MiB |
| Pi0.5 | not retained | not retained | not retained | not retained | not retained |

The cold-load test evicted only the inactive bundle's ELF pages; it was not a
full reboot. CMA is shared with the system, so free CMA is a system snapshot,
not model memory consumption. The client values come from fixture inference
without camera capture or motor control; full robot-client memory was not
measured. Pi0.5 resource snapshots were not retained for this series.

<details>
<summary>Execution-stage profile</summary>

Stage profiling used the sixth request after five warm-ups. It disables normal
accelerator queuing, so these diagnostic values should not be added to the
headline latency table.

| Model | Vision | Language | Action policy |
|---|---:|---:|---:|
| ACT | 6.39 ms | not applicable | 6.34 ms |
| GR00T N1.6 | 163.22 ms | 44.42 ms | 115.76 ms |
| GR00T N1.7 | 76.54 ms | 29.18 ms | 131.08 ms |
| Pi0.5 | 202.39 ms combined prefill | included in prefill | 155.26 ms |

Vision includes both camera views. GR00T action time includes K/V projection
and four denoising passes. Pi0.5 retained one combined vision-language prefill
time; its action time includes all ten denoising steps.

</details>

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

For native p50/p95/p99 and stage output, open the runtime CLI on the DevKit:

```bash
polima_cli --models-dir /media/nvme/polima/models \
  --interactive --bundle <bundle-id> --exclusive-control
bench 30 5
stages
check
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

This benchmark series did not record full robot-client memory, per-graph
compiled accuracy, live-camera p95/p99, or a controlled physical success rate.
