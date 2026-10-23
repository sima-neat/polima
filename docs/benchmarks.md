# Reference benchmarks

This page records build and validation evidence for the SO-101 pick-and-place
reference bundles. The measurements apply to these artifacts and test
conditions; they are not performance guarantees for another checkpoint or
workspace.

## GR00T N1.6 and N1.7

### Artifacts

| | GR00T N1.6 | GR00T N1.7 |
|---|---|---|
| Public bundle ID | `groot-so101-grey-eraser-pickplace-v1.6-50k-mixed-28ce4ebf` | `groot-so101-grey-eraser-pickplace-v1.7-50k-mixed-d32a6801` |
| Compiled bundle | [`simaai/GR00T-N1.6-SO101-PickPlace`](https://huggingface.co/simaai/GR00T-N1.6-SO101-PickPlace/tree/97b01c04d4c1751476184d13105df4f6d5670ec9) | [`simaai/GR00T-N1.7-SO101-PickPlace`](https://huggingface.co/simaai/GR00T-N1.7-SO101-PickPlace/tree/f0b5a25148f1d41d625043dcf8191d8ff7658feb) |
| Hugging Face revision | `97b01c04d4c1751476184d13105df4f6d5670ec9` | `f0b5a25148f1d41d625043dcf8191d8ff7658feb` |
| Fine-tuned checkpoint | step 50,000; 32 actions returned | step 50,000; 32 actions returned |
| Fine-tuned from | [`nvidia/GR00T-N1.6-3B`](https://huggingface.co/nvidia/GR00T-N1.6-3B) | [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) |
| Isaac-GR00T source | `9b37aa1ce69c73c6d165233fa88128283bba4508` | `51d4c89f72fda44cbf77285c6a8114b52676b8a1` |
| Dataset | [`Priyakarna/SO101-GreyEraser-PickPlace`](https://huggingface.co/datasets/Priyakarna/SO101-GreyEraser-PickPlace/tree/bfd7c1831092adfb0f02273e390d86bad193d419) | same |
| Dataset revision | `bfd7c1831092adfb0f02273e390d86bad193d419` | same |
| Training split | 769 of 961 episodes | 769 of 961 episodes |
| Task | Place the grey eraser in the white basket | same |
| PoLiMa build revision | `19c052efc4698054d9a759a63231465f7f4d8b3a` | same |
| ModelSDK | 2.1.3 | 2.1.3 |
| Modalix runtime | Neat 0.5.0, `sima_lmm` 0.5.0 | same |

The public repositories contain deployable compiled bundles. The source
fine-tuned checkpoints are not part of those repositories. The immutable
Hugging Face revisions above identify the compiled releases.

The dataset has 961 teleoperated demonstrations and 272,853 frames at 30 FPS.
It uses two 640×480 views and six state/action values: five arm joints and the
gripper. Its card documents the camera correction and the 769/96/96 split.

### Compilation

Both bundles were built on a Linux x86_64 host with an Intel Core Ultra 7
265KF (20 cores), 249 GiB RAM and an NVIDIA GeForce RTX 5090 with 32,607 MiB
VRAM. Three graphs were compiled concurrently.

```bash
polima compile --policy groot \
  --checkpoint "$CHECKPOINT" \
  --dataset "$LEROBOT_V2_DATASET" \
  --output "$BUILD_DIR" \
  --precision mixed \
  --jobs 3 \
  --check-devkit sima@modalix
```

The checkpoint and dataset paths are installation-specific. N1.6 and N1.7
require their corresponding Isaac-GR00T source and Python environment, as
described in [Run your own model](your-own-model.md#gr00t).

| Build stage | N1.6 | N1.7 |
|---|---:|---:|
| Export | 15 min 22 s | 8 min 56 s |
| Action-generator preparation | 2 min 21 s | 1 min 5 s |
| ModelSDK compilation | 2 h 33 min 44 s | 1 h 50 min 10 s |
| Final bundle packaging | 9 s | 5 s |
| Total of recorded stages | 2 h 51 min 36 s | 2 h 0 min 16 s |
| Retained build tree | 91.4 GB | 83.4 GB |

The retained-tree figure includes exports, compiler intermediates and the
bundle. Peak disk use and peak host RAM were not recorded, so this page does
not state them as requirements.

### Precision map

`mixed` keeps graph boundaries in BF16. Transformer linear weights are stored
as symmetric INT8 values per output channel; their BF16 inputs are dynamically
quantized to INT8 per token inside the accelerator operation. LayerNorm,
RMSNorm and Softmax remain compiler operations. The build descriptions contain
no `RsqrtOp` or `QuantizationTransformOp` fallback.

| Model | Graphs | Count | Weights | Linear inputs | Between graphs | Other operations | Per-graph accuracy |
|---|---|---:|---|---|---|---|---|
| N1.6 | `eagle_vision_00_01` … `eagle_vision_26_26` | 14 | INT8 symmetric per output channel | INT8 dynamic per token | BF16 | Vision attention, LayerNorm and Softmax remain compiler operations | not measured |
| N1.6 | `eagle_qwen_00_01` … `eagle_qwen_14_15` | 8 | INT8 symmetric per output channel | INT8 dynamic per token | BF16 | Language attention, RMSNorm, Softmax and KV values remain in the compiled graph/BF16 boundary contract | not measured |
| N1.6 | `dit_kv_00_14`, `dit_kv_16_30`; `ditkv_blocks_00_03` … `ditkv_blocks_28_31` | 10 | INT8 symmetric per output channel | INT8 dynamic per token | BF16 | Cross-attention K/V and denoising state cross graph boundaries in BF16 | not measured |
| N1.6 | `eagle_vision_patch`, `eagle_vision_post_norm`, `eagle_vision_connector`, `eagle_output_norm`, `state_project`, `action_project`, `action_tail` | 7 | BF16 | BF16 | BF16 | Precision-sensitive adapters, normalization, projectors and output remain BF16 | not measured |
| N1.7 | `vision_00_01` … `vision_22_23` | 12 | INT8 symmetric per output channel | INT8 dynamic per token | BF16 | Vision attention, LayerNorm and Softmax remain compiler operations | not measured |
| N1.7 | `language_00_00` … `language_15_15` | 10 | INT8 symmetric per output channel | INT8 dynamic per token | BF16 | Language attention, RMSNorm, Softmax and KV values remain in the compiled graph/BF16 boundary contract | not measured |
| N1.7 | `dit_kv_00_14`, `dit_kv_16_30`; `ditkv_00_03` … `ditkv_28_31` | 10 | INT8 symmetric per output channel | INT8 dynamic per token | BF16 | Cross-attention K/V and denoising state cross graph boundaries in BF16 | not measured |
| N1.7 | `vision_patch`, `vision_merge`, `deepstack_merge_0..2`, `vlln`, `vl_00..03`, `state_project`, `action_project`, `action_tail` | 13 | BF16 | BF16 | BF16 | Precision-sensitive patch/merge, bridge, projectors and output remain BF16 | not measured |

N1.6 contains 39 compiled graphs (32 mixed-precision, 7 BF16). N1.7
contains 45 (32 mixed-precision, 13 BF16).

The INT8 weight scale comes directly from each output channel's weights. The
activation scale is computed from each input token at run time; there is no
static activation calibration or INT16 path. The bundle plans contain no
`int8_inputs`, so graph boundaries remain BF16.

### Bundle size

| | N1.6 | N1.7 |
|---|---:|---:|
| Compiled ELF files | 3.431 GB | 3.627 GB |
| Complete downloaded artifact | 3.439 GB (3.202 GiB) | 3.639 GB (3.389 GiB) |
| Recommended free space during installation | 5.16 GB | 5.46 GB |

The free-space value is 1.5 times the artifact size so installation can stage
and validate a replacement before switching the active bundle. Logical file
sizes for every component below were also verified on the bench DevKit on
2026-10-23 and matched these values exactly. These are file bytes, not
filesystem allocated-block counts.

| Component | N1.6 | N1.7 |
|---|---:|---:|
| ELF files | 3,431,228,976 bytes | 3,626,659,792 bytes |
| Largest ELF | 131,075,120 bytes (`eagle_qwen_12_13`) | 128,309,872 bytes (`language_05_06`) |
| Constants | 5,178,924 bytes | 7,712,976 bytes |
| Fixtures | 2,061,134 bytes | 4,351,486 bytes |
| Robot client | 30,110 bytes | 30,110 bytes |
| Bundle metadata | 81,770 bytes | 383,340 bytes |

## ACT

### Artifact

| | Reference bundle |
|---|---|
| Public bundle ID | `act-so101-grey-eraser-pickplace-100k-bf16-1879ca7c` |
| Compiled bundle | [`simaai/ACT-SO101-PickPlace`](https://huggingface.co/simaai/ACT-SO101-PickPlace/tree/307e6dbccc32d8e52ea0a4a983dd9c024c157643) |
| Hugging Face revision | `307e6dbccc32d8e52ea0a4a983dd9c024c157643` |
| Training checkpoint | step 100,000 |
| Compiled output | 100 actions × 6 joints |
| Dataset | [`Priyakarna/SO101-GreyEraser-PickPlace`](https://huggingface.co/datasets/Priyakarna/SO101-GreyEraser-PickPlace/tree/bfd7c1831092adfb0f02273e390d86bad193d419) |
| Dataset revision | `bfd7c1831092adfb0f02273e390d86bad193d419` |
| Training split | 769 of 961 episodes; 192 held out |
| PoLiMa build revision | `e997878ee0a760563def8fa1bebe54c67c8504dd` |
| ModelSDK | 2.1.3 |
| Modalix runtime | PoLiMa `7342f90998eb517572151ac3e61bcbc6455e0b87`, Neat 0.5.0, `sima_lmm` 0.5.0 |

The public package preserves the validated ELFs, fixtures and robot client.
Release packaging added the task instruction, removed local build paths and
recomputed the content-addressed bundle ID. The dataset is the same corrected
two-camera dataset used by the GR00T reference bundles. The saved training
configuration lists the exact 769 training episodes.

### Compilation

ACT was compiled on the same Linux x86_64 host described above. The recorded
command was:

```bash
polima compile --policy act \
  --checkpoint "$CHECKPOINT" \
  --build-dir "$BUILD_DIR" \
  --output-root "$OUTPUT_ROOT" \
  --jobs 6
```

| Measurement | Result |
|---|---:|
| Export | 12.2 s |
| ModelSDK compilation | 139.8 s |
| Complete command | 2 min 45.9 s |
| Peak host resident memory | 2,225,684 KiB (2.12 GiB) |
| Retained build tree | 1.619 GB |

The complete-command time includes loading, export, compilation and packaging.
The six graph compilations ran concurrently, so their individual elapsed times
do not add up to the wall-clock compilation time.

### Precision map

All ACT graph weights, graph inputs, intermediate activations and graph outputs
are BF16. No dynamic INT8 activation quantization or static INT8 graph boundary
is used.

| Graph | Model section | Weights | Activations |
|---|---|---|---|
| `vision_backbone` | Two-camera ResNet-18 feature extraction | BF16 | BF16 |
| `encoder_layer_00_stem` | State/image token stem and first transformer encoder layer | BF16 | BF16 |
| `encoder_layer_01` | Transformer encoder layer 1 | BF16 | BF16 |
| `encoder_layer_02` | Transformer encoder layer 2 | BF16 | BF16 |
| `encoder_layer_03` | Transformer encoder layer 3 | BF16 | BF16 |
| `decoder_action_tail` | Decoder and 100×6 action output | BF16 | BF16 |

The export used eight calibration samples to establish representative graph
inputs and verify the exported chain. Because this bundle is BF16, those
samples do not set INT8 quantization scales. Norm, attention and softmax
operations remain inside the BF16 graphs.

### Bundle size

| Component | Bytes | Decimal size |
|---|---:|---:|
| Six ELF files | 112,796,480 | 112.80 MB |
| Largest ELF (`vision_backbone`) | 31,027,752 | 31.03 MB |
| Fixtures | 28,928,970 | 28.93 MB |
| Robot client | 32,128 | 0.03 MB |
| Bundle metadata | 10,966 | 0.01 MB |
| Complete bundle | 141,768,544 | 141.77 MB |
| Recommended free space during installation | 212,652,816 | 212.65 MB |

The bundle has no separate constants directory. Recommended free space is 1.5
times the complete bundle size. All component byte counts were verified on the
bench DevKit.

## DevKit measurements

All three compiled model payloads in the artifact tables were measured on the
same bench DevKit with runtime `7342f90998eb517572151ac3e61bcbc6455e0b87`.
The runtime was built in an isolated directory; the installed runtime,
activation, bundles and calibrations were preserved.
CLI SHA-256: `abead5178c4c820cda96fa56b92941b2648bb64315c5ae4c57ba38736efad70f`.
Server SHA-256: `3f07813243fb536550875bfe051e51042fa16fdc742c5a78f42be28c62f19615`.
All GR00T ELF hashes matched the pinned releases. The ACT release preserves the
measured artifact's six ELF files, fixtures and robot client byte for byte; its
manifest and plan differ only by the public task and provenance metadata noted
above. All 16 held-out input/reference sets matched the retained host artifacts.

The bench device ran eLxr 12 (aria), kernel `6.18.3-modalix`, Neat 0.5.0 and
`sima-lmm-core` 0.5.0. The native build used GNU C++ 12.2.0, Release mode and
two build jobs. The standard DevKit runtime recovery completed before the
measurement series. No further reset or service reconfiguration occurred.

Active services included `simaai-appcomplex`, pipeline manager, codec
encoder/decoder services, trace daemons, RPYC/VDP agents, LightDM, Mosquitto and
standard system services. No Studio, robot client or separate LLiMa/GenAI
inference workload was observed. All six CLI/server series had the same 29
running service units. Each model ran alone; all server processes exited
normally.

### Live camera-to-plan reference

The reference robot setup used two 640×480 USB cameras at 30 FPS. The policy
server and robot client ran on the same DevKit, and the arm received 30
commands/s. Camera-to-plan starts at the newer timestamp of the two captured
frames and ends when the action plan is ready. These are medians over 30
measured requests after five warm-ups.

| Model | Model time | Camera picture to plan ready |
|---|---:|---:|
| ACT | 13.3 ms | 25 ms |
| GR00T N1.6 | 317 ms | 361 ms |
| GR00T N1.7 | 233 ms | 279 ms |

The original live-camera record retained the medians but not its individual
samples, so live-camera p95 and p99 are not reported. The distributions below
use retained fixture inputs and loopback transport on the matched runtime.

### Latency and load

Each latency series retains 35 individual requests: five warm-ups followed by
30 measured requests. p50 is the conventional median; p95/p99 use nearest rank.
CLI times include `Plan::execute` and its returned-output copy; fixture loading
is outside the timer. Server times cover `Plan::execute` after socket input
and uint8 decoding. The client timer wraps the existing `predict_raw` call,
including request packing and loopback TCP. It does not include live camera
capture, image preprocessing or robot control. All inputs were fixtures; ACT
returns 100×6 actions, GR00T 32×6 with four denoising steps and two views.
The internal noise tensors are 50×128 for N1.6 and 40×132 for N1.7; the
returned action horizon is distinct from those padded model dimensions.

| Model | CLI p50 / p95 / p99, ms | Server p50 / p95 / p99, ms | Loopback client p50 / p95 / p99, ms |
|---|---:|---:|---:|
| ACT | 14.4 / 14.5 / 14.7 | 13.23 / 13.39 / 13.45 | 24.18 / 24.71 / 25.95 |
| N1.6 | 315.8 / 316.1 / 316.4 | 315.40 / 315.62 / 315.68 | 325.34 / 326.01 / 326.13 |
| N1.7 | 232.2 / 233.0 / 233.1 | 231.65 / 232.17 / 232.57 | 248.80 / 250.07 / 250.37 |

These distributions supplement the historical camera-to-plan medians; they
are not new camera-to-plan measurements. CLI text has 0.1 ms resolution.

Load timing is process start to the server's `READY` line, sampled with a
50 ms readiness wait. Before the first load, `POSIX_FADV_DONTNEED` was applied
only to this inactive bundle's ELF files. `mincore` verified zero resident ELF
pages for all three models. This is **ELF-cache-cold**, not a reboot or globally
cold filesystem: libraries, metadata, constants and fixtures may remain cached.
The second value is an immediate fresh-process reload, with no eviction.
Cache retention is not guaranteed under memory pressure; N1.7's reload was
slower. These are one observed pair per model, not load-time percentiles.

| Model | ELF-cache-cold to READY | Immediate reload to READY |
|---|---:|---:|
| ACT | 1.859 s | 1.371 s |
| N1.6 | 8.239 s | 5.853 s |
| N1.7 | 9.234 s | 9.518 s |

### Execution stages

`POLIMA_PROFILE_PHASES=1` collected six additional inferences per model; the
following sums use the sixth. They are separate from the latency distributions.

| Model | Prepare | Encode, including device-chain upload | Separately timed upload | Execute | Separately timed download | Decode, including device-chain readback |
|---|---:|---:|---:|---:|---:|---:|
| ACT | no separate preparation | 1.061 ms | included in adjacent phase | 12.059 ms | included in adjacent phase | 0.005 ms |
| N1.6 | 0.032 ms | 2.855 ms | 0.248 ms | 308.845 ms | 0.351 ms | 0.996 ms |
| N1.7 | 0.063 ms | 2.883 ms | 0.220 ms | 223.384 ms | 0.285 ms | 1.625 ms |

For device chains, `encode_ms` includes writes into mapped device buffers and
`decode_ms` includes output reads. Their upload/download fields are unset,
not measurements of zero transfer cost. Other runner types time these phases
separately. Explicit plan operations such as packing, pixel unshuffle,
normalization and Euler updates are outside these sums and remain in the full
`stages` logs. The phase sums are not a complete wall-time decomposition.

Adding `POLIMA_PROFILE_STAGES=1` gives the following graph-call sums for the
sixth request after five warm-ups. This switch disables MLA queuing for those
calls, so these diagnostic figures must not be added to the normal queued
latency or interpreted as a breakdown measured under identical execution.

| Model | Vision graph calls | Language graph calls | Action-policy graph calls |
|---|---:|---:|---:|
| ACT | 6.391 ms | not applicable | 6.342 ms |
| N1.6 | 163.216 ms | 44.417 ms | 115.756 ms |
| N1.7 | 76.537 ms | 29.177 ms | 131.082 ms |

Vision includes both views and image connectors/merges. N1.6 language includes
Qwen and output normalization. N1.7 language includes its 16 language layers;
`vlln` and `vl_00..03` belong to the action-policy group with state projection
and four denoising passes. ACT's action-policy group includes its encoder and
decoder; ACT has no language model. Every individual graph-call timing and
name is retained, including repeated calls and one-time DiT K/V projection.

### Memory while serving

All values are KiB (`kB` in `/proc`). Total RAM was 6,039,804 KiB and CMA total
was 1,830,912 KiB throughout. The table uses the immediate-reload server run:
before server start, after `READY` before requests, and snapshots immediately
after each of the 30 measured requests. Additional 50 ms samples are retained.

| Model / metric | Before load | After load | During measured requests |
|---|---:|---:|---:|
| ACT MemAvailable | 4,440,644 | 4,379,248 | 4,372,732–4,380,620 |
| ACT CmaFree | 158,008 | 147,792 | 145,520–146,940 |
| ACT server VmRSS | not started | 80,844 | 81,240 |
| ACT server VmHWM | not started | 80,844 | 81,240 |
| N1.6 MemAvailable | 4,507,504 | 4,199,512 | 4,194,604–4,200,220 |
| N1.6 CmaFree | 74,092 | 33,012 | 27,276–37,344 |
| N1.6 server VmRSS | not started | 95,612 | 101,140 |
| N1.6 server VmHWM | not started | 95,612 | 101,140 |
| N1.7 MemAvailable | 4,417,320 | 4,097,500 | 4,060,116–4,081,052 |
| N1.7 CmaFree | 54,180 | 18,192 | 12,684–22,000 |
| N1.7 server VmRSS | not started | 176,112 | 183,008 |
| N1.7 server VmHWM | not started | 178,188 | 183,008 |

CMA is shared and affected by reclaimable filesystem pages; a before/after
`CmaFree` difference is not isolated model allocation or weight memory.
CLI RSS/HWM and the full before/after/execution RAM/CMA series were also
retained during validation.

| Fixture inference-client process | Steady VmRSS | Observed VmHWM |
|---|---:|---:|
| ACT | 34,620 | 38,852 |
| N1.6 | 35,752 | 40,548 |
| N1.7 | 37,412 | 48,076–48,104 |

Maximum sampled client HWM over the entire reload series, including teardown,
was 39,000 / 40,640 / 48,396 KiB for ACT / N1.6 / N1.7 respectively. Server
maxima were 81,240 / 101,140 / 183,008 KiB.

This is the existing `PolimaSOMClient` running in the measurement harness,
Python 3.11.2 / NumPy 2.2.6, using source snapshot
`256f5bd0fca66d46d9f7551211b8f0473a251ff5`. It is **not full robot-client
memory**: camera capture, preprocessing and motor control were not started.

### Compiled-device fidelity

All three packaged fixture checks passed on `7342f909`:

| Model | Cosine | Normalized mean absolute error | Normalized maximum absolute error |
|---|---:|---:|---:|
| ACT | 0.999992 | 0.0031 | 0.0179 |
| N1.6 | 0.999490 | 0.0096 | 0.0436 |
| N1.7 | 0.998736 | 0.0090 | 0.1135 |

The ACT threshold is cosine ≥ 0.999 and mean absolute error ≤ 0.01; the mixed
GR00T threshold is cosine ≥ 0.995 and normalized mean absolute error ≤ 0.03.
The displayed fixture metrics have the native CLI's printed precision. Both
cold-load and reload server outputs also passed the same comparisons.

ACT's 16 held-out frames were also replayed through the measured release payload
and runtime `7342f909`, using the retained inputs whose hashes match the host
PyTorch reference inputs. Each output contains 100×6 normalized actions.
All 16 pass cosine ≥ 0.999 and mean absolute error ≤ 0.01; maximum absolute
error is reported, not used as a separate pass threshold.

| ACT compiled-vs-PyTorch metric across 16 frames | Mean | Worst frame |
|---|---:|---:|
| Cosine | 0.99999216 | 0.99998428 |
| Normalized mean absolute error | 0.00301033 | 0.00423845 |
| Normalized maximum absolute error | 0.01902639 | 0.02953436 |

Outputs on runtime `3718bcedba6f9d1582b39740b8d085142cee93e3`
gave the same results. The table above reports the fresh `7342f909` run, not
the host StubPlan report.

### GR00T held-out frame comparison

Both GR00T bundles were evaluated on six held-out episode/frame pairs:
`(1, 0)`, `(79, 77)`, `(174, 245)`, `(236, 132)`, `(316, 185)`, and
`(365, 89)`. None belongs to the 769 training episodes; this was verified
against the saved training metadata. The source was the corrected-camera
revision of the public dataset linked in the artifact table.
Every frame uses a distinct noise tensor: N1.6 seeds 3600–3605 and N1.7
seeds 2600–2605. Both models were replayed on runtime `7342f909`.

The existing N1.6 exporter supplies preprocessing and the original PyTorch
model trace. The evidence harness makes the trace's seed explicit and retains
the actual initial noise tensor instead of reconstructing it by undoing an
Euler update. A primary-fixture gate reproduced image/state inputs byte for
byte and the expected action output exactly. Direct initial noise differs from
the old reconstructed fixture noise by at most `2.384185791015625e-07`;
the six new device requests use exactly the noise captured for their PyTorch
passes. Staged input SHA-256 values were verified before scoring.

The output is the first 32×6 actions, clipped as in the deployment plan,
denormalized, and decoded relative to the first five state coordinates.
`compare_normalized_actions` reverses that same decoding using the bundle's
per-timestep statistics before computing errors. Thresholds are fixed at
cosine ≥ 0.995 and normalized mean absolute error ≤ 0.03. Maximum error is
reported without a separate pass threshold.

| N1.6 compiled-vs-PyTorch metric, six frames | Mean | Worst frame |
|---|---:|---:|
| Cosine | 0.99894051 | 0.99576583 |
| Normalized mean absolute error | 0.01148189 | 0.02919246 |
| Normalized maximum absolute error | 0.05451763 | 0.13427590 |

All six pass; the worst-frame mean error is close to the 0.03 threshold.
This small numerical sample does not establish physical task success.
The checkpoint source revision is the N1.6 revision in the artifact table;
the reference environment used PyTorch 2.7.1+cu128, Transformers 4.51.3,
NumPy 1.26.4 and TorchCodec 0.4.0. Source trees were clean at capture time.

N1.7 uses the existing official `Gr00tPolicy` on CPU, with hooks recording
its vision input patches and exact initial action noise. Its primary-fixture
gate matched every input byte and the expected output exactly. The retained
export's `episode_index: 2` is a position in its sparse training view; that
position resolves to original episode 3. The gate uses original episode 3,
while the six held-out comparisons use the original episode IDs listed above.
The full 961-episode view was verified to map each row index to the same
original episode ID.

| N1.7 compiled-vs-PyTorch metric, six frames | Mean | Worst frame |
|---|---:|---:|
| Cosine | 0.99848910 | 0.99607023 |
| Normalized mean absolute error | 0.01154192 | 0.02374876 |
| Normalized maximum absolute error | 0.06798807 | 0.12890625 |

All six pass the same fixed 0.995 cosine / 0.03 normalized-mean-error bars.
The N1.7 reference environment used PyTorch 2.9.0+cu128, Transformers 4.57.3,
NumPy 1.26.4 and TorchCodec 0.8.0, with the clean N1.7 source revision in the
artifact table. Both models' staged input hashes matched their captured
PyTorch inputs. Per-frame results were retained, not only means.

### Evidence and reproduction

SiMa.ai retains the raw samples, inputs, seeds, outputs, memory snapshots and
SHA-256 hashes with the release-validation records. The runtime can be rebuilt
at the pinned revision with:

```bash
git archive 7342f90998eb517572151ac3e61bcbc6455e0b87 runtime \
  | tar -x -C "$BENCH"
cmake -S "$BENCH/runtime" -B "$BENCH/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BENCH/build" -j2
"$BENCH/build/polima_kernel_tests"
```

With an exact bundle installed, reproduce the fixture latency distribution and
packaged-reference check without cameras or robot motion:

```bash
polima_cli --models-dir /media/nvme/polima/models \
  --interactive --bundle "$BUNDLE_ID" --exclusive-control
bench 30 5
stages
check
quit
```

Studio **Performance** exposes the same 30-run, five-warm-up benchmark. For a
single packaged fixture check from a host, use:

```bash
polima run --bundle "$BUNDLE_ID" --fixture
```

The multi-frame comparisons require the original checkpoint, the pinned
dataset revision and the frame/seed selections listed above. They are separate
from the packaged fixture check.

## What each result establishes

| Result | Establishes | Does not establish |
|---|---|---|
| Compilation completed | ModelSDK produced every graph and PoLiMa packaged a bundle. | Numerical agreement or robot operation. |
| Export comparison | Exported output agrees with PyTorch for the recorded input and tolerances. | Compiled-device agreement. |
| Device fixture comparison | Compiled output agrees with its reference for the packaged input and tolerances. | General task success. |
| Physical run | The stated setup completed observed trials. | Performance in another camera layout, workspace or checkpoint. |

Keep these boundaries when reporting a new bundle. An absent result means
not measured, never zero or pass.

## Coverage and remaining measurements

| Evidence | ACT | GR00T N1.6 | GR00T N1.7 |
|---|---|---|---|
| Compilation, precision map, retained sizes | recorded | recorded | recorded |
| Installed ELF identity and component sizes | verified | verified | verified |
| Raw latency, five warm-ups plus 30 measured runs | CLI and server | CLI and server | CLI and server |
| Stage timings | phase and diagnostic graph calls | phase and diagnostic graph calls | phase and diagnostic graph calls |
| ELF-cache-cold and immediate-reload load pair | recorded | recorded | recorded |
| RAM/CMA and server RSS/HWM | recorded | recorded | recorded |
| Full camera/robot-client RSS/HWM | unmeasured | unmeasured | unmeasured |
| Individual compiled-graph error | unavailable | unavailable | unavailable; exported-partition errors recovered separately |
| Multi-frame compiled-versus-PyTorch | 16 frames pass on `7342f909` | six frames pass on `7342f909` | six frames pass on `7342f909` |
| Controlled physical success rate | unmeasured | unmeasured | unmeasured |

The bench DevKit did not have robot hardware attached, so this measurement
series does not add camera-to-plan distributions, full robot-client memory or
controlled physical trials. No success rate is inferred from fixture fidelity.

Per-graph compiled error remains distinct from the recorded end-to-end and
export errors: device-resident fused chains do not expose all individual graph
outputs through the existing stage dump. Peak build RAM/disk absent from the
retained GR00T build logs cannot be reconstructed from retained-tree sizes.
