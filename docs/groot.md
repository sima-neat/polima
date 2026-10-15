# NVIDIA GR00T on the Modalix DevKit

Reference for GR00T N1.6 and N1.7 bundles. The step-by-step guides are
[examples/so101](../examples/so101/README.md) (our compiled models) and
[your-own-model.md](your-own-model.md) (your checkpoint).

## Choose your starting point

| You have | Start here | Dataset needed? |
|---|---|---|
| A compiled PoLiMa GR00T bundle | [Install and run on SO-101](../examples/so101/README.md) | No training dataset; the bundle includes inference fixtures and model metadata. |
| A fine-tuned N1.6 or N1.7 checkpoint | [Prepare the compiler and import your model](your-own-model.md#gr00t) | Yes, compatible representative observations and metadata for export. |
| A base model and demonstrations | NVIDIA's [SO100/SO101 training example](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/SO100/README.md) | Training happens upstream; PoLiMa starts from the resulting checkpoint. |

A compiled package is specific to its model version, embodiment, camera mapping,
action horizon and precision. A different SO-101 checkpoint is a new build.
The reference compiled bundles are on Hugging Face: [GR00T N1.6](https://huggingface.co/simaai/GR00T-N1.6-SO101-PickPlace) and
[GR00T N1.7](https://huggingface.co/simaai/GR00T-N1.7-SO101-PickPlace).
If repository access is restricted, sign in with an authorized account. Download instructions
are in the model cards; uploaded revisions are recorded in the
[SO-101 guide](../examples/so101/README.md#reference-models).

## Machines and environments

| Location | Responsibility |
|---|---|
| Laptop browser | Upload folders and view Workbench. No PyTorch or ModelSDK installation needed. |
| Linux x86_64 build host | Workbench, ModelSDK, model-specific export environment, checkpoint and dataset storage. |
| Modalix DevKit | PoLiMa runtime, Studio, model server, cameras and SO-101 client. |

Workbench paths refer to the build host. An uploaded folder is copied there;
its generated server path is expected and is different for each installation.
Install the matching GR00T interpreter, source and backbone once and
[save their paths](your-own-model.md). Uploading weights does not install software.
Use the source revisions documented below; upstream `main` can change APIs.

## What runs where

GR00T looks at two camera pictures and the arm's joint positions, reads a text
instruction (for example "Place the grey eraser in the white basket"), and
predicts an **action chunk**: a sequence of joint targets for the arm and
gripper (32 steps in the reference release bundles). The arm
carries out the plan while the next one is computed, so it keeps moving.

The neural-network layers run on the Modalix accelerator: the vision encoder,
the language model and the action generator. The DevKit's processor prepares
the camera pictures, computes the action generator's per-pass scaling values,
does the small steps between accelerator calls, and drives the motors.

The action generator reuses the vision and language results across its four
refinement passes and runs four of its layers per accelerator call.

## Compiling

`polima compile --policy groot` takes a GR00T N1.6 or N1.7 checkpoint and its
LeRobot training dataset and produces one bundle. Setup and the command are in
[your-own-model.md](your-own-model.md#gr00t). The stages:

| Stage | What it does |
|---|---|
| `export` | Runs the checkpoint once in PyTorch as the reference, cuts the model into parts the compiler accepts, and records calibration samples from the dataset. |
| `dit` | Prepares the action generator: four layers per part, scaling values computed ahead of time. |
| `compile` | Compiles every part with the ModelSDK (`-j` parts at once). |
| `bundle` | Packs the bundle: compiled parts, run plan, camera order, test data, robot client and rest pose. |
| `check` | Only with `--check-devkit`: runs the bundle's test input on a DevKit and compares with PyTorch. |

`--stop-after <stage>` stops early; `--resume` continues a stopped build.
Export requires the matching Isaac-GR00T source, Python environment and
backbone assets. Workers use the Hugging Face cache in offline mode by
default. See [backbone preparation](your-own-model.md#download-the-backbone-before-export).
The dataset supplies representative observations and bundle metadata, even
for BF16 export; compilation does not retrain the model.

N1.7 export also replays the complete ONNX policy against its PyTorch reference.
`export/onnx_verification_report.json` records normalized-action similarity and
error, with physical-unit errors reported separately. PoLiMa reports PASS when
cosine similarity is at least **0.98** and mean normalized absolute error is at
most **0.015**. A numerical FAIL is retained in the report and logged as a
warning; compilation and bundle creation continue. Export exceptions and actual
compiler failures still stop the build. These are PoLiMa export thresholds, not
ModelSDK settings or task-accuracy percentages. Device validation below is a
separate check with its own precision-specific thresholds.

**Precision.** `mixed` (default) stores the transformer weights as 8-bit
integers and computes in 16-bit floating point. It is the precision tested on
the arm. `bf16` uses 16-bit floating point. Measure numerical fidelity and latency
for the checkpoint and target device to compare the two.

**Pass marks** for the device check, over the complete action chunk:

| Precision | Similarity to PyTorch (cosine) | Mean difference |
|---|---|---|
| mixed | at least 0.995 | at most 0.03 |
| bf16 | at least 0.999 | at most 0.01 |

Our SO-101 models reach 0.9987 to 0.9995 in mixed precision.

## Preparing your dataset

GR00T reads LeRobot dataset format **v2.0 or v2.1**. `lerobot-record` writes
format v3.0; convert it with Isaac-GR00T's
[`scripts/lerobot_conversion`](https://github.com/NVIDIA/Isaac-GR00T/tree/main/scripts/lerobot_conversion)
(`convert_v3_to_v2.py`, set up as its README describes). GR00T also needs a
`meta/modality.json` that names the parts of the robot state and the cameras:

- **state and action:** two groups, `single_arm` (the arm joints) then
  `gripper` (one value), in the same order and units in both;
- **video:** two cameras, `front` (the overhead camera, `original_key`
  `observation.images.overhead`) then `wrist`;
- **arm actions are relative:** each arm move is learned as a change from the
  current position; the gripper is absolute.

The camera order in `modality.json` is the order the model expects. PoLiMa
reads it from the checkpoint and the dataset, and `polima robot run` feeds the
cameras in that order.

For N1.7 export, provide at least three complete episodes. The exporter
selects 16 valid observations from each of the first three episodes, uses the
first two episodes for calibration and reserves the third as an evaluation
split. Each episode must include the model's image history and action horizon
plus at least 16 valid sampling positions. Preserve the episode lengths and
indices in `meta/episodes.jsonl`; do not shorten videos independently of the
metadata. The packaged reference uses one observation and is not a task-level
evaluation of the held-out episode.

Our models were fine-tuned with NVIDIA's Isaac-GR00T code (N1.6 at commit
`9b37aa1`, N1.7 at `51d4c89`) with these settings:

| Setting | Value |
|---|---|
| Start from | the NVIDIA base model, vision-language part frozen |
| Trained parts | projector and action generator |
| Steps | 50,000 |
| Learning rate | 0.0001, 5% warm-up |
| Batch | 1 sample, 8 gradient-accumulation steps |
| Plan length | 32 (`delta_indices` 0 to 31) |
| Colour augmentation | brightness 0.3, contrast 0.4, saturation 0.5, hue 0.08 |

## Running

`polima robot run` on the DevKit drives the arm with the active bundle. For
GR00T it uses these motion settings, tested on the SO-101:

| | Default | `--slow` |
|---|---|---|
| Commands to the arm per second | 30 | 15 |
| Next plan requested when this much of the current plan is left | 60% | 50% |
| A new plan is blended in over | 10 commands | 5 commands |
| Each command sent as | 1 step | 2 half-steps |

`--fast` is accepted as an alias for the default GR00T settings.
`--fps` remains available for an explicit command-rate override.

Both start every plan from the same (zero) noise, so the same view gives the
same plan, and ease into and out of the return to rest. Each setting can be
overridden for one run with an environment variable ([robot.md](robot.md#motion-settings)).

## Performance

Measured on a Modalix DevKit driving an SO-101 arm with two USB cameras, mixed
precision, one 32-move plan:

| | GR00T N1.6 | GR00T N1.7 |
|---|---|---|
| Model time on the accelerator | 317 ms | 233 ms |
| Camera picture to plan ready | 364 ms | 285 ms |

These are reference-setup measurements. A newly compiled bundle needs its own
fidelity check and benchmark; model geometry, precision and runtime version
affect the result. Published model cards describe the exact artifact tested.

## Troubleshooting

| What you see | What to do |
|---|---|
| The server fails to start; its log says `MLA_LOAD_FAILED` | Follow [accelerator recovery](deploy.md#accelerator-reset-and-sudo); recovery affects shared DevKit services. |
| The arm consistently moves the wrong way | Check camera roles and framing, joint order, action units and calibration against the model contract; the symptom alone does not identify the cause. |
| "Goal position clamped" warnings | The requested joint target exceeded the configured per-command limit and was reduced. This does not prove object contact or a successful grasp. If motion is unexpected, stop and inspect calibration, observations and requested targets. |
| A camera or the motors drop out ("exceeded maximum consecutive read failures", "There is no status packet") | A USB connection dropped. Use a powered USB hub and secure the cables along the arm. |
| `polima compile` says the export needs CUDA | GR00T N1.6 export needs an NVIDIA GPU. |

## Limitations

- Tested on the arm: one pick-and-place task on an SO-101, GR00T N1.6 and N1.7
  in mixed precision.
- The robot client drives the SO-101.
- An all-8-bit build (`--precision w8a8`) compiles but is not accurate enough
  to pass the check.

## Licence

GR00T models are distributed by NVIDIA under the NVIDIA License. Bundles made
from GR00T checkpoints are derived from those models and may be used for
research and evaluation only, under the same licence. See NVIDIA’s published
[N1.6 licence](https://huggingface.co/nvidia/GR00T-N1.6-3B/blob/main/LICENSE)
and [N1.7 licence](https://huggingface.co/nvidia/GR00T-N1.7-3B/blob/main/LICENSE).
Redistributed model packages must retain the complete applicable licence and notices.

## Reading validation results

| Result | What it establishes | What it does not establish |
|---|---|---|
| Export PASS / FAIL | ONNX output agreement with a recorded PyTorch reference under the recorded tolerances. | Grasp success or accuracy across the full dataset. |
| Compilation completed | ModelSDK produced the compiled graphs and PoLiMa packaged them. | A validation PASS or successful execution on Modalix. |
| Device fixture PASS / FAIL | Compiled output agreement for the packaged input on the tested runtime. | Camera configuration or physical robot performance. |
| Robot evaluation | Observed task outcomes for a stated robot, workspace and number of trials. | Generalization to another workspace. |

Keep `export/onnx_verification_report.json`, `compiled/compile_results.json`
and, when requested, `check/report.json` with the build. The Workbench reads
nearby reports linked to matching artifacts; copying only the bundle does not
currently guarantee that these build reports travel with it. Include the
reports and their test conditions explicitly when preparing a public package.
An absent measurement means **not measured**, never zero or PASS.

Report camera capture rate, policy inference rate and robot command rate
separately. Repeating a packaged fixture measures inference under that test's
conditions; it does not include the full camera-to-motion loop. Record action
horizon, precision, runtime version, sample count and timing boundaries with
any latency claim.

## Upstream references

NVIDIA's guides cover training and the upstream policy interface. PoLiMa's
Modalix runtime and commands are documented here; NVIDIA's GPU server commands
do not launch a compiled PoLiMa bundle.

- [Data preparation](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md)
- [Policy API](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/policy.md)
- [Hardware guidance](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md)
- [Real-world deployment](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/real_world_deployment.md)
- [SO100/SO101 example](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/SO100/README.md)
