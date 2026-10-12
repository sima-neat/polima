# GR00T on SO-101

Deploy a compiled GR00T N1.6 or N1.7 policy to Modalix, verify inference, then
run an SO-101 follower arm. The reference task is placing a grey eraser in a
white basket. A model trained for this task is not a general-purpose robot policy.

## Before you start

| Item | Requirement |
|---|---|
| Model | A compiled PoLiMa bundle for SO-101, with its model card, licence and validation results. |
| DevKit | Modalix with a supported SDK and NVMe storage; see [installation](../../docs/install.md). |
| Robot | SO-101 follower, power supply and USB connection to the DevKit. |
| Cameras | Workspace and wrist views matching the model's training configuration. |
| Host | PoLiMa deployment installation, SSH and rsync; Linux/macOS or Windows through WSL. |

A precompiled bundle does not require the training dataset, PyTorch checkpoint
or ModelSDK on the customer's host. To compile your own checkpoint, use the
[compiler guide](../../docs/your-own-model.md#groot) first.

The compiled models are planned for `simaai/GR00T-N1.6-SO101-PickPlace` and
`simaai/GR00T-N1.7-SO101-PickPlace` on Hugging Face. Publication is pending;
use the download command from the published model card when available. An
upstream NVIDIA checkpoint is not a compiled Modalix bundle.

## 1. Install and connect

Follow [Install PoLiMa](../../docs/install.md) on the host and DevKit. The DevKit
installer registers Studio; installing only the Python package does not create
`polima-studio.service`.

On the host, confirm SSH access using the DevKit account and address. Workbench
can save this connection under **DevKits**. Its address and SSH-key paths refer
to the Workbench host, not the laptop displaying the page.

Expected result: the connection check succeeds. An authentication failure needs
the correct account/key or first-time password setup; it is not a model error.

## 2. Deploy the bundle

Run in the host terminal (Ubuntu/WSL on Windows). Enter the downloaded bundle
folder containing `bundle.json`, and the complete DevKit login when prompted:

```bash
printf "Bundle folder: "
read -r bundle_dir
printf "DevKit login (username@address): "
read -r devkit_login
polima deploy --bundle "$bundle_dir" --devkit "$devkit_login" --start
```

Deployment copies and verifies the bundle, builds the matching runtime,
activates the model and starts its server. The complete sequence and recovery
options are in [Deploying bundles](../../docs/deploy.md).

Expected result: deployment finishes and the model server starts. No robot
motion is requested by this command.

## 3. Validate inference

In the same host terminal, using the same local bundle:

```bash
polima run --bundle "$bundle_dir" --devkit "$devkit_login" --fixture --repeat 20
```

Expected result: output agreement is reported against the packaged reference,
along with latency measurements. A numerical FAIL remains a FAIL even when
compilation completed. Review [validation results](../../docs/groot.md#reading-validation-results)
before interpreting the numbers. This test does not move the arm. The host must
reach the model server as well as SSH; a forwarded SSH port alone does not
forward inference traffic.

## 4. Open Studio and check the cameras

On the DevKit:

```bash
polima studio start
polima studio status
polima robot status
```

Open the Studio address printed by `start` from a computer that can reach the
DevKit. In **Operate**, select **Take control**, select the deployed bundle,
and assign the **Overhead camera** and **Wrist camera**. Select **Start Preview**
to verify which physical camera supplies each view. Select **Graceful Stop**
when finished, before starting another camera preview or a CLI robot run.

For a terminal-only camera check, run `polima robot preview` on the DevKit.
Open its printed address, then press **Ctrl-C** in that terminal to end preview
and release the cameras.

| View | Reference configuration |
|---|---|
| Workspace | Logitech C920 camera hint; downward-looking view of the eraser, basket and working area. N1.7 names this `front`, mapped from `observation.images.overhead`. |
| Wrist | Sonix camera hint; close view along the gripper approach, with fingers visible at the bottom of the sampled training frames. |
| Recorded video | 640 × 480 RGB, 30 frames/s. Model preprocessing is recorded separately in the bundle. |

Match orientation, framing and visible workspace to the model's reference data.
Check camera roles throughout representative episodes before choosing data for
a new build; folder names alone do not establish that the views are correct.
Use stable device paths when selecting cameras; numeric indices can change.

### Reference placement

| Measurement | Reference value |
|---|---|
| Arm end to basket | 32 cm |
| Overhead camera lens above tabletop | 49 cm |
| Basket dimensions (length × width × height) | 17.5 × 12 × 6 cm |
| Eraser dimensions (length × width × height) | 12.5 × 5 × 2 cm |

Reference measurements supplied for this setup. The 32 cm distance is measured
from the arm end to the basket, not from the robot base.

Frame the cameras as follows for the reference eraser task:

- **Overhead:** the basket and eraser are centered below the gripper, with the
  gripper entering from the top of the image. Keep the pickup area visible.
- **Wrist:** look along the gripper toward the workspace, with the gripper
  fingers visible at the bottom of the image.

Confirm camera roles and image orientation before running the policy.

## 5. Calibrate and run

On the DevKit, calibrate the connected arm once:

```bash
polima robot calibrate
```

Calibration belongs to that physical arm. Model normalization statistics,
compiler calibration samples and motor calibration serve different purposes.
The compiled bundle includes model statistics and a dataset-derived rest pose;
customers must generate their own motor calibration.

Stop any camera-only preview. Place the grey eraser in the visible pickup area
and keep the white basket at its reference position. After checking the arm's
workspace:

```bash
polima robot run --slow
```

The command asks for device selection and confirmation, moves to the bundle's
rest pose and starts the policy. GR00T `--slow` uses 15 commands/s; the default
uses 30 commands/s. These are motor command rates, not inference rates.
**Ctrl-C returns the arm to rest** and ends the robot run; it is not an
instantaneous emergency stop. See [robot operation](../../docs/robot.md) for
motion settings, completion behavior and troubleshooting.

GR00T disables automatic gripper-sequence completion by default and lets the
policy perform the placement and return. Leave **Automatic return** off in
Studio to use this behavior; [automatic return](../../docs/robot.md#optional-automatic-return)
is an optional per-run setting.

For a physical acceptance check, repeat the task from several positions in the
trained pickup area. Confirm grasp, placement and return on every trial, and
check that the next pickup is not interrupted by an early return. Record both
successful and failed trials. Compilation, fixture tests and physical task evaluation are
separate results.

## Stop the services

On the DevKit, after ending the robot run:

```bash
polima server stop
polima studio stop
```

The `setup/` directory contains SO-101 installation and setup tools. Architecture,
precision, dataset requirements and upstream references are in the
[GR00T guide](../../docs/groot.md).
