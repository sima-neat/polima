# Running NVIDIA GR00T on a SiMa.ai Modalix DevKit

This guide explains how to run a prebuilt NVIDIA GR00T N1.6 or N1.7 bundle on a
SiMa.ai Modalix DevKit, driving an SO-101 arm. Compiling your own GR00T
checkpoint with `polima compile` comes in the next PoLiMa version.

## What GR00T does on Modalix

GR00T looks at two camera pictures and the arm's current joint positions,
reads a text instruction (for example "Place the grey eraser in the white
basket"), and predicts a **plan**: the next 32 joint targets for the arm and
gripper. The arm carries out the plan while the next one is being computed, so
it keeps moving instead of stopping to wait.

The model's neural-network layers run on the Modalix accelerator: the vision
encoder, the language model and the action generator. The DevKit's processor
does the rest:

- captures and prepares the camera pictures;
- computes the action generator's scaling values for each refinement pass;
- does the small steps between accelerator calls, such as each refinement
  update and converting the plan into joint units;
- sends the joint targets to the motors.

## The prebuilt bundles

| | GR00T N1.6 | GR00T N1.7 |
|---|---|---|
| Size | 3.3 GB | 3.4 GB |
| NVIDIA base model | `nvidia/GR00T-N1.6-3B` | `nvidia/GR00T-N1.7-3B` |
| Fine-tuning code | NVIDIA Isaac-GR00T, commit `9b37aa1` | NVIDIA Isaac-GR00T, commit `51d4c89` |
| Task | pick up the grey eraser and place it in the white basket | same |
| Plan length | 32 moves | 32 moves |
| Precision | mixed | mixed |

Both were fine-tuned for 50,000 steps on the same SO-101 recordings. SiMa.ai
provides the bundle folders.

**Mixed precision** stores the model weights as 8-bit integers and computes in
16-bit floating point. It is the only precision tested on a robot. A full
16-bit build also works but is slower. An all-8-bit build is
not supported, because it loses too much accuracy.

**Speed-ups inside these bundles.** The action generator reuses the vision and
language results across its four refinement passes, runs four of its layers per
accelerator call, and its per-pass scaling values are computed on the DevKit's
processor. Its actions stay within 0.4° of a straightforward compile of the
same checkpoint.

**Licences:** GR00T models are distributed by NVIDIA under NVIDIA's model
licence. The bundles, and any bundle made from a fine-tuned GR00T checkpoint,
are subject to that licence.

## Measured performance

Measured on a Modalix DevKit driving an SO-101 arm with two USB cameras, during
real pick-and-place runs on 2026-10-05:

| | GR00T N1.6 | GR00T N1.7 |
|---|---|---|
| Model time per plan (on the accelerator) | 317 ms | 233 ms |
| Camera picture to plan ready (complete loop) | 364 ms | 285 ms |

## What you need

1. **A Modalix DevKit** with SiMa.ai SDK 2.1.2 or 2.1.3, and PoLiMa installed on
   it (see the DevKit install step in the [README](../README.md)).
2. **An SO-101 arm**, connected to the DevKit by USB, calibrated
   ([robot.md](robot.md)).
3. **Two USB cameras**: one looking at the workspace from above or in front
   (the "front" camera) and one on the wrist. 640×480 cameras work.
4. **A GR00T bundle folder** from the table above.

## Install a bundle on the DevKit

From the host, with `POLIMA_BOARD` set ([deploy.md](deploy.md)):

```bash
polima deploy --bundle /path/to/<bundle-folder> --start
```

Or upload it from a browser with PoLiMa Studio ([polima-studio.md](polima-studio.md)).

Check the model without the robot:

```bash
polima run --bundle <bundle-name> --fixture --repeat 20   # checks the answer and reports the time
```

## Run the robot

On the DevKit:

```bash
export POLIMA_GROOT_NOISE=zero             # same starting point for every plan: steadier motion
export POLIMA_CHUNK_THRESHOLD=0.5          # ask for the next plan when half of the current one is left
export POLIMA_SMOOTH_MOVES=5               # blend each new plan in over 5 moves
export POLIMA_SMOOTH_SHAPE=smoothstep      # with an S-shaped blend (gentle start and end)
export POLIMA_SUBSTEPS=2                   # send each move in two half-steps: no stop-start shiver
export POLIMA_RESET_EASE=smoothstep        # ease into and out of the return to rest
polima robot run --fps 15
```

`polima robot run` asks for the arm's serial port, then each camera in the
bundle's order (the front camera first, then the wrist camera), then for
confirmation before the arm moves. It starts the model server if it is not
already serving this bundle. The arm first moves to its rest pose, then the
policy takes over. **Ctrl-C** stops the run and returns the arm to rest; the
model server keeps running until `polima server stop`.

What these settings do:

- **15 moves per second** with 32-move plans: each plan covers about 2 seconds
  of motion.
- **Next plan at half:** the next plan is ready before the current one runs
  out, so the arm does not stop to wait.
- **Plan blending and half-steps:** these remove the jolt when one plan hands
  over to the next, and the shiver between moves.
- **Fixed starting noise:** GR00T builds each plan from a random starting
  point; using the same one every time makes back-to-back plans agree more.

## Your robot

On each step the robot side supplies two camera pictures and the current joint
positions, and gets back 32 joint targets. Getting these right decides whether
the robot works.

**Cameras**

- Connect the same views as in training and assign them correctly when
  `polima robot run` asks. Swapped cameras do not cause an error message: the
  arm simply moves in the wrong direction.
- Keep the camera positions and angles as in the recordings.
- The bundle resizes the pictures itself; 640×480 is what we use.

**Joint positions**

- Same joints, same order and same units as in the training data: degrees for
  the arm and 0–100 for the gripper on the SO-101.
- Use a careful motor calibration. A shifted calibration moves every target by
  the same offset.

**Joint targets (the plan)**

- 32 rows, one per move, each with all the joints, as absolute targets.
- The robot client limits each move to 12° from the arm's current position.

## Preparing your own GR00T model

Compiling your own checkpoint arrives in the next PoLiMa version. To be ready
for it, record and fine-tune like this.

Record demonstrations with LeRobot in its dataset format **v2.0 or v2.1**.
GR00T needs a `meta/modality.json` that names the parts of the robot state and
the cameras:

- **state and action:** two groups, `single_arm` (the arm joints) then
  `gripper` (one value), in the same order and the same units in both;
- **video:** two cameras, named `front` (the overhead or scene camera) and
  `wrist`;
- **arm actions are relative:** the model learns each arm move as a change from
  the current position. The gripper is absolute.

Fine-tune with NVIDIA's Isaac-GR00T code at the commits above. These are the
settings of the prebuilt models:

| Setting | Value |
|---|---|
| Start from | the NVIDIA base model, with its vision-language part frozen |
| Trained parts | projector and action generator only |
| Steps | 50,000 |
| Learning rate | 0.0001, 5% warm-up |
| Batch | 1 sample, 8 gradient-accumulation steps |
| Plan length | 32 (`delta_indices` 0 to 31) |
| Colour augmentation | brightness 0.3, contrast 0.4, saturation 0.5, hue 0.08 |

## Troubleshooting

| What you see | What to do |
|---|---|
| The server fails to start; its log says `MLA_LOAD_FAILED` | `polima server stop`, then `sudo /usr/bin/fix_devkit_runtime.sh`, then start again. Needed after a crash or when switching models. |
| The arm consistently moves the wrong way | The cameras are in the wrong slots, or the camera views differ from training. Check the camera assignment and positions. |
| Many "goal position clamped" warnings while the gripper holds an object | Normal: the gripper is asked to close further, but the object stops it. |
| A camera or the motors drop out ("exceeded maximum consecutive read failures", "There is no status packet") | A USB connection dropped. Use a powered USB hub and secure the cables, especially along the arm. |
| A short pause between plans | Check that the settings above are exported in the same terminal as `polima robot run`. |

## Limitations

- Robot testing so far: one task (pick and place) on an SO-101 arm, with GR00T
  N1.6 and N1.7 in mixed precision.
- The robot client drives the SO-101 only.
- Compiling your own GR00T checkpoint is not in this version.
