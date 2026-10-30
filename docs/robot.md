# Running the SO-101 arm

Everything here runs on the DevKit. The arm and both cameras plug into the
DevKit by USB. The robot client runs in the LeRobot environment that the
DevKit installer creates at `/media/nvme/lerobot`.

## Check the hardware

```bash
polima robot status     # the active bundle, the attached arm and cameras
polima robot preview    # show each camera in a browser, to check the views
```

Use stable device names (`/dev/serial/by-id/...`, `/dev/v4l/by-id/...`) when
you pass ports by hand: `/dev/ttyACM0` and `/dev/video0` can change after a
reconnect or reboot.

## Calibrate the arm

Once per arm, before its first run:

```bash
polima robot calibrate
```

It backs up the existing calibration, then runs LeRobot's calibration. The
file is `~/.cache/huggingface/lerobot/calibration/robots/so_follower/so-arm101.json`;
`--calibration-dir DIR` uses another folder.

A run uses this DevKit calibration. A bundle can also carry its own, set in
PoLiMa Studio, and a run then uses the bundle's. Without either,
`polima robot run` refuses to move the arm.

Use the calibration recorded for the connected arm. Do not reuse another
arm's calibration.

## Run

```bash
polima robot run
```

It asks for the arm's serial port and for each camera, asks for confirmation,
moves the arm to its rest pose, then hands over to the active bundle's model.
**Ctrl-C** stops the run and returns the arm to rest. The model server keeps
running until `polima server stop`.

| Option | Use |
|---|---|
| `--slow` | GR00T at 15 commands/s with its slow motion settings. Default is 30 commands/s; `--fast` selects the default explicitly. |
| `--fps N` | Override the command rate, 1 to 60. Camera capture rates stay unchanged. |
| `--no-auto-complete` | Disable the gripper-sequence completion detector. |
| `--bundle DIR` | Run a bundle other than the active one. |
| `--robot-port PATH`, `--overhead-camera PATH`, `--wrist-camera PATH` | Skip the questions. |
| `--task N` | Which of the bundle's instructions to use (default 0, the first). |
| `--max-relative-target DEG` | How far one command may move a joint: a whole number of degrees from 1 to 30 (default 12). |
| `--yes` | Do not ask for confirmation. |

## After each pick

GR00T uses policy-controlled placement and return by default. When the policy
commands a return near the dataset rest pose, the client holds that pose for
two seconds before continuing. This rest hold is separate from the optional
gripper-sequence completion detector.
**Ctrl-C** ends the run and returns the arm to rest.

### Optional automatic return

To return to rest after a detected gripper release:

```bash
POLIMA_AUTO_COMPLETE=1 polima robot run
```

In Studio, enable **Automatic return**. The detector uses a gripper
open–close–open sequence; it does not verify that the object reached the basket.
With **Repeat task** enabled, the client waits two seconds at rest before
continuing. Set `POLIMA_AUTO_REPEAT=0` to stop after a detected cycle.

The command above applies to one run. The next plain `polima robot run` uses
the GR00T default. ACT enables automatic return by default; use
`--no-auto-complete` to disable it. Rest-position holding and return on
**Ctrl-C** remain available in either mode.

## Motion settings

GR00T uses fast motion settings by default and slow settings with `--slow`
([groot.md](groot.md#running)). Any of them can be changed for one run
by setting it in the same terminal before `polima robot run`:

Pi0.5 bundles use 30 commands per second, ask for the next 50-move plan when
half of the current one is left, and reuse one fixed noise sample for the run.
A three-tap filter smooths the arm targets inside each plan without changing
its endpoints or gripper commands. Overlapping plans are crossfaded. Pi0.5
reads its instruction as text:
`POLIMA_PI05_TASK="..." polima robot run` replaces the bundle's instruction.

| Variable | What it does |
|---|---|
| `POLIMA_CHUNK_THRESHOLD` | Each model answer is a plan of upcoming moves. Ask for the next plan when this fraction of the current one is left. |
| `POLIMA_SMOOTH_MOVES` | Spread the jump from the old plan to the new one over this many commands. |
| `POLIMA_SMOOTH_SHAPE` | `smoothstep`: make that spreading an S-curve instead of equal steps. |
| `POLIMA_SUBSTEPS` | Send each command as this many smaller steps. |
| `POLIMA_RESET_EASE` | `smoothstep`: ease into and out of the return to rest. |
| `POLIMA_GROOT_NOISE` | Plan noise: `zero`, `fixed` (one standard-normal sample reused for the run), or `random` (a new sample for every plan). |
| `POLIMA_CHUNK_FILTER` | `fir3`: smooth interior arm targets in each plan while preserving its endpoints and gripper commands; `off` disables it. |
| `POLIMA_AGGREGATE` | How an incoming plan is combined with queued commands. Pi0.5 uses `crossfade` across their overlap. |

## Camera recovery

If a camera drops out during a run, the client stops inference, opens the gripper,
returns to the bundle's rest pose, and closes the gripper. After that client has
shut down, it waits up to 60 seconds for both cameras to reconnect and provide
frames, then starts a fresh client. Old action queues are discarded. There are
at most three automatic restarts per run.

Ctrl-C or Studio's Graceful Stop cancels pending retries. A failed return to rest,
a motor error, or failed cleanup stops the run rather than restarting it.
Recovery cannot resume with a missing camera and does not repair USB faults.

## Troubleshooting

| What you see | What to do |
|---|---|
| A camera or the motors drop out ("exceeded maximum consecutive read failures", "There is no status packet") | A USB connection dropped. Use a powered USB hub and secure the cables, especially along the arm. |
| The arm consistently moves the wrong way | The cameras are in the wrong roles, or placed differently from the recording. |
| The arm misses by the same amount every time | Recalibrate (see above). |
| The run stops before moving and asks for `polima robot calibrate` | The arm has no calibration on this DevKit yet; calibrate it once. |
| The model server does not start | `polima server status`; on `MLA_LOAD_FAILED`, see [deploy.md](deploy.md#accelerator-reset-and-sudo). |

Run summaries stream the event log rather than loading it into RAM. Counts and
maximum latency are exact. For more than 4,096 valid latency measurements per
metric, p50/p95 use a deterministic reservoir sample and the summary marks them
with `percentiles_approximate: true` and `sample_size`.

If motor communication fails during shutdown, PoLiMa still attempts to close
all camera and transport handles and disables automatic retry for that client.
It reports the motor torque state as unconfirmed; losing the motor connection
can prevent both returning to rest and disabling torque.

### Logging for normal runs

Normal SO-101 runs do not record per-action or per-inference histories. Warnings
and errors are saved to `$POLIMA_ROOT/var/log/robot-errors.log`, with two rotated
backups (2 MiB each; approximately 6 MiB total). Terminal status remains available.
The optional queue-size plot keeps only the latest 900 samples in memory.

For a diagnostic run, explicitly enable detailed recording:

```bash
POLIMA_RECORD_RUN=1 polima robot run
```

Each detailed event log is capped at 8 MiB, plus its truncation notice and final
status. Once capped, counts and timing summaries cover only the recorded portion.
Normal runs do not add entries to Studio's recorded run history. Existing logs
are preserved; enabling diagnostics repeatedly creates separate run folders.
