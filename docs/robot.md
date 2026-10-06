# Running the SO-101 arm

Everything here runs on the DevKit. The arm and both cameras plug into the
DevKit by USB. `polima robot` runs in the LeRobot environment that the DevKit
installer creates at `/media/nvme/lerobot`.

## Check the hardware

Cheapest first:

```bash
polima robot ports      # which arm and cameras are attached, and which role each fills
polima robot doctor     # the same, plus the model server and the arm calibration
polima robot preview    # open each camera and report what it actually delivers
```

Use stable device names (`/dev/serial/by-id/...`, `/dev/v4l/by-id/...`) when
you pass ports by hand: `/dev/ttyACM0` and `/dev/video0` can change after a
reconnect or reboot.

## Calibrate the arm

Calibrate once per arm, before its first run:

```bash
polima robot calibrate
```

It backs up the existing calibration, then runs LeRobot's calibration. The
calibration file is
`~/.cache/huggingface/lerobot/calibration/robots/so_follower/so-arm101.json`;
`--calibration-dir DIR` uses another folder.

A run uses this DevKit calibration. A bundle can also carry its own, set from
PoLiMa Studio; a run then uses the bundle's. Without either, `polima robot run`
refuses to move the arm.

A shifted calibration moves every target the model sends by the same offset.
If an arm that worked starts missing in one direction, check the calibration
first.

## Run

```bash
polima robot run
```

It asks for the arm and the cameras, asks for confirmation, moves the arm to
its rest pose, then hands over to the active bundle's model. **Ctrl-C** stops
the run and returns the arm to rest.

Useful options:

| Option | Use |
|---|---|
| `--fps N` | Position commands per second. ACT uses the bundle's default; GR00T uses 15. |
| `--bundle DIR` | Run a bundle other than the active one. |
| `--robot-port`, `--front-camera`, `--wrist-camera` | Skip the questions. |
| `--task TEXT` | The instruction, for language-conditioned models. |
| `--no-live-view` | Do not open the camera window. |

## Motion settings

All are off unless set, in the same terminal as `polima robot run`. GR00T was
tested with all of them; see [groot.md](groot.md).

| Variable | What it does |
|---|---|
| `POLIMA_CHUNK_THRESHOLD=0.5` | Each model answer is a plan of upcoming moves. Ask for the next plan when this fraction of the current one is left. |
| `POLIMA_GROOT_NOISE=zero` | GR00T starts every plan from the same (zero) noise, so the same view gives the same plan. |
| `POLIMA_SMOOTH_MOVES=5` | Spread the jump from the old plan to the new one over 5 moves. |
| `POLIMA_SMOOTH_SHAPE=smoothstep` | Make that spreading an S-curve instead of equal steps. |
| `POLIMA_SUBSTEPS=2` | Send each command as 2 smaller steps. |
| `POLIMA_RESET_EASE=smoothstep` | Ease into and out of the return to rest. |

Each command may move a joint at most 12° from where it is.

## Troubleshooting

| What you see | What to do |
|---|---|
| A camera or the motors drop out ("exceeded maximum consecutive read failures", "There is no status packet") | A USB connection dropped. Use a powered USB hub and secure the cables, especially along the arm. |
| The arm consistently moves the wrong way | The cameras are in the wrong slots or placed differently from training. |
| The arm misses by the same amount every time | Recalibrate (see above). |
| The model server is not running | `polima server status`, then `polima server start`; on `MLA_LOAD_FAILED`, see [deploy.md](deploy.md). |
