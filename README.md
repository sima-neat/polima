# PoLiMa

**Po**licy + Si**Ma**: run robot policies (ACT, GR00T) on the SiMa.ai
Modalix DevKit.

PoLiMa takes a trained robot policy and gets it driving a robot arm from the
Modalix chip:

```
checkpoint -> export -> compile -> bundle -> deploy -> serve -> robot
```

It is the robot-policy counterpart of [LLiMa](https://github.com/sima-neat/llima),
which does the same for language models. PoLiMa does not train models. You train
with LeRobot (or NVIDIA's GR00T tools) as usual, and PoLiMa starts from the
checkpoint.

- **Export and compile** (on an x86 computer): turn an ACT checkpoint
  into a *bundle*: a folder with the compiled model, its settings and the robot
  client.
- **Deploy** (from the x86 computer): copy a bundle to the DevKit and build
  PoLiMa's runtime there.
- **Serve and drive** (on the DevKit): run the model on the chip and drive an
  SO-101 arm with two USB cameras.
- **Studio and Workbench**: web pages for running the robot and keeping track
  of models, builds and robot runs.

Version 0.2.0 supports:

| Model | Compile it yourself | Run on the DevKit | On the SO-101 arm |
|---|---|---|---|
| ACT | yes | yes | tested |
| GR00T N1.6, N1.7 | next version (prebuilt bundles for now) | yes | tested (mixed W8A16 bundles) |

---

## What you need

**To compile models (x86 host)**
- An x86_64 Linux computer with Python 3.10 to 3.12.
- The **SiMa ModelSDK** (model compiler). It is licensed separately and is not
  part of this repository; get it from the SiMa Developer Portal. ACT
  compilation was verified with ModelSDK 2.1.0.
- The training dataset of your checkpoint. PoLiMa uses a few samples from it to
  calibrate the compiled model.

**To run models (DevKit)**
- A SiMa.ai Modalix DevKit with one of these SDK versions:

  | DevKit SDK | `sima_lmm` runtime |
  |---|---|
  | 2.1.2 | 0.3.0 |
  | 2.1.3 | 0.4.0 |

- For robot runs: an SO-101 follower arm and two USB cameras (one that sees the
  workspace from the front, and one on the wrist), plugged into the DevKit.
- Internet access on the DevKit for the first install. It downloads LeRobot
  0.4.4 and its dependencies.

You do not need the ModelSDK to deploy and run a prebuilt bundle.

---

## Install

### On the x86 host

To deploy and run prebuilt bundles only:

```bash
git clone https://github.com/sima-neat/polima.git
cd polima
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[deploy]"
polima doctor
```

To also compile ACT checkpoints, build the full host environment
instead. It installs a CPU-only PyTorch, and LeRobot from a LeRobot source
checkout. Use the checkout your checkpoints were trained with; PoLiMa is tested
with LeRobot 0.6.1.

```bash
make venv LEROBOT_SOURCE=/path/to/lerobot
source .venv/bin/activate
export MODEL_COMPILER_BIN=/path/to/model-compiler/bin   # folder holding the ModelSDK's python
polima doctor                                           # checks that the compiler is found
```

`polima` and `python -m polima` are the same command.

Tell PoLiMa where your DevKit is, once:

```bash
export POLIMA_BOARD=sima@<board-address>
```

Or copy `.polima.yaml.example` to `.polima.yaml` and set `board.host`.

### On the DevKit

```bash
ssh sima@<board-address>
git clone https://github.com/sima-neat/polima.git
cd polima
./scripts/install_polima_modalix.sh
```

The installer:
- creates the LeRobot 0.4.4 environment at `/media/nvme/lerobot`, or reuses it,
- installs PoLiMa and PoLiMa Studio into it,
- builds the runtime (`polima_server`, `polima_cli`) under `/media/nvme/polima`,
- makes the board command `polima` available.

On the DevKit, `polima` is the board command: `polima activate`,
`polima server`, `polima robot`, `polima studio`. The `compile` and `deploy`
commands run on the host.

---

## Quickstart: ACT

**1. Compile (host).** Export the checkpoint, compile it, and pack a bundle:

```bash
polima compile --policy act \
    --checkpoint /path/to/outputs/train/<run>/checkpoints/100000/pretrained_model \
    --build-dir ~/polima-builds/act-my-task -j 3
```

PoLiMa reads the training dataset's path from the checkpoint. That is the path on
the machine that trained it, so pass `--dataset-root /path/to/dataset` if the
checkpoint or the dataset has moved since.
`-j 3` compiles three parts of the model at once (each needs about 1.6 GB of
memory). The bundle is written to `outputs/bundles/<bundle-id>/`, and the
command prints its id.

**2. Deploy (host).** Copy the bundle to the DevKit and make it the active one:

```bash
polima deploy --bundle <bundle-id>
```

**3. Run the robot (DevKit).**

```bash
polima robot run
```

This starts the model server for the active bundle, asks you to pick the arm and
the two cameras, asks for confirmation, and then drives the arm. Ctrl-C stops
the robot; the model server keeps running until `polima server stop`.

Before the first run on a new arm, calibrate it (see [docs/robot.md](docs/robot.md)).

You can also check the model without the robot, from the host:

```bash
polima deploy --bundle <bundle-id> --start   # deploy and start the model server
polima run --bundle <bundle-id> --fixture    # send the bundle's test input and check the answer
```

## Quickstart: a prebuilt GR00T bundle

**1. Deploy (host).**

```bash
polima deploy --bundle /path/to/<groot-bundle-folder>
```

You can also install a bundle from a browser with PoLiMa Studio (see below).

**2. Run with the SO-101 recipe (DevKit).**

```bash
export POLIMA_GROOT_NOISE=zero POLIMA_CHUNK_THRESHOLD=0.5
export POLIMA_SMOOTH_MOVES=5 POLIMA_SMOOTH_SHAPE=smoothstep
export POLIMA_SUBSTEPS=2 POLIMA_RESET_EASE=smoothstep
polima robot run --fps 15
```

See [docs/groot.md](docs/groot.md) for camera order, calibration and what is not
included.

---

## The SO-101 robot recipe

These are the settings GR00T was run with on the SO-101 arm.

| Setting | What it does |
|---|---|
| `--fps 15` | The arm receives 15 position commands per second. |
| `POLIMA_CHUNK_THRESHOLD=0.5` | Each model answer is a plan of upcoming moves. The next plan is requested when half of the current plan is left. |
| `POLIMA_GROOT_NOISE=zero` | GR00T starts every plan from zero noise instead of random noise, so the same camera view gives the same plan. |
| `POLIMA_SMOOTH_MOVES=5` | When a new plan takes over, the jump from the old plan to the new one is spread over 5 moves instead of 1. |
| `POLIMA_SMOOTH_SHAPE=smoothstep` | That spreading starts and ends gently (an S-curve) instead of using equal steps. |
| `POLIMA_SUBSTEPS=2` | Each command is sent in 2 smaller steps, so the motors move more continuously. |
| `POLIMA_RESET_EASE=smoothstep` | The return to the rest position speeds up and slows down gently instead of starting and stopping abruptly. |

The limit on how far the arm may move per command stays at its default of 12
degrees. Every setting is off unless you set it. Details:
[docs/robot.md](docs/robot.md).

## Performance

Measured on a Modalix DevKit with an SO-101 arm on 2026-10-05, with the
optimised mixed W8A16 GR00T bundles and the recipe above:

| Model | Chip time per plan | Camera to plan |
|---|---|---|
| GR00T N1.6 | 317 ms | 364 ms |
| GR00T N1.7 | 233 ms | 285 ms |

*Chip time per plan* is the time the model takes on the DevKit to produce one
plan of 32 moves. *Camera to plan* also includes reading the camera images and
preparing them for the model.

---

## PoLiMa Studio and Workbench

**PoLiMa Studio** runs on the DevKit (port 8080). From a browser you can:
- choose, start and install bundles (upload from the browser),
- watch the cameras and assign the arm and cameras,
- calibrate the arm,
- start and stop robot runs, with an emergency **HALT NOW** button,
- review and label robot runs, and export the failed ones for retraining,
- benchmark a bundle without the robot.

It is installed but stopped by default. On the DevKit:

```bash
polima studio start     # prints the address to open
polima studio status
polima studio stop
```

**PoLiMa Workbench** runs on your computer (port 8686). It lists your trained
checkpoints, your compiled bundles and your DevKits, with each bundle's
readiness and measured results:

```bash
polima workbench --models ~/checkpoints --bundles outputs/bundles --open
```

Add your DevKits on its **Boards** page: an address such as
`sima@192.168.1.20` (on your network, or cabled straight to the computer), an
optional ssh key and a name. The Workbench checks each one over ssh and through
its Studio, and keeps the list in `~/.config/polima/devkits.json`. A saved name
also works in every other command, for example `polima deploy --board <name>`.

> **Studio and Workbench have no login and no encryption (no TLS).** Use them
> only on an isolated robot network. Workbench listens on every network
> interface by default; add `--host 127.0.0.1` to keep it to your own computer.

More: [docs/polima-studio.md](docs/polima-studio.md).

---

## Known limits

- **Compiling your own GR00T checkpoint is not part of this release.** Use a
  prebuilt GR00T bundle. It comes in the next version.
- **SmolVLA** is not part of this release. It comes in a later version.
- Studio's robot-run review lists runs made with ACT bundles. GR00T
  runs are not recorded there.
- Studio and Workbench have no login or TLS (see above).

## Documentation

- [docs/deploy.md](docs/deploy.md): deploying bundles and managing the DevKit
- [docs/robot.md](docs/robot.md): running the SO-101, the recipe, troubleshooting
- [docs/groot.md](docs/groot.md): running prebuilt GR00T bundles
- [docs/polima-studio.md](docs/polima-studio.md): PoLiMa Studio
- [CHANGELOG.md](CHANGELOG.md): what changed

## Licence

PoLiMa is licensed under the Apache License 2.0 (see [LICENSE](LICENSE)). Model
weights and third-party software keep their own licences; see
[THIRD_PARTY.md](THIRD_PARTY.md). In particular, GR00T bundles are covered by
NVIDIA's licence for the GR00T checkpoints.
