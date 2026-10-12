# Run your own model

From demonstrations to your model driving the arm:

```
record ──▶ train ──▶ polima compile ──▶ check ──▶ polima deploy ──▶ polima robot run
(LeRobot)  (LeRobot / GR00T)  (your PC)  (DevKit)   (your PC)          (DevKit)
```

This guide uses an SO-101 arm with two cameras, an overhead camera and a wrist
camera. First install PoLiMa on your computer and on the DevKit
([install.md](install.md)).

## 1. Record

Record demonstrations with LeRobot's `lerobot-record`, moving a leader arm
while the follower copies it. [examples/so101/setup](../examples/so101/setup/README.md)
has tools to find and test the two arms and to start teleoperation on the
DevKit.

Name the two cameras `overhead` and `wrist`, so the dataset has
`observation.images.overhead` and `observation.images.wrist`. Record the arm
at the speed you want it to run, usually 30 frames per second.

Keep the cameras where they were during recording. The model learns the
views; if a camera moves, the arm misses.

## 2. Train

**ACT:** train with LeRobot's `lerobot-train --policy.type=act` on your
dataset. PoLiMa compiles the `pretrained_model` folder of a checkpoint.

**GR00T N1.6 or N1.7:** fine-tune with NVIDIA's
[Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) code. GR00T needs a
LeRobot dataset in format v2.0 or v2.1 with a `meta/modality.json`; see
[groot.md](groot.md#preparing-your-dataset) for the file and the settings our
models were trained with.

## 3. Compile

On the x86_64 Linux computer with the ModelSDK ([install.md](install.md)).

### ACT

```bash
polima compile --policy act \
    --checkpoint /path/to/outputs/train/<run>/checkpoints/100000/pretrained_model \
    --build-dir ~/polima-builds/my-task -j 3
```

PoLiMa finds the training dataset from the checkpoint; add
`--dataset-root /path/to/dataset` if it has moved. `-j 3` compiles three parts
at once, each needing about 1.6 GB of memory. The bundle is written to
`outputs/bundles/<bundle-id>/` and the command prints its id.

### GR00T

Use a separate export environment for each GR00T version. The ModelSDK runs
in its own environment; do not install GR00T into the compiler environment.

| Requirement | GR00T N1.6 | GR00T N1.7 |
|---|---|---|
| Isaac-GR00T commit | `9b37aa1ce69c73c6d165233fa88128283bba4508` | `51d4c89f72fda44cbf77285c6a8114b52676b8a1` |
| Export device | NVIDIA GPU with the matching FlashAttention dependencies | CPU |
| Python | Follow the pinned source requirements | 3.12 |
| Backbone assets | Assets referenced by the checkpoint and processor | `nvidia/Cosmos-Reason2-2B`, or the local backbone directory referenced by the checkpoint |

For N1.7, clone the pinned source with its submodules:

```bash
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git ~/isaac-groot-n1.7
git -C ~/isaac-groot-n1.7 checkout 51d4c89f72fda44cbf77285c6a8114b52676b8a1
git -C ~/isaac-groot-n1.7 submodule update --init --recursive
```

Follow the installation instructions in that checkout. Its documented x86_64
setup uses `uv sync --python 3.12` and includes GPU dependencies, even though
PoLiMa's N1.7 export runs on CPU. Install a supported FFmpeg runtime for
TorchCodec. Export has been exercised with PyTorch 2.9.0, Transformers 4.57.3,
NumPy 1.26.4, ONNX 1.20.0, ONNX Runtime 1.23.2 and TorchCodec 0.8.0. A separate
minimal CPU-only environment recipe has not been validated.

Point PoLiMa to the resulting interpreter and source:

```bash
export POLIMA_GROOT17_PYTHON=~/isaac-groot-n1.7/.venv/bin/python
export POLIMA_GROOT17_SOURCE=~/isaac-groot-n1.7
"$POLIMA_GROOT17_PYTHON" -m pip install onnx==1.20.0 onnxruntime==1.23.2
```

If the environment was created with uv and has no pip, use
`uv pip install --python "$POLIMA_GROOT17_PYTHON" onnx==1.20.0 onnxruntime==1.23.2`.
For N1.6, use its pinned commit and set `POLIMA_GROOT16_PYTHON` and
`POLIMA_GROOT16_SOURCE` to the matching environment and checkout. Save these
settings once using the persistent configuration below.

#### Download the backbone before export

PoLiMa export workers default to `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`. A checkpoint may still load its backbone weights,
tokenizer and image processor separately. Download these before compiling.
For N1.7, request access to the Cosmos model on Hugging Face and authenticate
using the `hf` executable from the export environment:

```bash
export HF_HOME=~/polima-cache/huggingface
"$(dirname "$POLIMA_GROOT17_PYTHON")/hf" auth login
"$(dirname "$POLIMA_GROOT17_PYTHON")/hf" download nvidia/Cosmos-Reason2-2B
```

Use the same `HF_HOME` when starting Workbench or running `polima compile`.
If `HF_HUB_CACHE` is set, keep that value consistent too. Restart Workbench
after changing these variables. Download the complete model, including its
weights, tokenizer and `preprocessor_config.json`.

Inspect `model_name` in both the checkpoint's `config.json` and
`processor_config.json` (`processor_kwargs.model_name`). A Hub model ID uses
the cache above. An absolute path refers to a separate local asset directory;
uploading a checkpoint does not upload assets outside its folder.

For an N1.7 checkpoint that references its training computer, select the same
backbone on the compiler host without editing the checkpoint:

```bash
export POLIMA_GROOT17_BACKBONE=nvidia/Cosmos-Reason2-2B
```

The value can also be an existing local directory containing that backbone.
Use the matching model and processor assets; this option does not download
files or select a different model architecture. PoLiMa applies the override
in a temporary checkpoint view and preserves the uploaded files. Set it
before starting Workbench, or restart Workbench after changing it. The
previous download and cache requirements still apply.

Verify cached configuration without downloading anything:

```bash
"$POLIMA_GROOT17_PYTHON" -c 'from huggingface_hub import hf_hub_download; print(hf_hub_download("nvidia/Cosmos-Reason2-2B", "preprocessor_config.json", local_files_only=True))'
```

This confirms the processor file is cached, not that all weights or export
dependencies are ready. The export stage checks model loading and execution.

#### Save compiler setup once

Install the ModelSDK, matching GR00T environment, source checkout and backbone
assets on the **Linux build host**. A laptop that only opens Workbench in its
browser needs none of these compiler dependencies. Uploading a checkpoint
does not install them.

For N1.7, save the downloaded backbone's local directory so a new shell does
not depend on a custom Hugging Face cache variable. After the download above:

```bash
export POLIMA_GROOT17_BACKBONE="$("$POLIMA_GROOT17_PYTHON" -c 'from huggingface_hub import snapshot_download; print(snapshot_download("nvidia/Cosmos-Reason2-2B", local_files_only=True))')"
```

If you already use a complete local backbone directory, keep that value.
Save the environment settings above to your per-user configuration. This
merges the supplied paths with existing settings:

```bash
python3 - <<'PYCONFIG'
import json, os
from pathlib import Path

path = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "polima/config.json"
config = json.loads(path.read_text()) if path.exists() else {}
paths = config.setdefault("paths", {})
for key in ("groot16_python", "groot16_source", "groot17_python", "groot17_source", "groot17_backbone"):
    value = os.environ.get("POLIMA_" + key.upper())
    if value:
        paths[key] = value
if os.environ.get("MODEL_COMPILER_BIN"):
    paths["model_compiler_bin"] = os.environ["MODEL_COMPILER_BIN"]
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(config, indent=2) + "\n")
print(path)
PYCONFIG
```

The default file is `~/.config/polima/config.json`; `XDG_CONFIG_HOME` changes
its parent directory. Environment variables override saved values. N1.6 uses
its own interpreter and source keys. These settings belong to the build host
and do not need to be entered on the laptop or DevKit.

Restart Workbench on the build host after saving settings or changing its
environment:

```bash
polima workbench stop
polima workbench start
```

Select the uploaded checkpoint and dataset, then check prerequisites. Saved
paths identify dependencies; they do not establish that export or compilation
will succeed. The build performs those checks.

Then compile:

```bash
polima compile --policy groot \
    --checkpoint /path/to/checkpoint-50000 \
    --dataset /path/to/your/dataset \
    --output ~/polima-builds/my-groot -j 3
```

PoLiMa reads the GR00T version, cameras, joints and plan length from the
checkpoint and the dataset. It exports the model, prepares the action
generator, compiles every part and packs one bundle under
`~/polima-builds/my-groot/bundles/`. Reference builds took several hours and
about 90 GB of disk; requirements depend on the checkpoint and host. If it stops, rerun the same command with `--resume`; finished parts are
kept.

`--precision mixed` (the default) stores weights as 8-bit integers and
computes in 16-bit floating point: the tested choice. `--precision bf16` is
16-bit floating point. Compare numerical fidelity and measured latency for
your checkpoint before selecting a precision.

## 4. Check

Every bundle carries a test input and the answer the original PyTorch model
gives for it. Deploy the bundle and compare, before the arm moves:

```bash
polima deploy --bundle <bundle-folder-or-id> --start
polima run --bundle <bundle-id> --fixture --repeat 20
```

It reports numerical agreement with PyTorch, inference latency and PASS or
FAIL against the bundle's thresholds. This check does not measure robot task
success. For GR00T, add
`--check-devkit <devkit-address or name>` to `polima compile` to run this check as
the last compile step; the DevKit's model server must be stopped for it
(`polima server stop`).

## 5. Run

On the DevKit, once per arm:

```bash
polima robot calibrate
```

Then:

```bash
polima robot run
```

It asks which serial port is the arm and which cameras are the overhead and
wrist cameras, asks for confirmation, moves the arm to its rest pose and hands
over to your model. Ctrl-C stops the run and returns the arm to rest.

GR00T uses 30 commands per second by default. `polima robot run --slow`
selects its 15-command-per-second motion profile. More options and troubleshooting:
[robot.md](robot.md).

## If the arm does not do the task

| What you see | What to check |
|---|---|
| The check in step 4 fails | Compile again with `--precision bf16`; if that passes, report the mixed result to us. |
| The arm moves the wrong way | The cameras are assigned to the wrong roles, or are placed differently from the recording. |
| The arm misses by the same amount every time | Recalibrate the arm. |
| The arm hesitates or wanders | Record more demonstrations of the situations where it fails, and train again. |
