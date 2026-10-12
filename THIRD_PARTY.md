# Third-party components and model weights

PoLiMa's own code is licensed under Apache-2.0 (see `LICENSE` and `NOTICE`).
It works with the components below. None of their source code or weights is
copied into this repository unless the table says so. Each keeps its own
licence; read it before you redistribute anything built with it.

## Model weights

| Model | Where it comes from | Licence |
|---|---|---|
| NVIDIA Isaac GR00T N1.6 (`nvidia/GR00T-N1.6-3B`) and checkpoints fine-tuned from it | Hugging Face, NVIDIA | NVIDIA licence shipped as `LICENSE` with the checkpoint. Its section 3.3 limits use to non-commercial research or evaluation; the model card links the NVIDIA One-Way Non-Commercial License. |
| NVIDIA Isaac GR00T N1.7 (`nvidia/GR00T-N1.7-3B`) and checkpoints fine-tuned from it | Hugging Face, NVIDIA | The model card names the NVIDIA Open Model License Agreement; the `LICENSE` file shipped with the checkpoint is the same non-commercial NVIDIA licence as N1.6. Check which terms apply to your use. |
| Eagle 2.5 vision-language backbone (inside GR00T N1.6) | NVIDIA | Covered by the GR00T N1.6 checkpoint's licence. |
| Cosmos-Reason2 / Qwen3-VL vision-language backbone (inside GR00T N1.7) | NVIDIA, built on Qwen3-VL (Alibaba Qwen team) | Covered by the GR00T N1.7 checkpoint's licence and the upstream Qwen licence. |
| ACT and SmolVLA checkpoints | Trained by you with LeRobot | Whatever licence applies to your training data and base weights (SmolVLA starts from `lerobot/smolvla_base`, Apache-2.0). |

A prebuilt bundle (`.../models/<bundle-id>/`) contains compiled weights. It is
a derivative of the checkpoint it was built from and is covered by that
checkpoint's licence, not by PoLiMa's.

## Software PoLiMa uses but does not include

| Component | Used for | Licence |
|---|---|---|
| NVIDIA Isaac-GR00T (`gr00t` Python package) | Exporting GR00T checkpoints on the host (not needed to run prebuilt bundles). PoLiMa's GR00T N1.7 port follows the `n1.7-release` tag. | N1.7 code: Apache-2.0. The N1.6 source tree carries an NVIDIA licence; check it before use. |
| Hugging Face LeRobot 0.4.4 (`lerobot`) | Training (outside PoLiMa), exporting ACT/SmolVLA, and the SO-101 robot client on the board | Apache-2.0, with notices for code LeRobot derives from Diffusion Policy, FOWM and simxarm |
| Hugging Face Transformers, PyTorch, ONNX, ONNX Runtime, onnxsim, NumPy, OpenCV, Flask, Waitress, Pillow, PyArrow, pandas, PyAV, safetensors, draccus, PyYAML | Installed by pip as dependencies | Their own permissive licences (Apache-2.0, BSD, MIT, HPND) |
| `lerobot_robot_mycobot280` and `pymycobot` | MyCobot 280 support only | Their own licences |
| SiMa.ai ModelSDK (model compiler, `afe`) | Compiling models on the x86 host | Proprietary; licensed separately by SiMa.ai. Install it yourself. |
| SiMa.ai Modalix runtime (`sima_lmm`, MLA dispatcher, `fix_devkit_runtime.sh`) | Running bundles on the Modalix DevKit; `native/` links against it | Proprietary; ships with the DevKit image and SDK. |

## Code in this repository that comes from elsewhere

| Path | Origin | Licence |
|---|---|---|
| `src/polima/studio/static/assets/index-*.js` | Built Studio UI. Bundles Preact and `@preact/signals` | MIT (Preact authors) |
| `studio-ui/` (development only) | Built with Vite and `@preact/preset-vite` | MIT |
| `lerobot_sima/` | SiMa.ai install script and hardware checks for LeRobot 0.4.4; they call LeRobot, they do not copy it | Apache-2.0 (this repository) |
| `patches/README.md` | Describes three local edits SiMa.ai made to LeRobot's async-inference stack and SO follower; the edited LeRobot clones are not in this repository | LeRobot's licence (Apache-2.0) applies to those edits |
| GR00T camera preprocessing in `native/` and `src/polima/policies/groot*/` | Re-implements the image processing of the GR00T processor so the board gives the same numbers; no NVIDIA source is copied | Apache-2.0 (this repository) |
