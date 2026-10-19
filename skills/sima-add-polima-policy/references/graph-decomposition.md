# Graph decomposition, export, and calibration

The upstream policy is the oracle. Preserve its preprocessing, prompt, state,
noise, and postprocessing while deciding where compiled graphs begin and end.

## Choose boundaries

- Prefer a small number of static graphs with explicit tensor contracts.
- Split at stable semantic boundaries such as vision output, language prefix,
  cached conditioning, or action-head steps.
- Keep repeated denoising blocks structurally identical where possible.
- Keep values on the device across graph calls when layouts and precisions are
  compatible; use the existing device-chain implementation.
- Move deterministic, inexpensive operations to existing host/runtime opcodes
  only when their arithmetic is verified against upstream.
- Pad accelerator-facing channels when required, then remove padding in the
  runtime plan. Never expose padding as robot actions.

Study the maintained implementations before choosing a form:

- ACT: `src/polima/policies/act/graphs.py` and `runtime.py`
- GR00T: `src/polima/policies/groot/graphs.py`, `dit.py`, and `runtime.py`
- Pi0.5: `src/polima/policies/pi05/graphs.py`, `profile.py`, and `runtime.py`

## Verify modules before export

For each static module:

1. Load real checkpoint weights.
2. Capture the exact upstream input to the proposed boundary.
3. Compare module outputs in float32 or the upstream compute dtype.
4. Cover more than one dataset frame and every repeated step or mask variant.
5. Reject non-finite values, shape-only agreement, and comparisons against a
   reimplementation of the same module.

## Export

- Use fully static shapes and the repository's current ONNX export pattern.
- Preserve graph names from `GraphSpec` through ONNX, ELF, bundle, and runtime
  plan.
- Run ONNX checker and ONNX Runtime against the same captured tensors.
- Make unsupported checkpoint features fail in `checkpoint_validator` before
  a multi-hour build.
- Keep export in the policy's upstream interpreter; keep ModelSDK compilation
  in its own environment.

## Calibration

Calibration must contain real intermediate inputs after the policy's actual
preprocessing. Sample across episodes and workspace conditions. For iterative
policies, include every denoising step and representative noise rather than
calibrating only the first step.

Record how each calibration tensor was produced. Random calibration is valid
only for a graph whose true input distribution is intentionally random.

## Precision

Declare weight and activation precision per graph. Do not describe a build as
"mixed" without stating which graphs or weights are quantized. Derive the
fidelity gate from measured upstream comparisons over multiple samples, then
keep that gate fixed while optimizing.

Quantization changes must report mean and worst-sample error, cosine where
useful, and non-finite checks. A single fixture or a relaxed threshold is not
evidence that a policy remains usable.

## Fixtures

Fixtures must contain wire-level requests and expected actions produced by the
original upstream policy. Preserve the exact frame, state, prompt, task ID, and
noise needed to reproduce each request. Use several samples spread through the
dataset; iterative policies need independent noise per sample.
