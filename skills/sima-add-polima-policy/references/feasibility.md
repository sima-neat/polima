# Feasibility and route selection

Answer the affected questions with file paths, commands, or recorded outputs.
Mark an item not applicable rather than silently skipping it.

## Provenance

- Checkpoint repository and immutable revision.
- Upstream framework, source revision, Python version, and the interpreter that
  successfully loads the checkpoint.
- Weight, tokenizer, processor, preprocessing, normalization, and statistics
  files required outside the main weight file.
- Dataset repository/revision or local source, LeRobot format version, camera
  streams, task metadata, and licence constraints.
- Whether the checkpoint is base or robot-fine-tuned. A base policy without
  the robot's statistics and action contract may be compiled and compared with
  upstream, but it is not ready to drive the arm.

## Closest maintained analogue

| Policy | Useful boundary to reuse | Source |
|---|---|---|
| ACT | Single-pass policy, profile specialization, `uint8` camera wire, device chain | `src/polima/policies/act/` |
| GR00T | Vision/language conditioning, task catalog, multi-graph denoising policy | `src/polima/policies/groot/` |
| Pi0.5 | Free-text client processing, LLiMa backbone, device-resident denoising | `src/polima/policies/pi05/` |

Compare architecture, checkpoint layout, camera count/order/shape, state and
action representation, normalization, relative versus absolute actions,
language interface, chunk length, and denoising schedule. Reuse a boundary
only when those contracts match.

## Checkpoint acceptance

Run the closest policy's checkpoint validator and profile specialization. A new
fine-tune or size is compile-only when those accept it. A naming, shape, or
feature mismatch is evidence for a profile or extension; it is not permission
to discard fields or edit the checkpoint.

Confirm the upstream policy can load and run one real sample before exporting.
Keep its preprocessing and postprocessing in the comparison path.

## Compiler fit

- Export representative blocks with the real static shapes and inventory their
  ONNX operators.
- Check dynamic indexing, unsupported reductions, high-rank boundaries, large
  tensors, and stateful or temporal features.
- Use `sima-model-surgery` to assess ModelSDK compatibility.
- If LLiMa already supports the language backbone, compare an LLiMa prefill
  against an ONNX decomposition before choosing the backend.
- Estimate device weights, intermediate buffers, host assets, bundle size,
  build disk, DevKit storage, RAM, and CMA. Measure rather than relying on an
  earlier policy's figures.

## Robot and dataset fit

Verify:

- the exact robot embodiment, joint order, state/action units and dimensions;
- checkpoint camera keys and their mapping to dataset and robot roles;
- capture and model resolution, color order, resize/crop/pad, and normalization;
- trained FPS, control FPS, action horizon, executed actions, and whether a
  plan outlasts model latency with margin;
- fixed task catalog versus free-text tokenization;
- required rest pose and calibration source.

Camera names stored in a checkpoint are part of its learned input contract.
Use an explicit camera map when supported; do not silently rename checkpoint
files or rely on `/dev/videoN` enumeration.

## Decision record

Record the route, analogue, expected graph split and precision, resource
estimate, required environments, validation plan, and blockers. Stop before a
long build when the checkpoint cannot load, the robot contract is absent, or
the device resource estimate is infeasible.
