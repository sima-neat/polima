---
name: sima-add-polima-policy
description: Assess, add, or optimize support for a VLA or robot policy (a new architecture, a new checkpoint source, a new embodiment or camera layout) in sima-neat/polima so it compiles, runs, and drives a robot on a Modalix DevKit. Use when asked to add, port, bring up, or speed up a policy such as ACT, GR00T, SmolVLA, or Pi0.5 in PoLiMa. Do not use for compiling a checkpoint of an architecture PoLiMa already supports (follow docs/your-own-model.md), for generic ONNX model compilation, or for LLiMa language-model work.
---

# Assess, Add, or Optimize a PoLiMa Policy

Decide the route first, then make the smallest change that fits. Every phase
leaves evidence (a report, a test, or a measurement) before the next starts.

## Classify

1. Read `README.md`, `docs/your-own-model.md`, `src/polima/policies/base.py`,
   and the closest existing policy under `src/polima/policies/` (ACT for a
   small single-pass policy; GR00T for a VLM backbone with a denoising head).
   Policies on open branches are references too: SmolVLA (PR #25,
   `feature/smolvla-optimize`) and Pi0.5 (PR #21, `feature/pi05-llima`).
2. Record the model ID and revision, the checkpoint source (LeRobot, openpi,
   Isaac-GR00T, ...), its `config.json` type, cameras, state/action
   dimensions, chunk length, and denoising steps.
3. Read `references/feasibility.md` and fill in its checklist: provenance
   (weights and processor files present), memory, ops, estimated latency, and
   the closest analogue.
4. For a supported architecture, run the analogue's own checks on the
   checkpoint: its `checkpoint_validator` and `profile_entry`
   (`specialize_spec`), plus the limits its loader enforces (ACT:
   `policies/act/graphs.py` `load_policy`/`validate_checkpoint` and
   `policies/act/__init__.py` `_checkpoint_profile`: both cameras 480x640
   named `overhead`/`wrist`, `chunk_size == n_action_steps == 100`, 512/4/1
   layers, no temporal ensembling). Their verdict decides the route.
5. Choose the route:

| Route | When | Work | Reference |
| --- | --- | --- | --- |
| Compile only | The analogue's validator and profile accept the checkpoint (a new fine-tune or size) | No source changes; `polima compile` | `docs/your-own-model.md` |
| Profile | Supported architecture, new embodiment the spec already parameterizes: joints, dimensions, task text | `profile_entry` / `specialize_spec`, wire and robot | `references/wire-and-robot.md` |
| Extend policy | Supported architecture, but a geometry the policy hard-codes: camera resolution or names baked into the checkpoint, chunk or `n_action_steps`, camera count | Generalize the analogue's graphs, plan, and validator; keep its existing bundles byte-identical | `references/graph-decomposition.md`, `references/wire-and-robot.md` |
| New policy | New architecture or checkpoint format | New `policies/<name>/`, builder, plan, robot, integration | all references, in phase order |
| Optimize | A supported policy is too slow, or too inaccurate after quantization | Measure, change one thing, re-check | `references/optimization.md` |

A base or pretrained checkpoint (no statistics for your robot, padded state
and action, other cameras) can be brought up and checked against PyTorch on a
dataset with matching cameras, but cannot drive the arm until it is fine-tuned:
say so, and ask for the checkpoint and dataset to use.

Camera names live in the checkpoint (its `input_features` and the normalizer's
statistics keys), not only in the dataset, so renaming dataset cameras does not
fix a trained checkpoint. Map them in the builder (Pi0.5's `--camera-map`
pattern) rather than editing checkpoint files or special-casing one checkpoint.
Then read `references/validation-matrix.md` for the evidence each phase needs.

## Code on open branches

Some references cite code that is on open PR branches rather than `develop`:
Pi0.5 (PR #21, `feature/pi05-llima`: the LLiMa backend, `run_device_chain`
in `OPCODES`, the device-resident denoising loop, multi-fixture checks, the
camera map) and SmolVLA (PR #25, `feature/smolvla-optimize`). Check whether
they have merged; if not, read them with
`git show origin/<branch>:<path>` and port what you need rather than
reinventing it.

## Implement (new policy), in phase order

1. **Contract**: `DatasetContract`, `WireSpec`, `RobotSpec`, `profile_entry`;
   reject unsupported checkpoints at load (a `validate_checkpoint`).
2. **Graphs**: split the model as in `references/graph-decomposition.md`;
   write a static torch module per graph that matches the reference
   implementation numerically before any export.
3. **Export and verify**: ONNX opset 17 with `dynamo=False`; calibration from
   real traced activations; a fixture whose expected actions come from the
   *original* PyTorch policy; an onnxruntime-vs-PyTorch report.
4. **Compile**: per-graph precision and layout; a staged builder
   (`compile/<name>.py`) for multi-hour builds, with `--resume`.
5. **Runtime plan**: build `RuntimePlan` from existing opcodes as in
   `references/runtime-plan-and-opcodes.md`; a new opcode needs the native
   kernel, the Python reference interpreter, and tests.
6. **Wire and robot**: request tensors, normalization side, motion recipe,
   rest pose; `references/wire-and-robot.md`.
7. **Bundle, deploy, check**: the bundle's fixtures on a DevKit against
   PyTorch; `polima_cli --interactive` `bench` and `stages`.
8. **Optimize**: `references/optimization.md`; each change measured, fixtures
   still passing.
9. **Integrate and document**: registry, CLI routing, Workbench, Studio,
   `polima doctor`, docs, README tables.

Run `scripts/scaffold_policy.py <name>` to create the package, plan, and test
skeleton from the ACT template instead of copying files by hand.

Use the SiMa skills for their areas instead of repeating them here:
`sima-model-surgery` (unsupported operators, graph rewrites),
`sima-model-quantize-compile` (one graph's quantization and compilation),
`sima-environment-setup` (SDK, DevKit pairing), and `sima-neat` (Neat
applications on the DevKit).

## Boundaries

- Never substitute another checkpoint, revision, dataset, precision, camera
  order, or denoising-step count. Fewer steps or fewer cameras change the
  workload: ask before trying them, and report them separately.
- Judge accuracy against the original PyTorch policy on several dataset frames
  with their own noise, never one fixture (see `references/validation-matrix.md`).
- Do not weaken a pass mark to make a build pass; find the cause.
- Do not commit checkpoints, datasets, ONNX files, ELFs, or bundles.
- The DevKit and the build host are shared: keep compile threads within the
  host's cores, take `/tmp/polima-bench.lock` for on-device measurements, and
  ask before stopping services you did not start.
- Robot motion is a safety path: changes to the return to rest, the per-command
  clamp, or the control rate need a run on the arm before merge.

## Finish

Report: the route and analogue; model, revision, and dataset; changed surfaces;
per-graph precision (weights and activations); accuracy against PyTorch (mean
and worst over the evaluated frames, and the pass mark); latency (median and
breakdown by stage, conditions); evidence not gathered (for example, no arm
run); known pitfalls hit and how each was resolved. Fold every place a person
had to help into `references/pitfalls.md` or the reference that should have
answered it.
