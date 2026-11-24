---
name: sima-add-polima-policy
description: Assess, add, or optimize support for a VLA or robot policy on PoLiMa's SO-101 integration. Use when support may require a new architecture, checkpoint format, supported camera layout, runtime plan, wire contract, or SO-101 model integration. Do not use for a new fine-tune of an architecture PoLiMa already supports; follow docs/your-own-model.md instead.
---

# Assess, Add, or Optimize a PoLiMa Policy

Inspect the repository and the deployed integration before writing code. Extend
the closest working policy and the shared robot path; do not create a parallel
motor client.

## Classify

1. Read `AGENTS.md`, `docs/your-own-model.md`,
   `src/polima/policies/base.py`, `src/polima/policies/registry.py`, and the
   closest policy under `src/polima/policies/`.
2. Pin the checkpoint ID and revision, upstream framework and version, dataset,
   robot profile, camera keys and shapes, state/action dimensions, action
   horizon, executed actions, denoising steps, and task interface.
3. Complete `references/feasibility.md`. Verify the checkpoint with the
   analogue's validator and profile code before assuming source changes are
   required.
4. Confirm that the target is the supported SO-101 contract. For another robot
   embodiment, assess model compilation separately and report that a new
   launcher and control integration require their own design and validation;
   do not adapt the SO-101 launcher or claim physical readiness.
5. Choose the smallest route:

| Route | Use when | Change |
|---|---|---|
| Compile | The existing validator and profile accept the checkpoint | No source changes; follow `docs/your-own-model.md` |
| Profile | The architecture is supported and the current spec can express the SO-101 camera or model geometry | Extend `profile_entry` and its tests |
| Extend | The architecture is supported but a hard-coded model boundary must change | Generalize that boundary while preserving existing bundles |
| New policy | The architecture or checkpoint contract is new | Add a policy spec, export, runtime, bundle, and robot integration |
| Optimize | A supported policy misses a measured latency, memory, or fidelity target | Change one measured bottleneck at a time |

ACT, GR00T, and Pi0.5 on `develop` are the maintained examples. Do not make a
new feature depend on an unmerged branch. Read a branch only when the user asks
to build on that exact work, and re-check it against current `develop`.

## Implement

For a profile, extension, or new policy, work in this order:

1. **Contract:** declare `DatasetContract`, `CompilePlan`, `WireSpec`,
   `RobotSpec`, `checkpoint_validator`, and `profile_entry` where needed.
2. **Reference:** load the pinned upstream policy in its own interpreter and
   preserve it as the numerical oracle.
3. **Graphs:** choose static graph boundaries and verify each module against
   the upstream policy before export. Read `references/graph-decomposition.md`.
4. **Export and calibration:** export static ONNX, capture real intermediate
   activations, and write fixtures whose expected actions come from upstream.
5. **Compile and runtime:** declare precision per graph and express execution
   with the existing runtime plan and device-chain facilities. Read
   `references/runtime-plan-and-opcodes.md`.
6. **Bundle and robot:** reuse `CompiledPolicyAdapter`, the SO-101 launcher,
   calibration, camera discovery, rest pose, recovery, and motion controls.
   Add only policy-specific preprocessing and action decoding. Read
   `references/wire-and-robot.md`.
7. **Product surfaces:** update registry and compile routing, bundle packaging,
   Workbench prerequisites, Studio behavior, `polima doctor`, and customer
   documentation where the new policy changes them.
8. **Validation:** follow `references/validation-matrix.md`. Run local checks
   before requesting DevKit or arm time.
9. **Optimization:** measure a stable baseline, keep the workload fixed, and
   retain only changes that pass the same fidelity gate. Read
   `references/optimization.md`.

Use `sima-model-surgery` for unsupported operators and graph rewrites, and
`sima-model-quantize-compile` for ModelSDK quantization and compilation of an
individual ONNX graph.

## Boundaries

- Never substitute a checkpoint, revision, dataset, precision, camera order,
  prompt, action horizon, or denoising-step count without explicit agreement.
- Do not edit a checkpoint to make validation pass. Normalize supported naming
  differences at a declared profile or camera-map boundary.
- Do not commit checkpoints, datasets, ONNX files, ELFs, bundles, credentials,
  calibrations, or machine-specific paths.
- Keep generated artifacts in the current user's workspace. Confirm host bind
  mounts before using a container path.
- Preserve existing bundles and their numerical outputs when generalizing a
  policy.
- Robot motion, return-to-rest behavior, command rate, smoothing, and safety
  clamps require an arm test. Compilation and fixture checks do not establish
  robot readiness.

## Finish

Report the chosen route and analogue; pinned model, dataset, and robot
contract; changed surfaces; graph precisions; fidelity over multiple samples;
latency conditions and percentiles; local, DevKit, and arm validation as
separate results; and every check that remains unavailable.
