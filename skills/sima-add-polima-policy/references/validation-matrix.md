# Validation Matrix

Select every row the route touches. Each row names the evidence that closes it.

| Phase | Evidence | Command or test |
| --- | --- | --- |
| Feasibility | The `feasibility.md` checklist answered, route chosen | Written report |
| Route (compile only) | The analogue's validator and profile accept the checkpoint, or the exact rejection reasons | Run `validate_checkpoint` / `specialize_spec` on the checkpoint; report each refused limit and the route it implies |
| Contract | Spec validates; the checkpoint validator refuses a mismatched checkpoint | `polima doctor`; a unit test that `get_policy("<name>")` validates and a bad checkpoint is refused |
| Graph modules | Each static module matches the reference implementation in FP32 | Unit test per module (e.g. Pi0.5 `test_nchw_expert_matches_the_expert_step`), max abs diff ~1e-5 |
| Export | ONNX chain in onnxruntime vs the original PyTorch policy, with thresholds | `onnx_verification_report.json` / `verification.json`; the build stops on FAIL |
| Compile | Every graph compiles; norms fused; compiled input order recorded | `compile_results.json` / `compile_state.json`; `.sima` op list has `RMSNormOp`, no `RsqrtOp`; MPK input order |
| Single graph on device | The heaviest graph alone on the DevKit vs its ONNX on the same real inputs | One-graph bundle with `polima_cli --bundle <dir> --input-dir ...`; relative error and cosine |
| Runtime plan | Plan validates; replay matches the PyTorch fixture in the Python reference | `tests/unit/test_plan.py`-style StubPlan test; `PolicySpec.validate` |
| Bundle check | The bundle's fixtures on the DevKit vs PyTorch within the precision's pass mark | `polima compile ... --check-devkit <devkit>` or `polima run --bundle <id> --fixture --repeat 20` |
| Accuracy | Mean and worst over 8-16 dataset frames with their own noise | A multi-frame run (Pi0.5's 8-fixture check); report mean, median, worst |
| Latency | Median of >= 20 runs after warm-up, breakdown by stage, conditions | `polima_cli --interactive` `bench 20 3`, `stages`; `POLIMA_PROFILE_PHASES=1` |
| Deploy | Bundle deploys, server answers over TCP | `polima deploy --bundle <id> --start`; `polima run --bundle <id> --fixture` |
| Robot | Client starts, reaches rest, executes plans, returns to rest on Ctrl-C | `polima robot run` on the arm; run log |
| Integration | CLI routing, Workbench, doctor, docs | Unit tests for `compile_command`, prerequisites, CLI routing; `ruff`; `pytest tests/unit` |

## Pass marks

The bundle carries its own mark in `bundle.json` `smoke`; the check uses it.

| Build | Cosine (min) | Mean abs. error (max) | Source |
| --- | --- | --- | --- |
| BF16 (ACT and others) | 0.999 | 0.01 | `SmokeSpec` default; GR00T `ENVELOPE` |
| INT8 weights ("mixed", `w8a16`) | 0.995 | 0.03 | GR00T `ENVELOPE`; Pi0.5 |
| GR00T N1.7 export | 0.98 | 0.015 (normalized) | `policies/groot/n17.py` |
| `w8a8` | reported, not gated | | `compile/groot.py` |

Errors are on normalized actions. Apply a mark to the mean over the fixtures,
and report the worst fixture beside it: one fixture can sit far from the
mean (Pi0.5's SO-101 bundle: 0.047 on frame 0, 0.017 over eight).

## Accuracy procedure

1. Pick 8-16 frames spread over the dataset's episodes; give each its own noise
   (for flow-matching heads).
2. Run the original PyTorch policy in FP32 on each: these are the expected
   actions.
3. Run the bundle on the DevKit on the same requests.
4. Report cosine and mean absolute error per frame, then mean, median, worst.
5. When accuracy misses, localize before fixing: run each part alone on real
   inputs (the prefix's K/V with the reference action head, the action head
   alone), compare against CPU simulations of the precision (BF16, INT8
   weights, per-token INT8 activations). A part that is worse on the device
   than its simulation has a compile problem (an unfused norm), not a
   precision problem.

## Report

State the model and revision, the dataset and frames, the per-graph precision,
each measured number with its conditions (DevKit, runtime build, other
services running), and every check not run.
