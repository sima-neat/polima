# Validation matrix

Keep the workload fixed across comparisons: checkpoint revision, dataset
frames, prompt/task, camera order, preprocessing, precision, action horizon,
denoising steps, and noise.

| Stage | Required evidence | What it establishes |
|---|---|---|
| Contract | Invalid checkpoints fail early; accepted checkpoint specializes to exact shapes | Supported checkpoint boundary |
| Static modules | Each module matches captured upstream intermediates on multiple samples | Graph rewrite correctness |
| ONNX | Checker passes; ONNX Runtime matches each static module | Export correctness |
| Compiled graphs | Per-graph and chained outputs meet a fixed fidelity gate | Compiler/precision fidelity |
| Bundle fixtures | Several wire-level requests match original upstream actions; mean and worst reported | End-to-end model fidelity |
| Packaging | Manifest/integrity checks; all client, constant, fixture, rest, and calibration inputs present | Deployable artifact completeness |
| Fresh install | Install and discovery in a clean user environment; no source-tree assumptions | Customer setup path |
| DevKit no-motion | Bundle loads repeatedly; prompt/cameras/state produce finite correctly shaped plans | Runtime and client compatibility |
| Supervised arm | Representative starts, success/failure counts, smoothness, stop, recovery, return to rest | Physical readiness |

## Accuracy reporting

Use several dataset frames spread across episodes. Report sample count, mean,
median, worst sample, pass threshold, and any non-finite result. For stochastic
policies, retain each sample's noise so upstream and compiled paths receive the
same input; do not reuse one noise tensor as the entire accuracy set.

Per-graph checks localize errors. The final acceptance check must compare the
complete compiled output with the original upstream policy after matching
preprocessing and postprocessing.

## Performance reporting

After warm-up, retain individual samples and report at least median, p95, and
p99 when sample count supports them. State the device/runtime revision, bundle
revision, graph precision, cameras, action horizon, denoising steps, and whether
the measurement is server-only or camera-to-plan. Record cold load separately
from warm inference.

## Test scope

Run focused tests while developing, then the repository's required lint and
unit suite. Run native kernel/plan tests when runtime code changes. Validate a
fresh checkout or install for discovery and packaging changes.

Hardware evidence must name the exact bundle and deployed source revision.
Passing local tests is not a substitute for a DevKit check, and a DevKit
fixture check is not a substitute for an arm test.

## Capability language

Use these labels precisely:

- **implemented:** code exists;
- **locally verified:** unit/numerical checks passed;
- **DevKit verified:** compiled bundle loaded and fixtures passed on the named
  runtime;
- **arm verified:** the named physical trials passed.

List unavailable evidence and remaining limits directly.
