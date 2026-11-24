# Notes for coding agents

## Skills

| Task | Skill |
|---|---|
| Assess, add, port, or optimize an SO-101 VLA or robot policy | [`skills/sima-add-polima-policy`](skills/sima-add-polima-policy/SKILL.md) |
| Compile a checkpoint of an architecture PoLiMa already supports | Follow [`docs/your-own-model.md`](docs/your-own-model.md); no source changes are normally required |

Use `sima-model-surgery` for unsupported operators and graph rewrites, and
`sima-model-quantize-compile` for ModelSDK quantization and compilation of an
individual ONNX graph.

## Checks

```bash
make test
make lint
```

## Rules

- Inspect current policy, bundle, deployment, robot, and documentation paths
  before proposing replacements. Do not guess interpreter paths, device roles,
  package contents, or deployed versions.
- Extend the shared bundle launcher, transport, `CompiledPolicyAdapter`, and
  SO-101 control path. Do not create a parallel motor client for a new policy.
- Treat export, ELF compilation, bundle packaging, deployment, fixture checks,
  and robot execution as separate validation stages.
- Keep `src/polima/policies/base.py` stdlib-only. Specs cross the training,
  compiler, and DevKit interpreters; name framework code by dotted path.
- Do not commit checkpoints, datasets, ONNX files, ELFs, bundles, credentials,
  calibrations, or machine-specific paths.
- Keep generated artifacts in the current user's workspace and confirm host
  bind mounts before using a container path.
- Preserve working deployments, calibrations, datasets, and checkpoints.
- The build host and DevKits are shared. Bound compile parallelism and take
  `/tmp/polima-bench.lock` around on-device measurements.
- State local, DevKit, and arm validation separately. Compilation or local
  tests do not establish robot readiness.
- Changes to robot motion, return to rest, command rate, smoothing, or safety
  clamps require a supervised arm test before merge.
