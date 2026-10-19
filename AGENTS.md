# Notes for coding agents

## Skills

| Task | Skill |
| --- | --- |
| Assess, add, port, or optimize a VLA or robot policy (new architecture, checkpoint format, embodiment, or camera layout) | [`skills/sima-add-polima-policy`](skills/sima-add-polima-policy/SKILL.md) |
| Compile a checkpoint of an architecture PoLiMa already supports | No code changes: follow [docs/your-own-model.md](docs/your-own-model.md) |

The SiMa skills cover their own areas: `sima-model-surgery` (unsupported
operators), `sima-model-quantize-compile` (one graph's quantization and
compilation), `sima-environment-setup` (SDK and DevKit setup), and `sima-neat`
(Neat applications on the DevKit).

## Checks

```bash
make test   # .venv/bin/pytest -q tests/unit
make lint   # .venv/bin/ruff check src tests
```

## Rules

- `src/polima/policies/base.py` is stdlib only: specs cross the training,
  compiler, and DevKit interpreters. Name torch code by dotted path.
- Do not commit checkpoints, datasets, ONNX files, ELFs, or bundles.
- The build host and DevKits are shared: keep compile threads within the
  host's cores, and take `/tmp/polima-bench.lock` around on-device
  measurements.
- Changes to robot motion (return to rest, per-command clamp, control rate)
  need a run on the arm before merge.
