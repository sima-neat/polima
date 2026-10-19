# Pitfalls

Each was hit on a real policy. Read the rows for the phase you are in.

## Numerics and compilation

| Symptom | Cause | Rule | Seen in |
| --- | --- | --- | --- |
| Non-finite outputs | ModelSDK INT8 saturates ACT's vision weights | ACT is BF16 only; a BF16 failure stops the build, never falls back to INT8 | `policies/act/__init__.py` |
| Non-finite outputs from an all-BF16 denoising expert | ModelSDK 2.1.3 code for large BF16 experts | Give the expert INT8 weights | SmolVLA, Pi0.5 |
| ~6% relative error per denoising step | RMSNorm written as `x * rsqrt(...)` does not fuse; `rsqrt` runs as an INT8 lookup | Fusible norm pattern plus `--no-simplify`; check the `.sima` op list | Pi0.5 |
| `Sqrt` lowered to INT8 | Matmul-mean norm form | Use the `ReduceMean` form | GR00T N1.7 |
| Reductions rejected | Mean/variance in LayerNorm | Matmul with a constant `ones/N` vector | GR00T N1.6 |
| Scalar `Cast`/`Expand` rejected | In-graph timestep embedding | Compute on the host, ship as sidecars | GR00T |
| Policy acts at the wrong denoising phase | Sinusoidal formula off (base, `half` vs `half - 1`, 2 pi) | Parameterize by min/max period; test against the reference | GR00T (`sincos_time`) |
| Vision cosine 0.88 | Position ids bucketed in FP32 instead of the tower's BF16 | Index math in the model's native dtype | SmolVLA |
| AFE rejects the ONNX | Dynamo exporter | `dynamo=False`, opset 17 | all |
| ELF hash changes with no math change | Torch exports nodes in evaluation order; the model name is in the ELF | Keep statement order and graph names stable | ACT |
| Attends to the wrong tokens, no error | One mask for every block | Feed each block its mask (GR00T alternates text and image) | GR00T |
| Wrong attention | `-inf` masks | Additive 0 / -1e4 | GR00T |
| Cosine 0.978 instead of 0.9996 | Tessellated HWC16 vision output | Plain HWC; HWC16 only where measured worth it | GR00T |
| Non-finite after decoding | Decoded as HWC16 while the bare ELF returns plain BF16 | Decode what the ELF actually returns | GR00T |
| Deterministic token/channel permutation | Rank-3 promotion as a reshape in NCHW | Transpose before wrapping (`--promote-rank3-hwc`) | build |
| Inputs fed in the wrong order | Compiled MLA input order differs from ONNX | Read the order from the MPK; `--exit-on-stable-elf` can leave no MPK | GR00T, Pi0.5 |
| Wrong buffers bound past 10 bindings | Bindings presented in lexicographic name order | Write bindings in that order | GR00T |
| First queued result wrong | Same ELF twice in one MLA queue submission (sima-lmm 0.4.0) | Split submissions | runtime |
| Persistent ELF loses part of the graph | Compiler split it into several MLA segments | Require one MLA segment | build |
| Large-tensor assertion | One input with ~100k+ channels as its own binding | Declare it as rows with the same bytes | Pi0.5 |
| Later denoising steps drift | Calibrated only at the first step | Calibrate across every step | SmolVLA |
| INT8-weight layers lose accuracy | Inputs quantized to INT8 per token at run time; one outlier channel coarsens the whole token | Fused norms, SmoothQuant, BF16 for sensitive graphs | GR00T N1.7, Pi0.5 |
| Fixture off by one ulp | LeRobot divides by `std + eps` | Same eps on the wire | ACT |
| Clean export, wrong actions | Unsupported checkpoint variant (e.g. temporal ensembling: the runtime is stateless) | Validate the checkpoint at load and refuse it | ACT |
| A LeRobot fine-tune of a supported policy is refused | `n_action_steps` < `chunk_size`, another camera resolution, or other camera names baked into the checkpoint | It is the "extend policy" route, not compile only; never edit the checkpoint to pass | ACT |
| A Hub snapshot has no weights | The cache holds only `config.json` | Check `model.safetensors` and the processor files before anything else | feasibility |

## Accuracy judgement

| Symptom | Cause | Rule |
| --- | --- | --- |
| A build fails (or passes) by luck | One fixture | Judge the mean over 8-16 frames, report the worst |
| Blaming the hardware for INT8 error | Not localized | Compare each part on the device against its CPU simulation on the same inputs |

## Robot

| Symptom | Cause | Rule |
| --- | --- | --- |
| Arm reaches for the wrong place, no error | Cameras swapped (plug order) or role names mismatched | by-id camera hints; model camera order from the processor keys; check roles on a preview |
| Frames dropped | Raw video over USB | MJPG |
| Bundle cannot drive the arm | No `robot_rest_position.json` | Every SO-101 bundle path writes it |
| Plans run out before the next arrives | Chunk too short for the model latency | `actions / fps x threshold > latency` |

## Operations

| Symptom | Cause | Rule |
| --- | --- | --- |
| `MLA_LOAD_FAILED` | CMA pool fragmented (often after a server was killed) | `polima server stop`, then `polima deploy --reset-mla` (or `sudo /usr/bin/fix_devkit_runtime.sh`); never SIGKILL the server |
| Compiles take 4x longer | Build host oversubscribed (several compiles plus CPU work) | Total threads within the cores; one heavy compile at a time |
| Permission error on `/tmp/polima-compile-slot.lock` | Another user owns it on a shared host | Open it read-only for `flock` |
| A background build dies with your shell | Started with plain `nohup` from a session | `setsid nohup ... < /dev/null &` |
| `pkill -f <pattern>` kills your own shell | The pattern matches the command running pkill | Select processes by `/proc` cmdline or PID |
| A local script shadows the standard library | Named like a stdlib module (`bisect.py`) | Never name scripts after stdlib modules |
| Board check uses an old runtime | DevKit's installed `polima_cli` predates new opcodes | Deploy the branch's runtime, or point the check at a dev build (`POLIMA_CHECK_CLI` where the builder supports it) |
