# Measured optimization

Optimize only after the unoptimized path passes the same end-to-end fixture
set used as the correctness gate.

## Baseline

Pin the source, bundle, runtime, checkpoint, dataset inputs, precision, camera
count, prompt, action horizon, and denoising steps. Record warm-up and measured
sample counts, individual latency samples, stage timings, memory, and fidelity.

Separate model-server time from client preprocessing, camera age, transport,
queue wait, and motor command rate. A faster model does not establish smoother
motion, and a higher command rate does not reduce model latency.

## Optimize one measured bottleneck

Common reusable options already present in PoLiMa include:

- keeping compatible activations in a device chain;
- caching conditioning or other request-invariant inputs;
- moving small deterministic operations to existing runtime opcodes;
- fusing supported normalization patterns;
- choosing graph boundaries that reduce submissions without increasing error;
- quantizing selected weights while retaining required activation precision;
- using LLiMa for a supported language backbone;
- sending raw `uint8` camera frames when server decoding is numerically exact;
- overlapping independent host preparation without sharing mutable buffers.

Change one variable, rerun the complete fidelity set, then benchmark under the
same conditions. Remove variants that are slower, fail the gate, increase
resource use beyond the target, or depend on a checkpoint-specific accident.

## Guardrails

- Do not reduce cameras, image size, action horizon, denoising steps, or model
  precision and report the result as the same workload.
- Do not relax the accuracy threshold after seeing a result.
- Do not increase motor safety limits to hide model or scheduling problems.
- Verify repeated loads and consecutive requests; a single successful request
  can hide resource leaks or stale device state.
- Report cold-load and warm-inference changes separately.
- Keep benchmark-only logging and diagnostics out of normal robot hot paths.

Promote an optimization only when the final report includes before/after
latency, fidelity, memory or disk effects, exact conditions, and the retained
test evidence.
