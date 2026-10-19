# Optimize

## Method

1. **Measure first.** `bench` for the median, `stages` for each step,
   `POLIMA_PROFILE_PHASES=1` for host encode vs MLA execute vs decode. Attack
   the largest share; a change to a 2% step cannot matter.
2. **One change at a time**, each with its latency and its accuracy over the
   multi-frame evaluation (`validation-matrix.md`). Keep the fixtures passing.
3. **Know the ceiling before building.** Host work per step shows what a
   device-resident loop can save (Pi0.5, 2 cameras: 16.5 ms MLA + 1.8 ms host
   per step, so the device loop saved 8 ms, not more).
4. **Workload changes are not optimizations.** Fewer denoising steps, fewer
   cameras, or shorter chunks need the user's sign-off and are reported
   separately.

## Techniques

| Technique | What it does | Measured effect | Where |
| --- | --- | --- | --- |
| Device-resident chain | Graph outputs stay in MLA memory and feed the next graph; one upload and download per request | ACT 19.0 -> 15.8 ms, output byte-identical | ACT (PR #22) |
| Whole policy as one device program | Vision, connector, prefix, and all denoising steps; K/V stays on the device | SmolVLA 248 -> 151 ms | SmolVLA (PR #25) |
| Device-resident denoising loop | Euler folded into the step graph; `x` ping-pongs; per-step conditioning cached on the device | Pi0.5 369.5 -> 361.4 ms; accuracy unchanged | Pi0.5 (PR #21) |
| Persistent bindings (`run_bound_chain`) | Host-fed inputs uploaded only when they change | Matches the hand-made original byte for byte | GR00T DiT |
| K/V cache for cross-attention | Project K/V once per request in a prelude, not every step | Relative error < 1e-3 (FP32) | GR00T; SmolVLA and Pi0.5 project in the prefix |
| N layers per graph | Fewer, larger graphs (GR00T DiT 16 -> 8 graphs, linear as 1x1 conv) | Fewer submissions | GR00T |
| GR00T DiT form for transformer heads | NCHW, 1x1 conv linears, einsum attention, host adaRMS | Pi0.5 expert, 3 cameras: 22.2 -> ~20 ms per step, more accurate | Pi0.5 |
| Host precomputation | AdaLN/adaRMS modulation, timestep and tau embeddings, prompt embeddings as sidecars | Removes graph work every step | GR00T, SmolVLA, Pi0.5 |
| INT8 weights on heavy graphs | Vision and language layers with INT8 weights, BF16 elsewhere | Pi0.5 559 -> 361 ms (2 cameras); GR00T "mixed" cosine 0.9987-0.9995 | GR00T, Pi0.5 |
| LLiMa for a supported backbone | LLiMa's chained prefill instead of PoLiMa ONNX graphs | Pi0.5 prefix 2325 -> 702 ms with a passing check | Pi0.5 |
| Fused RMSNorm | The fusible norm pattern and `--no-simplify` | SmolVLA denoise 8.7 -> 7.1 ms per step; Pi0.5 per-step error 0.058 -> 0.018 at no cost | SmolVLA, Pi0.5 |
| SmoothQuant | Moves activation outliers into weights before INT8-weight layers quantize their inputs per token | Halved the per-token error in simulation (Pi0.5); used in GR00T N1.7 (alpha 0.6) | GR00T N1.7 |
| Merged small graphs | Fold a projection or norm into a neighbour; a pixel-shuffle + projection as one strided conv | SmolVLA connector 3.5 -> 1.7 ms per camera, 151 -> 146 ms | SmolVLA |
| Queued MLA submissions | `add_to_queue` / `run_queue` instead of a round trip per graph | Always on in the runtime | all |
| Host-side work | Direct writes into MLA mappings, fused finite check, worker-thread overlap, tiled client preprocessing | Finite check 3.6 -> 0.6 ns/value; GR00T client 60 -> 33 ms (N1.6) | all |
| uint8 camera wire | The server normalizes frames with a lookup table | ACT request 7.4 -> 1.8 MB, 76 -> 24 ms | ACT (PR #24) |
| Parallel compile | `--jobs`; keep total threads within the host's cores | 9.5 -> 2.5 min (ACT, byte-identical ELFs) | build |

## Precision

- On the MLA, a layer with INT8 weights and BF16 activations quantizes its
  input to INT8 per token at run time; there is no INT16 option. Weights are
  ideal per-channel INT8. BF16 layers match CPU BF16.
- Expect INT8 weights to cost some accuracy and fix outliers before blaming
  the hardware: fused norms first, then SmoothQuant, then BF16 for the most
  sensitive graphs.
- Known exceptions: ACT's backbone saturates in INT8 (BF16 only); some
  all-BF16 denoising experts return non-finite values on ModelSDK 2.1.3
  (SmolVLA, Pi0.5), so keep their weights INT8.
- Tessellated HWC16 outputs saved 2 ms on GR00T vision but cost cosine
  (0.9996 -> 0.978); keep plain HWC unless the gain is measured worth it.

## Startup

Robot startup is part of the experience: the client's imports, the configuration
parser, the move to rest (PR #30 adds a `[startup]` timeline to every run log
and cuts the client's start from 24 to 8.6 s). Keep new policy code, and
heavy libraries, out of the robot client's import path, or import them
lazily.
