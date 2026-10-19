# Split the Model into Graphs, Export, and Verify

One `GraphSpec` becomes one ONNX file and one ELF. Its `name` is also the ONNX
stem, the bundle's `models/<name>/` directory, and the plan's `graph` key
(`src/polima/policies/base.py`, `GraphSpec`).

## Contents

- [Where to cut](#where-to-cut)
- [Static graph modules](#static-graph-modules)
- [Rewrites the compiler needs](#rewrites-the-compiler-needs)
- [LLiMa backbones](#llima-backbones)
- [Precision names](#precision-names)
- [Per-request host inputs](#per-request-host-inputs)
- [Export](#export)
- [Calibration](#calibration)
- [Fixtures and verification](#fixtures-and-verification)

## Where to cut

- Cut where data naturally changes owner: per camera through vision; per group
  of language layers; per denoising step through the action head.
- Group identical layers N per graph (GR00T: DiT blocks four per graph). Fewer,
  larger graphs mean fewer submissions; each graph must still compile and fit.
- Keep anything the host must touch between steps at a graph boundary, and
  keep that boundary small. Precompute per-step constants on the host instead
  of crossing back mid-graph.
- Put K/V that every denoising step reuses at the boundary the prefix writes
  once (GR00T `dit_kv_*` prelude graphs; SmolVLA and Pi0.5 project K/V in the
  prefix).

## Static graph modules

Write one `torch.nn.Module` per graph, built from checkpoint tensors, with
fully static shapes (`policies/<name>/graphs.py`, or `modules.py`/`blocks.py`
when large). Match the reference implementation numerically (FP32) before
exporting anything.

- **Rank 4.** Wrap tensors as `[1, 1, T, C]` (NHWC) or `[1, C, 1, T]` (NCHW).
  ACT's vision graph is NCHW and everything downstream NHWC rank 4.
- **GR00T DiT form** for transformer heads: NCHW, `Linear` as 1x1 `Conv2d`,
  einsum attention. It made Pi0.5's expert faster and more accurate.
- **Keep the batch axis** through reshapes; ModelSDK's surgery mis-folds a
  reshape that adds axes right after a dense layer.
- **Alignment.** Pad small outputs to 16 channels (ACT widens its 6 actions to
  16 and slices on the host).
- **Masks** are additive (0 / -1e4), never `-inf`. Feed each attention the
  mask it needs; GR00T alternates text and image masks by block-pair parity,
  and one shared mask attends to the wrong tokens without any error.
- **Attention with one K/V head** (GQA): expand K/V to the query heads as
  rank-4 batched matmuls; a matmul whose right operand is rank 2 becomes a
  dense layer that must be a constant.
- **Positions and timesteps** are constants or host inputs: frozen positional
  embeddings, host-computed timestep/tau embeddings, host adaRMS/AdaLN
  modulation per step (GR00T `ada_*`, Pi0.5 `ada_XX` sidecars).
- **Numerics in the native dtype**: compute position ids and similar index
  math in the dtype the tower uses (SmolVLA's FP32 bucketing dropped vision
  cosine to 0.88).
- **Statement order is ELF identity**: torch exports nodes in evaluation
  order, so reordering statements changes the ELF bytes (ACT
  `graphs.py`, `state_token`).

## Rewrites the compiler needs

| Construct | Problem | Rewrite |
| --- | --- | --- |
| RMSNorm `x * rsqrt(mean(x^2) + eps)` | Not fused: `rsqrt` becomes an INT8 lookup and the mean an INT8 conv (~6% error per step) | `Mul(x,x) -> ReduceMean(axis) -> Add(eps) -> Sqrt -> Div(x, .) -> Mul(weight)`, an all-ones weight if none; compile with `--no-simplify` (onnxsim removes `Mul(ones)`). Reduce over channels of the compiled layout: axis 1 for NCHW, the last axis for NHWC with `--model-layout NHWC` |
| LayerNorm / mean / variance where `ReduceMean` cannot be placed | Reduction not supported there | Matmul with a constant `ones/N` vector (GR00T N1.6 Eagle) |
| Standalone `Sqrt` from a matmul-mean norm | Lowered to INT8 | Prefer the fusible `ReduceMean` pattern (GR00T N1.7 per-head norm) |
| Scalar `Cast` / `Expand`, sinusoidal embeddings | Rejected | Compute on the host; ship as a sidecar or plan opcode (`sincos_time`) |
| One input with very many channels (113,664) as its own binding | Large-tensor assertion (SDK 2.1.3) | Declare it as rows, e.g. `[1, 3072, 37, 1]`, with the same bytes |
| LayerNorm right before a merger fold | Cannot go on the MLA; APU fallback crashes | Move it to the end of the previous graph (GR00T N1.7) |

After quantizing, load the `.sima` (`afe.apis.model.Model.load`) and list the
operation types: expect `RMSNormOp` and no `RsqrtOp` or `QuantizationTransformOp`
inside BF16-activation graphs.

## LLiMa backbones

When the backbone is a language model LLiMa supports, compile it with
`llima-compile` instead of exporting it. Pi0.5 is the worked example
(`src/polima/compile/pi05.py` `stage_llima` and `LLIMA_CONFIG`,
`src/polima/policies/pi05/runtime.py` `build_llima_plan`; PR #21).

**Input folder.** LLiMa compiles from a Hugging Face model folder. Repack the
checkpoint's backbone weights (vision tower, projector, language model) into
that folder with the model's own config (Pi0.5 `weights.paligemma_dir`).

**Command** (run in the ModelSDK environment, `SIMA_MLA_COMPILE_USE_JAX=0`,
GPU hidden):

```bash
llima-compile <model_folder> -o out -c layer_config.py -j 12 \
  --max_num_tokens 2048 --language_group_size <group> \
  --no-quantize_kv_cache --no-quantize_embeddings --resume
```

- `<group>`: the prefix length rounded up to a multiple of 128 (712 -> 768).
  Keep `--max_num_tokens` above it; equal values hit an SDK error.
- `layer_config.py` defines `get_layer_configuration(model_properties, layer)`,
  which returns `{"compile": bool, "precision": "BF16" | "A_BF16_W_INT8"}` for
  each `layer` (keys `part` in VISION / PRE / CACHE / POST, `index`,
  `is_group`). Compile vision, every group PRE and POST, one CACHE (attention)
  graph, no decode graphs, and skip POST graphs whose output nothing reads
  (Pi0.5 uses only the last layer's K/V).
- Outputs land in `out/sima_files/mpk/*_mpk.tar.gz`. Take the ELF out of each
  and record its per-binding sizes from the MPK (`persistent_input_elements`,
  `persistent_output_elements` in `bundle.json`).
- The generic driver (`compile/driver.py`) does not run LLiMa for
  `compiler="llima"`; a staged builder runs `llima-compile` itself.

**Using LLiMa's graphs as a policy prefix** (run them yourself in a
`run_device_chain`):

- Per layer: `pre` (RMSNorm, QKV, RoPE -> q, K, V as three outputs), the shared
  `attn` graph (q, K, mask, V -> attention), `post` (o_proj, residual, MLP).
  Hidden states ping-pong between two device buffers.
- Inputs are hidden states, not token ids: vision writes image tokens into the
  first rows, the host writes prompt embeddings after them, already multiplied
  by the model's input scale (Gemma: sqrt(width)).
- The attention mask is an input `[group, group]`, additive: broadcasting one
  key-padding row to every query row gives a bidirectional prefix; padded keys
  get -1e4. RoPE tables (`freq_real`, `freq_imag`) are cached sidecars.
- Bind each `pre` graph's K and V outputs straight to the action head's cache
  buffer; nothing returns to the host.

**Precision.** LLiMa's `A_BF16_W_INT8` quantizes each linear's input to INT8
per token at run time: it is dynamic W8A8, not W8A16.

## Precision names

The same choice has several names; keep them straight in specs and reports.

| Weights / activations | `GraphSpec` | LLiMa layer config | Bundle label | Pass mark |
| --- | --- | --- | --- | --- |
| BF16 / BF16 | `precision="bf16"` | `BF16` | `bf16` | 0.999 / 0.01 |
| INT8 / BF16 (inputs quantized per token at run time) | `weight_precision="int8"`, `activation_precision="bf16"` | `A_BF16_W_INT8` | `w8a16` (GR00T "mixed") | 0.995 / 0.03 |
| INT8 / INT8 at graph boundaries | `precision="int8"` | n/a | `w8a8` | reported only |

## Per-request host inputs

Some inputs depend on the request, not the checkpoint: when a prompt encodes
the state (Pi0.5), the number of valid prompt tokens changes the key-padding
mask and the action tokens' positions. Compute them on the host per request
(wire tensors or plan steps such as `gather_strided` over a RoPE table) and
keep the graphs' shapes fixed at the padded maximum.

## Export

- ONNX opset 17, `do_constant_folding=True`, `dynamo=False` (the legacy
  tracer; AFE does not accept dynamo graphs), under `inference_mode`, then
  `onnx.checker.check_model` (ACT `export_graph`, GR00T `_export_onnx`).
- The export runs in the policy's own interpreter (`POLIMA_<NAME>_PYTHON`),
  seeded (the generic driver seeds 123).
- Record the compiled MLA input order after compiling: it is not the ONNX
  order. Read it from the MPK (`compile/groot.py` `_mla_input_order`) and
  order the plan's inputs by it.

## Calibration

- Capture inputs of every graph from real traced activations on dataset
  observations (`calibration/<graph>.npz`, leading sample axis).
- For a denoising head, calibrate across every Euler step, not only the first
  (SmolVLA).
- BF16 graphs ignore calibration data (identical ELF).

## Fixtures and verification

- **Expected actions come from the original PyTorch policy** (LeRobot,
  Isaac-GR00T), never from your own modules or onnxruntime. Per-graph stage
  dumps may come from onnxruntime.
- Write several fixtures: dataset frames spread over the episodes, each with
  its own noise (Pi0.5 writes eight under `fixtures/set/sNN`). One request is
  a noisy test of accuracy.
- Verify the ONNX chain in onnxruntime against PyTorch with explicit
  thresholds and write the report (`onnx_verification_report.json` or the
  builder's `verification.json`). A failing verification stops the build
  before compilation.
