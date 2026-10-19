# Runtime Plan and Opcodes

The policy's `runtime_plan_builder` (`policies/<name>/runtime.py:build_plan`)
returns a `RuntimePlan`: named float32 host buffers, an ordered list of steps,
the result buffer, and sidecar constants. The board's native `Plan`
(`runtime/src/plan.cpp`) executes it; there is no control flow, so loops are
unrolled when the plan is built.

## Contents

- [Opcodes](#opcodes)
- [Device-resident programs](#device-resident-programs)
- [Plan rules](#plan-rules)
- [Add an opcode](#add-an-opcode)
- [Measure a plan](#measure-a-plan)

## Opcodes

Host and single-graph opcodes (Python `OPCODES`, `policies/base.py`):

| Opcode | Does |
| --- | --- |
| `run_elf` | Run one graph: upload `in` buffers (concatenated in compiled input order), download to `out` |
| `run_elf_chain` | Linear chain of HWC graphs on the device: one upload, one download |
| `pack` | Scatter parts (buffers or sidecars) into a zeroed buffer at fixed offsets |
| `slice` | Contiguous copy out of a buffer |
| `gather_strided` | For `i < count`, copy `take` elements from `src + i*stride` (e.g. drop padded action lanes) |
| `pixel_unshuffle` | Fold a grid x grid x C image into (grid/f)^2 x C*f^2 |
| `scale`, `clip_unit` | Multiply by a scalar; clamp to [-1, 1] |
| `matvec` | y = b + W x with `W`, `b` from sidecars |
| `sincos_time`, `sincos_state` | Timestep embedding (min/max period); grouped sin/cos of raw angles |
| `euler` | x -= dt * v (the scalar is dt; mind the sign convention of the flow) |
| `normalize`, `denormalize` | (x - mean)/std and x*std + mean with sidecar statistics |

Native-only chain opcodes (`runtime/include/polima/*.hpp`):

| Opcode | For | Header |
| --- | --- | --- |
| `run_dit_chain` | GR00T N1.6's 16 two-block DiT graphs with 5 persistent inputs | `dit_chain.hpp` |
| `run_bound_chain` | Graphs with named persistent bindings: a prelude (e.g. K/V projection once per request), host-fed inputs uploaded only when changed, a carried state that ping-pongs | `bound_chain.hpp` |
| `run_device_chain` | A general device program: plan-wide BF16 `device_buffers`, uploads (optionally cached across requests), graph calls bound to sub-spans of those buffers (several inputs and outputs per call), downloads | `device_chain.hpp` |

On `develop`, Python `OPCODES` does not list these three, so `Step.validate`
rejects them in a spec's plan. GR00T adds them by rewriting `plan.json` after
validation through `BundleInputs.plan_rewrite` (`bundle/pack.py`,
`policies/groot/dit.py`, `bundle/device_chain.py`). Pi0.5 (PR #21) adds
`run_device_chain` to `OPCODES` and emits it directly with `device_buffers` on
`RuntimePlan`; prefer that path for new device programs. If PR #21 has not
merged, port that small change (the `OPCODES` entry, `RuntimePlan.device_buffers`,
and `bundle/pack.py` writing `device_buffers`) rather than adding another
`plan_rewrite`; the rewrite path is GR00T's legacy.

## Device-resident programs

Keep intermediates in MLA memory instead of round-tripping through the host:

- Vision writes straight into the language input's slots; each layer's K/V
  goes straight into the action head's cache input (ACT, SmolVLA, Pi0.5).
- A denoising loop can run as one program: compile the step with the Euler
  update folded in (`x - dt*v`) and one persistent binding per input
  (`compile_onnx.py persistent`), let `x` ping-pong between two device
  buffers, and bind each step's own cached conditioning buffer (Pi0.5
  `runtime._device_denoise`).
- Rules the native checker enforces: every span a call reads was written
  earlier in the same request; cached upload regions are never written by
  anything else; a call never reads and writes overlapping spans; INT8 graphs
  cannot join a chain (their boundaries are requantizations).
- One `MLAModelWithBuffer` per call; instances of the same ELF share one
  loaded model, so repeating a graph costs no memory.
- The queue never submits the same ELF twice in one batch (sima-lmm 0.4.0
  returned wrong results for the first instance); `queue_segments` splits it.
- Persistent bindings are presented in lexicographic order of their names
  (`input_10` before `input_2`); the bundle lists them that way and the Plan
  refuses otherwise.

## Plan rules

- Buffer sizes are element counts. `PolicySpec.validate` checks that each
  `run_elf` `in` totals the first graph's inputs and `out` its last graph's
  outputs.
- Sidecars are `constants/<name>.f32` files loaded by name (prompt embeddings,
  timestep embeddings, adaRMS vectors, statistics).
- The task id travels in the request header's `flags`; a `pack` part with
  `selector_count` picks that row of a sidecar (GR00T's task catalog). It is
  range-checked.
- Every MLA output is checked for non-finite values in its decode pass.
- Keep a Python reference of the plan in step: `StubPlan` in
  `src/polima/wire/server_stub.py` executes plans for tests and must match
  `plan.cpp` (it has no `run_bound_chain` branch).

## Add an opcode

Only when no combination of existing opcodes expresses the step.

1. `OPCODES` in `policies/base.py`; extend the graph and element-count checks
   in `RuntimePlan.validate`, `PolicySpec.validate`, and
   `bundle/pack.py:_check_plan_against_elfs` if it runs graphs.
2. `runtime/include/polima/plan.hpp`: the `Opcode` value and any new `Step`
   fields.
3. `runtime/src/plan.cpp`: `parse_opcode`, `opcode_name`, argument decoding,
   load-time validation, the `run_step` case; for MLA opcodes also the graph
   scan, `finish_step` and, if lanes apply, `lanes_match`.
4. `src/polima/wire/server_stub.py` `StubPlan._run_step`: the Python
   reference.
5. MLA opcodes: the `mla_ops` set in `workbench/app.py`.
6. Tests: `tests/unit/test_plan.py` (StubPlan), `tests/unit/test_device_chain.py`
   (builds `plan.cpp` against `tests/native/llima_mock` and compares with the
   reference), `tests/unit/test_native_kernels.py` for kernels.

## Measure a plan

- `polima_cli --interactive --bundle <id>`: `bench [n] [warmup]` for
  min/median/p95, `stages` for per-step time, `check` against the bundle's
  pass mark.
- `POLIMA_PROFILE_PHASES=1` prints, per device chain, the host encode, MLA
  execute, and decode times; it shows how much of a step is host work.
- Take `/tmp/polima-bench.lock` around measurements on a shared DevKit.
