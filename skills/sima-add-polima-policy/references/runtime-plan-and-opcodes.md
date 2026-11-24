# Runtime plans and device execution

`PolicySpec.runtime_plan_builder` returns the `RuntimePlan` serialized into the
bundle. The native runtime executes the plan; the Python server stub provides a
test reference for supported operations.

## Existing operations

Inspect `OPCODES` and validation in `src/polima/policies/base.py`, native plan
execution in `runtime/src/plan.cpp`, and the Python reference in
`src/polima/wire/server_stub.py`. Current plans cover:

- graph execution and compatible linear graph chains;
- packing, slicing, and strided gathers;
- normalization, denormalization, scaling, clipping, and small matrix/vector
  operations;
- time/state features and Euler updates;
- general device-resident programs through `run_device_chain`.

GR00T also contains older specialized plan-rewrite paths. Reuse them only when
maintaining compatible GR00T bundles. Prefer the validated `RuntimePlan` and
`run_device_chain` path for new policies.

## Plan rules

- Declare every host buffer and plan-wide device buffer with exact element
  counts.
- Match graph input/output order to the compiled artifact, not an assumed ONNX
  order.
- Keep shared device edges compatible in size, layout, and precision.
- Upload host inputs once when they remain unchanged across repeated calls.
- Keep cached inputs immutable for the duration expected by the plan.
- Download only the final values the client needs.
- Range-check task selectors and action slicing.
- Validate the plan against `CompilePlan` and the packed ELFs before deploy.

## Adding an operation

Add an opcode only when existing operations cannot express the computation.
Update all affected layers together:

1. `src/polima/policies/base.py` validation and serialization contract.
2. Native parse, validation, execution, and naming under `runtime/`.
3. `src/polima/wire/server_stub.py` reference behavior.
4. Bundle/ELF consistency checks and Workbench inspection where applicable.
5. Unit tests comparing Python and native results, including invalid plans.

Do not add a model-specific native operation for arithmetic that can remain in
the compiled graph or an existing device chain.

## Measurement

Use the bundle's own check before benchmarking. Record warm-ups, sample count,
runtime revision, bundle revision, precision, camera count, action horizon, and
denoising steps. Measure server latency, stage latency, camera-to-plan latency,
memory, and queue behavior separately. Take the repository's shared benchmark
lock when measuring a shared DevKit.
