# Bundle, wire, and robot integration

Reuse the current bundle and SO-101 path. A policy-specific model adapter may
prepare requests and decode actions; it must not duplicate motor control,
camera lifecycle, calibration, recovery, logging, or return-to-rest behavior.

## Wire contract

`WireSpec` defines request headers, tensor order and dtype, response shape,
normalization side, protocol version, and default port. The packer, Python
client, and native server derive their behavior from that contract.

- Choose a unique magic, version, and port after inspecting the registered
  policies; do not copy a remembered value.
- Use `TensorSpec.dtype` for supported wire encodings. Camera tensors can use
  `uint8` when the server reproduces the policy's normalization exactly.
- Bump the protocol version when the request tensor contract changes.
- Keep request names aligned with runtime-plan buffers.
- Compare wire encoding/decoding numerically with upstream preprocessing.

## Normalization and actions

Declare whether normalization occurs on the client or server. Match the
upstream order and arithmetic, including epsilon, quantiles, relative-action
conversion, joint order, and gripper treatment. Expected fixture actions must
be in the same units and coordinate system returned over the wire.

## Tasks and language

- A compiled task catalog selects among instructions embedded in bundle
  constants. Validate the selected task ID and list the catalog in
  `bundle.json`.
- A free-text policy tokenizes/prepares the instruction in its policy client.
  Package every tokenizer or embedding asset required offline.
- An ACT-style policy with no language input may display a task label, but the
  label does not change inference.

## Bundle contents

Trace packaging through `src/polima/bundle/importer.py` and the closest policy
compiler. A runnable robot bundle needs more than ELFs:

- `bundle.json`, plan, model artifacts, constants, and fixtures;
- the policy client assets required on the DevKit;
- an SO-101 rest pose and the declared robot/camera contract;
- integrity metadata used by deployment and checking.

Verify the installed bundle rather than only the host build directory.

## Shared robot path

Start with:

- `src/polima/robot/policy.py` for model-specific request preparation and
  action decoding;
- `src/polima/robot/so101.py` for accepted policies, recipes, task handling,
  calibration, and launcher configuration;
- the packaged client under `src/polima/robot/assets/so101/` for the control
  loop, cameras, recovery, and logging.

Add the smallest policy-specific branch at the model boundary. Keep camera and
joint observations ordered by the specialized `RobotSpec`.

## Cameras, calibration, and motion

- Match cameras through stable by-id identities and declared role hints. Do
  not rely on `/dev/videoN` ordering.
- Treat checkpoint camera order and preprocessing as part of the model.
- Preserve existing arm calibration; installation must not overwrite it.
- Derive and package the rest pose using the established compiler path.
- Keep command rate, chunk threshold, aggregation, smoothing, substeps,
  automatic completion, and safety clamps configurable. Model defaults belong
  in one policy recipe, not scattered across the client.

A no-motion client check can validate camera preprocessing, task preparation,
wire requests, and action shapes. Only a supervised arm test establishes
motion quality, recovery, and return-to-rest behavior.

## Product integration

Check every surface that discovers or launches a policy: registry and compile
routing, export environment discovery, bundle packing, deploy, Workbench model
identification/prerequisites/commands, Studio selection and task handling,
`polima doctor`, install assets, and customer documentation.
