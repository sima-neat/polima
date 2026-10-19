# Wire Protocol and Robot Client

## Contents

- [Wire](#wire)
- [Normalization side](#normalization-side)
- [Language and tasks](#language-and-tasks)
- [Robot client](#robot-client)
- [Cameras](#cameras)
- [Rest pose](#rest-pose)
- [Motion recipe](#motion-recipe)
- [Studio and Workbench](#studio-and-workbench)

## Wire

One TCP connection per request (`runtime/src/server_main.cpp`,
`src/polima/wire/protocol.py`):

- Request: header `<IIII` (magic, version, request id, flags), then each
  `request_tensors` entry in order as little-endian **float32**. `TensorSpec.dtype`
  does not change the framing; there is no uint8 wire on `develop` (PR #24 adds
  one for ACT: 7.4 -> 1.8 MB per request, 76 -> 24 ms).
- Response: header `<IIIIfI` (magic, version, request id, status, latency ms,
  count), then `count` float32 values; status 1 with count 0 on any error.
- The server reads the payload straight into plan buffers, so request tensor
  names must be plan buffers.
- Give each policy its own magic and default port. In use: ACT `ACTM` 8092,
  GR00T `GRUT` 8093, Pi0.5 `PI05` 8094, SmolVLA `SMOL` 8081. Bump `version`
  when the request tensors change (GR00T N1.7 is version 2).
- Large inputs: prefer sending what the client already has over making the
  board gather it. Pi0.5 sends prompt embeddings (200 x 2048 float32, 1.6 MB)
  from a memory-mapped table on the client, because a 257k-row table on the
  device would cost 1 GB of MLA-side memory and a gather; the request crosses
  localhost on the DevKit.

## Normalization side

- `client`: the robot client normalizes state and images and denormalizes
  actions with `normalization_stats.npz` (ACT). The bundle's expected actions
  are normalized.
- `server`: the plan does it with `normalize`/`denormalize` and sidecar
  statistics (GR00T, including relative actions).
- Policy-specific schemes (quantile normalization, relative actions that add
  the state except the gripper, prompt tokenization) live in a client module
  of the policy (Pi0.5 `policies/pi05/client.py`).
- Match LeRobot's arithmetic exactly, including `std + eps` (1e-8).

## Language and tasks

- Fixed instructions compiled into constants: a task catalog whose row the
  request's `flags` selects (GR00T). `bundle.json` lists `tasks` and
  `default_task`.
- Free text: tokenize and embed on the client (Pi0.5 ships the tokenizer and a
  memory-mapped embedding table in `robot_client/<policy>/`); any instruction
  works, and `POLIMA_<POLICY>_TASK` overrides the bundle's.

## Robot client

The SO-101 client is not generic yet; a new policy adds a branch in each of:

| Where | What |
| --- | --- |
| `src/polima/robot/policy.py` `CompiledPolicyAdapter._prepare_request` | Build the request tensors from camera frames, state, noise, task; any other policy raises |
| `CompiledPolicyAdapter.predict` | Decode raw actions (denormalize, relative to absolute) when not done on the server |
| `src/polima/robot/so101.py` `SUPPORTED_POLICIES`, `RECIPES` | Accept the policy; its motion recipe |
| `so101.load_contract` | Its checks: `so-arm101`, the six joints, two cameras, task rules |
| `src/polima/bundle/importer.py` `collect_robot_files` | Package the robot files (rest pose, launcher, client assets) |

Observation order: state from `RobotSpec.joint_names`, images in
`camera_roles` order. `RobotSpec.image_preprocessor` is declared but the
adapter dispatches on the policy name.

## Cameras

Each layer enforces its own names; there is no general alias table on
`develop`:

- ACT: exactly `observation.images.overhead` and `observation.images.wrist`.
- GR00T SO-101: video modalities `front`/`wrist` (default rename
  overhead -> front); the native launcher renames `front` to `overhead`.
- `so101.load_contract`: {`front`, `wrist`} or {`overhead`, `wrist`}.
- Public SO-101 checkpoints often use `up`/`side` or `top_cam`/`wrist_cam`:
  add a camera map to the builder (Pi0.5 `--camera-map CKPT_CAM=DATASET_CAM`)
  rather than special-casing a checkpoint.
- Model camera order comes from the checkpoint's processor keys, not
  `info.json` (SO-101 data lists wrist first). Swapped cameras produce wrong
  reaches with no error.
- `/dev/videoN` follows plug order: match by-id hints (`RobotSpec.camera_hints`);
  use MJPG, since two raw 640x480@30 streams exceed USB bandwidth.

## Rest pose

Every SO-101 bundle needs `robot_client/robot_rest_position.json`: the median
`observation.state` at `frame_index == 0` across the dataset's episodes
(`export/robot.py`, `compile/groot.py:write_rest_pose`). Packaging raises
without it for GR00T; make sure a new builder's bundle path writes it (Pi0.5's
LLiMa bundles missed it until `pi05_robot_files`).

## Motion recipe

`so101.RECIPES[<policy>]` sets defaults the user has not overridden:

- `fps` (commands per second), `POLIMA_CHUNK_THRESHOLD` (request the next plan
  when this fraction of the current one is left), `POLIMA_SMOOTH_MOVES`
  (spread the jump between plans), `POLIMA_SUBSTEPS` (interpolated sends per
  tick), `POLIMA_GROOT_NOISE` (`zero` keeps consecutive plans consistent).
- A plan must outlast the model: `actions / fps x threshold > model latency`.
- The per-command clamp is `max_relative_target` (default 12 degrees); the
  return to rest and the clamp are safety paths and need an arm run.

## Studio and Workbench

- Studio (DevKit) runs `polima robot run --yes --bundle ...`; policy-specific
  pieces are the auto-complete default (off for GR00T) and task lists from
  `bundle.json`.
- Workbench (build host): `workbench/app.py` `POLICIES` maps `config.json`
  `type` to a label; `workbench/inputs.py` `prerequisites` checks the export
  environment and dataset; `workbench/jobs.py` `compile_command` builds the
  `polima compile` arguments. Unknown labels raise.
