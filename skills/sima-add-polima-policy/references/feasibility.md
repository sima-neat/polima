# Feasibility

Answer these before writing code. Record each answer with its evidence; mark a
question not applicable rather than skipping it silently.

## Contents

- [Provenance](#provenance)
- [Closest analogue](#closest-analogue)
- [Operators](#operators)
- [Memory and size](#memory-and-size)
- [Latency estimate](#latency-estimate)
- [Robot and dataset fit](#robot-and-dataset-fit)
- [Environments](#environments)
- [Decision](#decision)

## Provenance

| Question | Evidence |
| --- | --- |
| Which checkpoint, at which revision? | Hub ID and commit. A Hub ID maps to `~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<rev>/`, which is the `pretrained_model` folder the builders take |
| Are the weights and processors there? | `model.safetensors` (a snapshot can be config-only), and for LeRobot the `policy_preprocessor*` / `policy_postprocessor*` JSON and safetensors files that carry the normalization statistics |
| Which code defines it? | LeRobot policy type, openpi, Isaac-GR00T commit; the version installed in the export interpreter |
| Does the installed library load this checkpoint? | The config parses in the export interpreter, then a full load when the weights may be loaded on that host. A checkpoint saved by a newer LeRobot may carry config fields an older one rejects; drop only fields that are disabled, never enabled features (Pi0.5 `lerobot_reference`) |
| Which dataset calibrates it and supplies fixtures? | A local LeRobot dataset (`meta/info.json`, `data/`, videos) with the checkpoint's cameras. When `train_config.json` names only a `repo_id`, download it (`huggingface_hub.snapshot_download(repo_id, repo_type="dataset", local_dir=...)`) and pass its folder (`--dataset-root` / `--dataset`) |
| Base or fine-tuned? | A base checkpoint has no statistics for your robot (empty `features`), padded state/action (e.g. 32), and other cameras: it can be checked against PyTorch, not deployed on the arm |

## Closest analogue

| Policy | Shape | Where |
| --- | --- | --- |
| ACT | CNN backbone + transformer, one pass, 100 actions | `src/polima/policies/act/` |
| GR00T N1.6 / N1.7 | VLM backbone (Eagle / Qwen) + DiT flow-matching head, 4 steps | `src/polima/policies/groot/` |
| SmolVLA | SigLIP + small LM prefix + action expert, 10 steps | PR #25, `feature/smolvla-optimize` |
| Pi0.5 | PaliGemma (SigLIP + Gemma-2B) via LLiMa + Gemma-300M expert, 10 steps | PR #21, `feature/pi05-llima` |

Note how the new model differs from the analogue: backbone, prefix length,
action head, denoising steps, chunk length, normalization (mean/std or
quantiles), relative or absolute actions, language input.

## Operators

- Export one representative block to ONNX (opset 17, `dynamo=False`) and check
  its operators against the compiler's supported list with
  `sima-model-surgery` (it carries `supported_operators.json`). For an
  operator inventory a randomly initialized block of the right shape is
  enough; numerics need the real weights.
- Flag reductions (`ReduceMean` outside the fusible RMSNorm pattern,
  variance), scalar `Cast`/`Expand`, dynamic shapes, gathers by data-dependent
  index, and rank-5 or larger tensors. Each has a known rewrite
  (`graph-decomposition.md`).
- If the backbone is a language model LLiMa already supports (Gemma,
  Qwen, Llama; see LLiMa's docs), plan to compile it with LLiMa rather than as
  PoLiMa ONNX graphs; Pi0.5's PaliGemma prefill went from 2325 ms (ONNX graphs,
  failing accuracy) to 702 ms this way.

## Memory and size

- Weights per precision: parameters x 2 bytes (BF16) or x 1 byte (INT8
  weights), for what runs on the device only (a tied LM head used only for
  decoding stays off). ELF size tracks weights: GR00T N1.7 3.6 GB,
  Pi0.5 2 cameras 4.3 GB (BF16 vision, INT8 Gemma layers and expert), ACT 0.1 GB.
- Client-side assets count too: Pi0.5's prompt-embedding table is 1 GB
  (memory-mapped on the DevKit).
- Check the DevKit's free disk (`df -h /media/nvme`; deploy wants 1.5x the
  bundle) and memory (`free -m`, `CmaTotal`/`CmaFree` in `/proc/meminfo`) with
  the other services that will run beside the model.
- Very large single tensors (one input over ~100k channels) trip ModelSDK
  2.1.3's large-tensor handling; plan to split them into rows.

## Latency estimate

Estimate before building, then compare with the measurement.

- Prefix (vision + language): roughly proportional to tokens x layers x width.
  Pi0.5's prefix is ~12x GR00T N1.7's language compute.
- Action head: per-step time x denoising steps.
- Reference end-to-end model times on Modalix: ACT 15.8 ms, GR00T N1.7 233 ms,
  GR00T N1.6 317 ms, SmolVLA 146 ms (PR #25; denoise 7.1 ms per step).
- Pi0.5 (PR #21), INT8 weights, device-resident loop; all figures in the same
  build:

  | Cameras | Prefix tokens (group) | Prefill (vision + 18 layers) | 10 expert steps | Total |
  | --- | --- | --- | --- | --- |
  | 2 | 712 (768) | 202.6 ms | 157.3 ms (15.7 per step) | 361.4 ms |
  | 3 | 968 (1024) | 419.9 ms | 192.3 ms (19.2 per step) | 613.4 ms |

  Both use INT8 vision. Prefill grows faster than the token count because the
  group rounds up to a multiple of 128 and attention is quadratic; expert
  steps grow with the K/V the expert attends to.
- A measured single-layer or single-block probe on the DevKit is the best
  estimate; build one for the dominant block before the full export.
- A measured single-layer or single-block probe on the DevKit is the best
  estimate; build one for the dominant block before the full export.

## Robot and dataset fit

- Cameras: count, names, resolution. The SO-101 client expects two cameras
  ({`overhead`, `wrist`} or {`front`, `wrist`}); ACT requires
  `observation.images.overhead`/`wrist`, GR00T SO-101 `front`/`wrist`.
- Joints: the SO-101 client requires the six `*.pos` joints in its order, state
  and action dimension 6.
- Camera resolution and normalization: each camera's shape (the policy may fix
  one resolution for all cameras, as ACT's 480x640), and how the checkpoint
  normalizes images (`normalization_mapping` VISUAL: identity, mean/std,
  ImageNet statistics).
- Chunk length and executed steps: LeRobot's `chunk_size` and
  `n_action_steps` differ in many fine-tunes; check which the policy's
  validator requires (ACT requires both 100).
- Control rate: a plan must last longer than the model time plus margin
  (`actions / fps x chunk threshold > model latency`).
- Camera count: LeRobot fills a missing camera with a masked image; dropping a
  camera can be exact for some policies (Pi0.5) but is still a workload change
  that needs the user's agreement.
- Language: a fixed instruction compiled into constants (GR00T task catalog)
  or text tokenized on the client (Pi0.5).

## Environments

- Export interpreter: the policy's framework (torch, transformers, LeRobot or
  Isaac-GR00T) in its own environment, never the ModelSDK's. Configure it as
  `POLIMA_<NAME>_PYTHON` / `paths.<name>_python` and check it in
  `polima doctor`.
- ModelSDK (and LLiMa) version on the build host; the DevKit's platform and
  `polima_cli` build.

## Decision

Write down the route (compile only, profile, new policy, optimize), the
analogue, the expected precision per part, the latency estimate, and the open
risks. Stop and report if memory or operators rule the model out.
