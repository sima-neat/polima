"""Draw calibration samples from the dataset the checkpoint was trained on.

Shared by every policy: ACT, SmolVLA and GR00T all calibrate against real
observations passed through the checkpoint's own preprocessor, because that is
what the graphs will see at inference. Calibrating on raw dataset tensors instead
would quantize against the wrong range entirely -- the preprocessor is where
normalization happens.

The dataset is resolved from `train_config.json` inside the checkpoint rather
than being asked for, so calibration cannot silently use a different dataset than
training did. An explicit root still overrides it, because datasets get moved.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path


def resolve_dataset(checkpoint: str | Path, override: str | Path | None = None) -> tuple[str, Path]:
    """(repo_id, root) from the checkpoint's own training config."""
    checkpoint = Path(checkpoint)
    config_path = checkpoint / "train_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found; pass --dataset-root to name the dataset explicitly"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(override) if override else Path(config["dataset"]["root"])
    return config["dataset"]["repo_id"], root.resolve()


def _processor_overrides(
    checkpoint: str | Path, config_filename: str
) -> dict[str, dict[str, str]]:
    """Resolve checkpoint-relative processor artifacts for older LeRobot loaders.

    Newer checkpoints record portable assets such as SmolVLA's tokenizer in an
    ``artifacts`` mapping. The compiler's pinned LeRobot pipeline predates that
    resolution step and otherwise interprets ``tokenizer`` as a Hub repository.
    Passing the resolved paths as normal step overrides works with both loader
    generations and keeps the mounted training checkout untouched.
    """
    root = Path(checkpoint).resolve()
    overrides: dict[str, dict[str, str]] = {
        "device_processor": {"device": "cpu"}
    }
    manifest_path = root / config_filename
    if not manifest_path.is_file():
        return overrides
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for step in manifest.get("steps", []):
        step_key = step.get("registry_name") or str(step.get("class", "")).rsplit(
            ".", 1
        )[-1]
        for config_key, relative in (step.get("artifacts") or {}).items():
            artifact = Path(relative)
            if artifact.is_absolute() or ".." in artifact.parts:
                raise ValueError(
                    f"processor artifact path must be checkpoint-relative: {relative!r}"
                )
            resolved = manifest_path.parent / artifact
            if not resolved.exists():
                raise FileNotFoundError(
                    f"processor artifact {resolved} declared by {manifest_path} is missing"
                )
            overrides.setdefault(step_key, {})[config_key] = str(resolved)
    return overrides


def _load_pretrained_processors(policy, checkpoint: str | Path):
    """Restore a supported policy's saved pipelines without the global factory.

    LeRobot's factory imports configuration modules for unrelated optional
    policies at module load time. ACT and SmolVLA checkpoints already serialize
    their complete processor pipelines, so restoring those pipelines directly
    is both narrower and more faithful to training.
    """
    policy_type = getattr(policy.config, "type", None)
    if policy_type not in {"act", "smolvla"}:
        raise ValueError(
            f"shared LeRobot sample loading supports ACT and SmolVLA, got {policy_type!r}"
        )

    from lerobot.processor import (
        AbsoluteActionsProcessorStep,
        PolicyProcessorPipeline,
        RelativeActionsProcessorStep,
        batch_to_transition,
        policy_action_to_transition,
        transition_to_batch,
        transition_to_policy_action,
    )

    checkpoint = str(Path(checkpoint).resolve())
    preprocessor_config = "policy_preprocessor.json"
    postprocessor_config = "policy_postprocessor.json"
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=checkpoint,
        config_filename=preprocessor_config,
        overrides=_processor_overrides(checkpoint, preprocessor_config),
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=checkpoint,
        config_filename=postprocessor_config,
        overrides=_processor_overrides(checkpoint, postprocessor_config),
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    # Match LeRobot's factory behavior for checkpoints trained with relative
    # actions, without importing that all-policy factory.
    relative = next(
        (step for step in preprocessor.steps if isinstance(step, RelativeActionsProcessorStep)),
        None,
    )
    if relative is not None:
        for step in postprocessor.steps:
            if isinstance(step, AbsoluteActionsProcessorStep) and step.relative_step is None:
                step.relative_step = relative
    return preprocessor, postprocessor


def load(policy, checkpoint: str | Path, observation_keys: Sequence[str],
         dataset_root: str | Path | None = None, count: int = 8,
         lerobot_src: str | Path | None = None):
    """`count` preprocessed observations spread evenly across the dataset.

    Even spacing matters more than it looks: episodes are recorded in order, so
    the first N frames of a LeRobot dataset are all the same moment of the same
    episode. Calibrating on those would cover a fraction of the activation range.
    """
    import sys

    if lerobot_src and str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id, root = resolve_dataset(checkpoint, dataset_root)
    # A checkpoint may rename raw dataset features before the model sees them
    # (the combined SmolVLA run maps overhead/wrist to camera1/camera2).  The
    # graph module names processed tensors; dataset lookup must use the inverse
    # map and then let the saved preprocessor perform the rename normally.
    inverse_rename: dict[str, str] = {}
    processor_manifest = Path(checkpoint) / "policy_preprocessor.json"
    if processor_manifest.is_file():
        manifest = json.loads(processor_manifest.read_text(encoding="utf-8"))
        for step in manifest.get("steps", []):
            if step.get("registry_name") == "rename_observations_processor":
                rename = step.get("config", {}).get("rename_map") or {}
                inverse_rename.update({target: source for source, target in rename.items()})
    dataset = LeRobotDataset(repo_id=repo_id, root=str(root))
    preprocessor, postprocessor = _load_pretrained_processors(policy, checkpoint)

    indices = np.linspace(0, len(dataset) - 1, num=min(count, len(dataset)), dtype=int)
    samples = []
    for index in indices:
        item = dataset[int(index)]
        observation = {
            inverse_rename.get(key, key): item[inverse_rename.get(key, key)]
            for key in observation_keys
        }
        processed = preprocessor(observation)
        # Language-conditioned preprocessors replace the string ``task`` with
        # generated token and attention-mask tensors. Preserve every tensor the
        # processor emits rather than projecting back down to the raw keys;
        # ACT sees the same state/images as before, while SmolVLA also receives
        # ``observation.language.{tokens,attention_mask}``.
        samples.append({
            key: value.detach().cpu().float() if value.is_floating_point()
            else value.detach().cpu()
            for key, value in processed.items()
            if hasattr(value, "detach")
        })
    return samples, postprocessor, root
