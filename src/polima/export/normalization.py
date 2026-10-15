"""Pull normalization statistics out of a LeRobot checkpoint.

Normalization is host-owned in every SiMa deployment here: the MLA graphs take
already-normalized tensors and return normalized actions, and the board's C++
does the mean/std arithmetic. So these numbers have to travel with the bundle,
and they have to come from the checkpoint rather than the dataset -- a policy
trained on one normalization and served with another is silently wrong, not
broken, which is the worst failure mode available.

LeRobot stores them indirectly. `policy_preprocessor.json` lists processor steps;
the one named `normalizer_processor` names a safetensors file, and that file
holds `<feature>.mean` / `<feature>.std`. The unnormalizer for actions is the
mirror image in `policy_postprocessor.json`. Following that indirection is the
whole job, and it is identical for every policy -- which is why this lives in
`export/` rather than under a policy.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

PREPROCESSOR = "policy_preprocessor.json"
POSTPROCESSOR = "policy_postprocessor.json"


def _state_file(checkpoint: Path, manifest: str, registry_name: str) -> Path:
    path = checkpoint / manifest
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- is {checkpoint} a LeRobot `pretrained_model` directory?"
        )
    steps = json.loads(path.read_text(encoding="utf-8"))["steps"]
    for step in steps:
        if step.get("registry_name") == registry_name:
            return checkpoint / step["state_file"]
    raise KeyError(
        f"{path} has no {registry_name!r} step; found "
        f"{[s.get('registry_name') for s in steps]}"
    )


def from_lerobot_checkpoint(checkpoint: str | Path, camera_keys: Sequence[str],
                            state_key: str = "observation.state",
                            action_key: str = "action") -> dict:
    """Return the arrays a bundle needs, keyed as the board expects them.

    Camera statistics are indexed positionally (`camera0_mean`, ...) because the
    board addresses cameras by slot, not by LeRobot feature name. The order is
    the policy's `camera_roles` order and must match how the runtime packs them.
    """
    import numpy as np
    from safetensors.torch import load_file

    checkpoint = Path(checkpoint)
    pre = load_file(_state_file(checkpoint, PREPROCESSOR, "normalizer_processor"))
    post = load_file(_state_file(checkpoint, POSTPROCESSOR, "unnormalizer_processor"))

    arrays = {
        "state_mean": pre[f"{state_key}.mean"].numpy(),
        "state_std": pre[f"{state_key}.std"].numpy(),
        "action_mean": post[f"{action_key}.mean"].numpy(),
        "action_std": post[f"{action_key}.std"].numpy(),
    }
    for index, key in enumerate(camera_keys):
        arrays[f"camera{index}_mean"] = pre[f"{key}.mean"].numpy()
        arrays[f"camera{index}_std"] = pre[f"{key}.std"].numpy()

    zero_std = [name for name, value in arrays.items()
                if name.endswith("_std") and float(np.min(np.abs(value))) == 0.0]
    if zero_std:
        raise ValueError(
            f"{checkpoint} has a zero standard deviation in {zero_std}; "
            "dividing by it would produce inf on the board"
        )
    return arrays


def write(checkpoint: str | Path, camera_keys: Sequence[str], output: str | Path) -> Path:
    import numpy as np

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **from_lerobot_checkpoint(checkpoint, camera_keys))
    return output
