"""Robot facts a bundle takes from its training dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from polima.util.jsonio import read_json, write_json


def _episode_starts(root: Path, joint_names) -> np.ndarray:
    import pandas as pd

    starts = []
    for path in sorted((root / "data").rglob("*.parquet")):
        rows = pd.read_parquet(path, columns=["frame_index", "observation.state"])
        starts += rows.loc[rows.frame_index == 0, "observation.state"].tolist()
    states = np.asarray(starts, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != len(joint_names) or not np.isfinite(states).all():
        raise ValueError(f"no usable episode-start states in {root}")
    return states


def write_rest_pose(dataset_root: str | Path, output: str | Path,
                    joint_names: tuple[str, ...], calibration_id: str) -> Path:
    """Write the median episode-start joint state: where every demonstration began.

    The robot client starts each run here and returns here afterwards, so the
    policy always begins from a pose it saw in training. Used for ACT bundles.
    """
    root = Path(dataset_root)
    names = read_json(root / "meta/info.json")["features"]["observation.state"]["names"]
    if tuple(names) != tuple(joint_names):
        raise ValueError(f"dataset joints {names} differ from the robot's {list(joint_names)}")
    states = _episode_starts(root, joint_names)
    write_json(Path(output), {
        "source_dataset": root.name,
        "method": f"median observation.state at frame_index 0 across {len(states)} episodes",
        "calibration_id": calibration_id,
        "positions": dict(zip(joint_names, map(float, np.median(states, axis=0)))),
    })
    return Path(output)


def write_so101_rest_pose(dataset_root, output, joint_names):
    """GR00T export: the ACT convention, median episode-start state.

    Records the dataset's resolved path as its source.
    """
    root = Path(dataset_root)
    info = read_json(root / "meta/info.json")
    if tuple(info["features"]["observation.state"]["names"]) != tuple(joint_names):
        raise ValueError("Dataset joint order differs from the SO101 bundle")
    try:
        states = _episode_starts(root, joint_names)
    except ValueError:
        raise ValueError("No valid episode-start states for the SO101 rest pose") from None
    write_json(
        Path(output),
        {
            "source_dataset": root.resolve().name,
            "method": f"median observation.state at frame_index 0 across {len(states)} episodes",
            "calibration_id": "so-arm101",
            "positions": dict(zip(joint_names, map(float, np.median(states, axis=0)))),
        },
    )
