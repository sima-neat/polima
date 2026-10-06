"""Portable sparse-dataset manifest support shared by GR00T N1.x exporters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FORMAT = "polima-groot-calibration-v1"
VALID_SPLITS = frozenset({"calibration", "evaluation"})


@dataclass(frozen=True)
class SelectedObservation:
    dataset: str
    root: Path
    split: str
    episode_index: int
    frame_index: int
    instruction: str


@dataclass(frozen=True)
class CalibrationManifest:
    path: Path
    embodiment_tag: str
    observations: tuple[SelectedObservation, ...]
    raw: dict[str, Any]

    def select(self, split: str) -> tuple[SelectedObservation, ...]:
        if split not in VALID_SPLITS:
            raise ValueError(f"unknown GR00T calibration split {split!r}")
        return tuple(item for item in self.observations if item.split == split)


def load_manifest(path: str | Path, *, require_payloads: bool = True) -> CalibrationManifest:
    path = Path(path).resolve()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid GR00T calibration manifest {path}") from error
    if raw.get("format") != FORMAT:
        raise ValueError(
            f"unsupported GR00T calibration manifest format {raw.get('format')!r}"
        )

    roots = {}
    for dataset in raw.get("datasets") or ():
        name = str(dataset.get("name", ""))
        relative_root = dataset.get("root")
        if not name or not relative_root or name in roots:
            raise ValueError(f"invalid GR00T dataset entry in {path}: {dataset}")
        root = (path.parent / str(relative_root)).resolve()
        if not (root / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"GR00T sparse dataset metadata is missing under {root}")
        roots[name] = root

    observations = []
    seen = set()
    for sample in raw.get("samples") or ():
        dataset = str(sample.get("dataset", ""))
        split = str(sample.get("split", ""))
        instruction = str(sample.get("instruction", "")).strip()
        if dataset not in roots or split not in VALID_SPLITS or not instruction:
            raise ValueError(f"invalid GR00T sample in {path}: {sample}")
        episode_index = int(sample["episode_index"])
        frame_index = int(sample["frame_index"])
        identity = (dataset, split, episode_index, frame_index)
        if identity in seen:
            raise ValueError(f"duplicate GR00T sample in {path}: {identity}")
        seen.add(identity)
        observations.append(
            SelectedObservation(
                dataset=dataset,
                root=roots[dataset],
                split=split,
                episode_index=episode_index,
                frame_index=frame_index,
                instruction=instruction,
            )
        )

    manifest = CalibrationManifest(
        path=path,
        embodiment_tag=str(raw.get("embodiment_tag", "")),
        observations=tuple(observations),
        raw=raw,
    )
    if manifest.embodiment_tag != "gr1":
        raise ValueError(
            f"expected a GR1 calibration manifest, got {manifest.embodiment_tag!r}"
        )
    if not manifest.select("calibration") or not manifest.select("evaluation"):
        raise ValueError("GR00T calibration manifest requires both calibration and evaluation")

    if require_payloads:
        for filename in raw.get("required_files") or ():
            payload = path.parent / str(filename)
            if not payload.is_file():
                raise FileNotFoundError(f"GR00T sparse payload is missing: {payload}")
    return manifest


__all__ = [
    "CalibrationManifest",
    "SelectedObservation",
    "load_manifest",
]
