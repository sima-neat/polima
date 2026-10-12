"""Find LeRobot datasets on disk: directories that hold meta/info.json."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from polima.util.jsonio import read_json_or


@dataclass(frozen=True)
class DatasetEntry:
    root: Path
    repo_id: str
    episodes: int | None = None
    frames: int | None = None
    fps: int | None = None
    codebase_version: str | None = None

    @property
    def name(self) -> str:
        return self.root.name

    def describe(self) -> str:
        parts = [self.name]
        if self.episodes is not None:
            parts.append(f"{self.episodes} eps")
        if self.frames is not None:
            parts.append(f"{self.frames} frames")
        if self.fps:
            parts.append(f"{self.fps}fps")
        return "  ".join(parts)


def is_dataset(path: str | Path) -> bool:
    path = Path(path)
    return (path / "meta" / "info.json").is_file() and (path / "data").is_dir()


def default_repo_id(root: str | Path) -> str:
    """The `local/<basename>` id used when the caller does not pass
    --dataset-id."""
    return f"local/{Path(root).resolve().name}"


def describe_dataset(root: str | Path) -> DatasetEntry:
    root = Path(root).resolve()
    info = read_json_or(root / "meta" / "info.json", {}) or {}
    return DatasetEntry(
        root=root,
        repo_id=default_repo_id(root),
        episodes=info.get("total_episodes"),
        frames=info.get("total_frames"),
        fps=info.get("fps"),
        codebase_version=info.get("codebase_version"),
    )


def discover(parent: str | Path, *, max_depth: int = 1) -> list[DatasetEntry]:
    """List datasets directly under `parent`, sorted by name."""
    parent = Path(parent)
    if not parent.is_dir():
        return []
    found: list[DatasetEntry] = []
    for candidate in sorted(parent.iterdir()):
        if candidate.is_dir() and is_dataset(candidate):
            found.append(describe_dataset(candidate))
        elif max_depth > 1 and candidate.is_dir():
            found.extend(discover(candidate, max_depth=max_depth - 1))
    return found


def resolve_roots(
    names_or_paths: list[str],
    *,
    parent: str | Path,
) -> list[Path]:
    """Accept either a bare dataset name (resolved under `parent`) or a path.

    `polima train --dataset rcwb_f_t` and
    `polima train --dataset-root /ml_datasets/rcwb_f_t` both work.
    """
    parent = Path(parent)
    roots: list[Path] = []
    for item in names_or_paths:
        candidate = Path(item)
        if candidate.is_absolute() or candidate.exists():
            roots.append(candidate.resolve())
        else:
            roots.append((parent / item).resolve())
    missing = [str(r) for r in roots if not is_dataset(r)]
    if missing:
        raise FileNotFoundError(
            "not a LeRobot dataset (no meta/info.json + data/): " + ", ".join(missing)
        )
    return roots
