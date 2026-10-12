"""Canonical JSON read/write.

Every manifest PoLiMa writes (bundle.json, plan.json, deploy_report.json,
artifact_manifest.json) goes through here so byte-level comparison between two
builds is meaningful -- `compute_bundle_id` hashes canonical JSON, so key order
and separators must be stable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def dumps(value: Any) -> str:
    """Canonical form: sorted keys, 2-space indent, trailing newline."""
    return json.dumps(value, indent=2, sort_keys=True, default=_fallback) + "\n"


def dumps_compact(value: Any) -> str:
    """Canonical single-line form, for hashing and for log lines."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_fallback)


def write_json(path: str | Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(value), encoding="utf-8")
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_json_or(path: str | Path, default: Any = None) -> Any:
    """Read, returning `default` when the file is missing or unparsable."""
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError):
        return default


def append_jsonl(path: str | Path, record: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(dumps_compact(record) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[Any]:
    path = Path(path)
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _fallback(value: Any) -> Any:
    """Make Path, set, tuple and numpy scalars serializable without importing numpy."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if hasattr(value, "item") and callable(value.item):  # numpy scalar, duck-typed
        return value.item()
    if hasattr(value, "tolist") and callable(value.tolist):  # numpy array
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")
