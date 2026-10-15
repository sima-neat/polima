"""Path resolution.

PoLiMa never stores a machine-specific path. Everything resolves from this
file's own location or from an environment variable:

  POLIMA_OUTPUTS          build and bundle outputs; default <checkout>/outputs.
  POLIMA_DATASET_PARENT   optional parent directory for LeRobot datasets.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/polima/util/paths.py -> src/polima/util -> src/polima -> src -> polima/
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def package_root() -> Path:
    """The `polima/` project directory (contains pyproject.toml, runtime/)."""
    return _PACKAGE_ROOT.parent


def native_dir() -> Path:
    return package_root() / "runtime"


def outputs_root() -> Path:
    return Path(os.environ.get("POLIMA_OUTPUTS", package_root() / "outputs"))
