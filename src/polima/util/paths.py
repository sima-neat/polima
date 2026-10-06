"""Path resolution.

PoLiMa never stores a machine-specific path. Everything resolves from this
file's own location, which is the convention every existing script already
follows via BASH_SOURCE, or from an environment variable:

  POLIMA_REPO_ROOT        the workspace that holds the model stacks (ACT/,
                          SmolVLA/, GR00T-N1.6/, lerobot_sima/ ...); default:
                          the parent of this checkout.
  POLIMA_WORKSPACE_LINK   optional convenience symlink that should point at
                          that workspace (some training scripts reach the
                          stacks through one). `polima doctor` checks it.
  POLIMA_OUTPUTS          build and bundle outputs; default <checkout>/outputs.
  POLIMA_DATASET_PARENT   where LeRobot datasets live; default /ml_datasets.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/polima/util/paths.py -> src/polima/util -> src/polima -> src -> polima/
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def package_root() -> Path:
    """The `polima/` project directory (contains pyproject.toml, native/, board/)."""
    return _PACKAGE_ROOT.parent


def repo_root() -> Path:
    """The workspace holding the model stacks (ACT/, SmolVLA/, GR00T-N1.6/, ...).

    $POLIMA_REPO_ROOT when set, else the parent of this checkout.
    """
    override = os.environ.get("POLIMA_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return package_root().parent


def native_dir() -> Path:
    return package_root() / "native"


def board_scripts_dir() -> Path:
    return package_root() / "board"


def legacy_stack(name: str) -> Path:
    """Path to a legacy stack dir. Read-only for PoLiMa through Phase 5."""
    return repo_root() / name


def outputs_root() -> Path:
    return Path(os.environ.get("POLIMA_OUTPUTS", package_root() / "outputs"))


def default_dataset_parent() -> Path:
    return Path(os.environ.get("POLIMA_DATASET_PARENT", "/ml_datasets"))


def symlink_report() -> dict:
    """Facts about the optional $POLIMA_WORKSPACE_LINK symlink for `polima doctor`."""
    configured = os.environ.get("POLIMA_WORKSPACE_LINK")
    resolved = repo_root()
    if not configured:
        return {
            "home_symlink": None,
            "exists": False,
            "is_symlink": False,
            "target": None,
            "resolved_repo_root": str(resolved),
            "consistent": True,
        }
    link = Path(os.path.expanduser(configured))
    return {
        "home_symlink": str(link),
        "exists": link.exists(),
        "is_symlink": link.is_symlink(),
        "target": str(link.resolve()) if link.exists() else None,
        "resolved_repo_root": str(resolved),
        "consistent": link.exists() and link.resolve() == resolved,
    }


def expand(path: str | Path) -> Path:
    """`~` and `$VAR` expansion, then resolve. Callers store what the user typed,
    never the resolved path."""
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()
