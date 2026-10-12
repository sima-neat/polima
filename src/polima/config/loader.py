"""Load configuration from defaults, package and user files, environment, and CLI.

Precedence, lowest to highest:
    defaults -> package .polima.yaml/.json -> user config.json -> environment
    -> --config FILE -> explicit CLI flags.
"""

from __future__ import annotations

import os
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from polima.config import devkits
from polima.config.base import SiteConfig
from polima.config.schema import ConfigError, from_dict, merge, to_dict
from polima.util.jsonio import read_json
from polima.util.paths import package_root

ENV_PREFIX = "POLIMA_"

# $POLIMA_<NAME> -> dotted path into SiteConfig. Only settings a human would
# plausibly want to set per-shell are exposed; everything else goes in the file.
ENV_MAP = {
    "BOARD": "board.host",
    "BOARD_HOST": "board.host",
    "DEVKIT": "board.host",  # wins over the older names above
    "BOARD_ROOT": "board.root",
    "BOARD_PORT": "board.port",
    "BUILD_JOBS": "board.build_jobs",
    "DATASET_PARENT": "paths.dataset_parent",
    "OUTPUTS": "paths.outputs",
    "MODEL_COMPILER_BIN": "paths.model_compiler_bin",
    "LEROBOT_DIR": "paths.lerobot_dir",
    "GROOT16_PYTHON": "paths.groot16_python",
    "GROOT16_SOURCE": "paths.groot16_source",
    "GROOT17_PYTHON": "paths.groot17_python",
    "GROOT17_SOURCE": "paths.groot17_source",
    "GROOT17_BACKBONE": "paths.groot17_backbone",
    "CONDA_BASE": "conda_base",
    "OFFLINE": "offline",
    "DRY_RUN": "dry_run",
    "LOG_LEVEL": "log_level",
}

# Unprefixed env vars PoLiMa also honours, so an operator's existing shell keeps
# working.
UNPREFIXED_ENV_MAP = {
    "MODEL_COMPILER_BIN": "paths.model_compiler_bin",
}


def config_file_candidates() -> list[Path]:
    return [package_root() / ".polima.yaml", package_root() / ".polima.json"]


def user_config_file() -> Path:
    """The current user's persistent settings, independent of installation location."""
    root = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config").expanduser()
    return root / "polima" / "config.json"


def load(
    *,
    config_file: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    use_env: bool = True,
) -> SiteConfig:
    """Assemble a SiteConfig from all layers."""
    data: dict[str, Any] = {}

    for candidate in config_file_candidates():
        if candidate.is_file():
            data = merge(data, _read_config_file(candidate))
            break

    user_file = user_config_file()
    if user_file.is_file():
        data = merge(data, _read_config_file(user_file))

    if use_env:
        data = merge(data, _from_env())

    if config_file is not None:
        path = Path(config_file)
        if not path.is_file():
            raise ConfigError(f"--config file not found: {path}")
        data = merge(data, _read_config_file(path))

    if overrides:
        data = merge(data, _expand_dotted(overrides))

    config = from_dict(SiteConfig, data)
    devkits.apply(config.board)  # --board NAME: a DevKit saved in the Workbench
    return config


def save(config: SiteConfig, path: str | Path) -> Path:
    """Write a config back out. Used by `polima deploy` to ship robot.json to the
    DevKit."""
    from polima.util.jsonio import write_json

    return write_json(path, to_dict(config))


def _read_config_file(path: Path) -> dict:
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ConfigError(
                f"{path} needs PyYAML; install it or use .polima.json instead"
            ) from exc
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        loaded = read_json(path)
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return loaded


def _from_env() -> dict:
    flat: dict[str, Any] = {}
    for suffix, dotted in ENV_MAP.items():
        value = os.environ.get(ENV_PREFIX + suffix)
        if value not in (None, ""):
            flat[dotted] = value
    for name, dotted in UNPREFIXED_ENV_MAP.items():
        value = os.environ.get(name)
        if value not in (None, "") and dotted not in flat:
            flat[dotted] = value
    return _expand_dotted(flat)


def _expand_dotted(flat: dict[str, Any]) -> dict:
    """{"board.host": "x"} -> {"board": {"host": "x"}}, dropping None values so
    unset CLI flags do not clobber lower layers."""
    out: dict[str, Any] = {}
    for key, value in flat.items():
        if value is None:
            continue
        cursor = out
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return out


def describe(config: SiteConfig) -> list[tuple[str, str]]:
    """Flatten to (dotted_key, value) rows for `polima doctor` output."""
    rows: list[tuple[str, str]] = []

    def walk(obj: Any, prefix: str = "") -> None:
        if is_dataclass(obj) and not isinstance(obj, type):
            for f in fields(obj):
                walk(getattr(obj, f.name), f"{prefix}{f.name}.")
            return
        rows.append((prefix.rstrip("."), "" if obj is None else str(obj)))

    walk(config)
    return rows
