"""Logging setup, configured in one place."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_CONFIGURED = False
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"


def setup(level: str | int | None = None, *, log_file: str | Path | None = None,
          force: bool = False) -> logging.Logger:
    """Configure the root logger once. `POLIMA_LOG_LEVEL` overrides `level`."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return logging.getLogger("polima")

    resolved = os.environ.get("POLIMA_LOG_LEVEL") or level or "INFO"
    if isinstance(resolved, str):
        resolved = getattr(logging, resolved.upper(), logging.INFO)

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
    root.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
        root.addHandler(file_handler)

    root.setLevel(resolved)
    _CONFIGURED = True
    return logging.getLogger("polima")


def get(name: str) -> logging.Logger:
    return logging.getLogger(f"polima.{name}" if not name.startswith("polima") else name)
