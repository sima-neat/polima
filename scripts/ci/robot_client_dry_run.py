#!/usr/bin/env python3
"""Exercise the installed SO-101 policy client without connecting to an arm."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from polima.robot.so101 import SO101PolicyClient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    os.environ["POLIMA_SO101_BUNDLE"] = str(args.bundle)
    os.environ["POLIMA_TASK_ID"] = "0"
    client = SO101PolicyClient(args.server)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    state = np.zeros(6, dtype=np.float32)
    actions, _ = client.predict(image, image, state)
    actions = np.asarray(actions)
    expected = tuple(client.adapter.spec.wire.response_shape)
    if actions.shape != expected:
        raise ValueError(f"robot client returned {actions.shape}, expected {expected}")
    if not np.isfinite(actions).all():
        raise ValueError("robot client returned non-finite actions")
    args.output.write_text(json.dumps({"shape": list(actions.shape), "finite": True}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
