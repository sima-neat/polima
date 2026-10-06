#!/usr/bin/env python
"""Capture one safe preview frame from each board-attached camera."""

from __future__ import annotations

import argparse
from pathlib import Path


def capture(role: str, source: str, fourcc: str) -> tuple[Path, int, int]:
    import cv2

    camera = cv2.VideoCapture(source, cv2.CAP_V4L2)
    if not camera.isOpened():
        raise RuntimeError(f"cannot open {source}")
    try:
        code = cv2.VideoWriter_fourcc(*fourcc)
        camera.set(cv2.CAP_PROP_FOURCC, code)
        frame = None
        for _ in range(12):
            ok, candidate = camera.read()
            if ok:
                frame = candidate
        if frame is None:
            raise RuntimeError(f"no frames received from {source}")
        output = Path(f"/tmp/polima-{role}-preview.jpg")
        if not cv2.imwrite(str(output), frame):
            raise RuntimeError(f"could not write {output}")
        return output, int(frame.shape[1]), int(frame.shape[0])
    finally:
        camera.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perspective", required=True)
    parser.add_argument("--wrist", required=True)
    parser.add_argument("--fourcc", default="YUYV")
    parser.add_argument("--host")
    parser.add_argument("--port")
    parser.add_argument("--focus-config")
    parser.add_argument("--view-only", action="store_true")
    args = parser.parse_args()

    failures = []
    for role, source in (("front", args.perspective), ("wrist", args.wrist)):
        try:
            output, width, height = capture(role, source, args.fourcc)
            print(f"{role}: {source} -> {width}x{height}; snapshot {output}")
        except Exception as error:  # noqa: BLE001 - report each camera, then the next
            failures.append(role)
            print(f"{role}: FAILED: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
