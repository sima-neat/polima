"""The robot client's startup timeline: seconds from process start to each phase.

Each phase is printed and recorded in the run log, so every run shows where its
launch-to-first-action time went (imports, arm and cameras, the startup move
to rest, the model server, the first action).
"""
from __future__ import annotations

import os
import time

_seen: dict[str, float] = {}


def process_start(now: float | None = None) -> float:
    """Wall-clock time this process started (Linux /proc), else the first call's time."""
    now = time.time() if now is None else now
    try:
        with open("/proc/self/stat", encoding="ascii") as stat:
            ticks = int(stat.read().rsplit(")", 1)[1].split()[19])
        with open("/proc/uptime", encoding="ascii") as uptime:
            booted = now - float(uptime.read().split()[0])
        return booted + ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return _START_FALLBACK


_START_FALLBACK = time.time()
_START = process_start()


def mark(phase: str) -> float | None:
    """Record the first time `phase` is reached; later calls for the same phase are ignored."""
    if phase in _seen:
        return None
    now = time.time() - _START
    previous = max(_seen.values(), default=0.0)
    _seen[phase] = now
    print(f"[startup] {phase.replace('_', ' ')} at {now:.1f} s (+{now - previous:.1f} s)", flush=True)
    from polima.robot.so101 import record

    record("startup", phase=phase, seconds=round(now, 3))
    return now
