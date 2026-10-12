#!/usr/bin/env python3
"""Interactively identify the leader and follower serial ports by disconnection."""

from __future__ import annotations

import glob
import os
import sys
import time

DEVICE_PATTERNS = ("/dev/ttyACM*", "/dev/ttyUSB*")
STABLE_PATTERNS = ("/dev/serial/by-id/*", "/dev/serial/by-path/*")


def devices() -> set[str]:
    return {
        os.path.realpath(path)
        for pattern in DEVICE_PATTERNS
        for path in glob.glob(pattern)
        if os.path.exists(path)
    }


def stable_paths() -> dict[str, str]:
    """Return resolved tty -> preferred stable link (by-id before by-path)."""
    result: dict[str, str] = {}
    for pattern in STABLE_PATTERNS:
        for path in sorted(glob.glob(pattern)):
            target = os.path.realpath(path)
            if target.startswith(("/dev/ttyACM", "/dev/ttyUSB")):
                result.setdefault(target, path)
    return result


def prompt(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
    try:
        input()
    except EOFError:
        raise SystemExit("FAIL: interactive terminal input is required") from None


def wait_for_reconnect(target: str, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if target in devices():
            return True
        time.sleep(0.25)
    return False


def identify(role: str, explanation: str, forbidden: set[str]) -> tuple[str, str]:
    while True:
        before = devices()
        if not before:
            raise SystemExit("FAIL: no /dev/ttyACM* or /dev/ttyUSB* devices are connected")
        links_before = stable_paths()

        prompt(
            f"\nIdentify {role}: {explanation}\n"
            f"  Connected serial devices: {len(before)}\n"
            f"  Disconnect ONLY the {role} arm's USB cable, then press Enter."
        )
        after = devices()
        disappeared = before - after
        if len(disappeared) != 1:
            print(
                f"  Could not identify one device: {len(disappeared)} disappeared.\n"
                "  Reconnect all arms and try this role again.",
                file=sys.stderr,
            )
            prompt("  Press Enter after all arms are connected.")
            continue

        target = disappeared.pop()
        if target in forbidden:
            raise SystemExit(f"FAIL: {target} was already assigned to the other role")
        selected = links_before.get(target, target)
        print(f"  {role} identified as {selected} -> {target}", file=sys.stderr)
        prompt(f"  Reconnect the {role} arm, then press Enter.")
        if not wait_for_reconnect(target):
            print("  Device did not reappear within 15 seconds; try again.", file=sys.stderr)
            continue
        return selected, target


def main() -> int:
    print(
        "AUTOMATIC ROLE IDENTIFICATION\n"
        "  LEADER   = input/controller arm; YOU move it by hand.\n"
        "  FOLLOWER = output/robot arm; its motors copy the leader.\n"
        "Connect both arms before continuing. Prompts are shown here; no motors will move.",
        file=sys.stderr,
    )
    prompt("Press Enter when both arms are connected.")
    if len(devices()) < 2:
        raise SystemExit("FAIL: fewer than two serial devices found; connect both arms")

    leader_path, leader_target = identify(
        "LEADER", "the arm the operator moves by hand", set()
    )
    follower_path, _ = identify(
        "FOLLOWER", "the powered robot arm that will copy the leader", {leader_target}
    )

    print("\nRole identification complete.", file=sys.stderr)
    print(f"  LEADER:   {leader_path}", file=sys.stderr)
    print(f"  FOLLOWER: {follower_path}", file=sys.stderr)

    # Only these two machine-readable lines go to stdout. The launcher captures them.
    print(leader_path)
    print(follower_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
