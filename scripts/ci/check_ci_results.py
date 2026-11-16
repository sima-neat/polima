#!/usr/bin/env python3
"""Fail an aggregate CI job unless every required lane succeeded."""

from __future__ import annotations

import argparse
import json

COMMON_REQUIRED = (
    "resolve-test-scope",
    "resolve-sysroot",
    "resolve-vulcan-config",
    "build-release",
    "compiler",
)
PUSH_REQUIRED = (*COMMON_REQUIRED, "publish-vulcan-artifacts", "promote-latest-tag")
VALIDATION_REQUIRED = (*COMMON_REQUIRED, "build-develop")
DEVKIT_REQUIRED = ("resolve-test-runner", "test-devkit")


def failures(results: dict[str, object], *, publish: bool, devkit: bool) -> list[str]:
    required = PUSH_REQUIRED if publish else VALIDATION_REQUIRED
    if devkit:
        required = (*required, *DEVKIT_REQUIRED)
    def result(name: str) -> str:
        value = results.get(name, "missing")
        if isinstance(value, dict):
            value = value.get("result", "missing")
        return str(value)

    return [f"{name}: {result(name)}" for name in required if result(name) != "success"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="JSON object mapping job id to result")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--devkit", action="store_true")
    args = parser.parse_args()
    problems = failures(json.loads(args.results), publish=args.publish, devkit=args.devkit)
    if problems:
        print("Required CI jobs did not succeed:")
        print("\n".join(f"- {problem}" for problem in problems))
        return 1
    print("All required CI jobs succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
