#!/usr/bin/env python3
"""Decide whether a push must test its package on a DevKit.

Follows sima-neat/llima tools/ci/resolve_test_scope.py: manual runs, tags and
protected branches always test; another branch skips the DevKit only when every
file it changes against its base is documentation or a GitHub-hosted check.
Anything not listed here runs the test.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

NEUTRAL = frozenset({
    ".github/PULL_REQUEST_TEMPLATE.md", ".github/workflows/pr-sanitization.yml",
    ".github/workflows/source-ci.yml", ".gitignore", "AGENTS.md", "CONTRIBUTING.md",
    "LICENSE", "NOTICE", "README.md", "THIRD_PARTY.md",
})
NEUTRAL_PREFIXES = (".github/ISSUE_TEMPLATE/", "docs/", "skills/")


def device_paths(paths: list[str]) -> list[str]:
    """The changed paths that can change what runs on the DevKit."""
    clean = sorted({p.strip().removeprefix("./") for p in paths if p.strip()})
    return [p for p in clean if p not in NEUTRAL and not p.startswith(NEUTRAL_PREFIXES)]


def decide(event: str, ref_type: str, ref: str, changed: list[str] | None) -> tuple[bool, str]:
    """(run the DevKit test, why); `changed` is None when the base could not be compared."""
    if event == "workflow_dispatch":
        return True, "manual run"
    if ref_type != "branch":
        return True, f"{ref_type} {ref}"
    if ref in ("develop", "main") or ref.startswith("release"):
        return True, f"protected branch {ref}"
    if changed is None:
        return True, "the branch could not be compared with its base"
    hits = device_paths(changed)
    if hits:
        return True, f"{len(hits)} changed file(s) can affect the DevKit, first {hits[0]}"
    return False, "only documentation and GitHub-hosted checks changed"


def changed_files(base: str) -> list[str] | None:
    try:
        merge_base = subprocess.run(["git", "merge-base", "HEAD", base], check=True,
                                    capture_output=True, text=True).stdout.strip()
        diff = subprocess.run(["git", "diff", "--name-only", "--no-renames", merge_base, "HEAD"],
                              check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return diff.splitlines()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--event", required=True)
    parser.add_argument("--ref-type", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--base", default="origin/develop")
    parser.add_argument("--github-output", type=Path, required=True)
    parser.add_argument("--github-summary", type=Path, required=True)
    args = parser.parse_args(argv)
    run, reason = decide(args.event, args.ref_type, args.ref, changed_files(args.base))
    with args.github_output.open("a", encoding="utf-8") as output:
        output.write(f"run_devkit={str(run).lower()}\ndevkit_reason={reason}\n")
    with args.github_summary.open("a", encoding="utf-8") as summary:
        summary.write(f"DevKit test: {'run' if run else 'skip'} ({reason})\n")
    print(f"DevKit test: {'run' if run else 'skip'} ({reason})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
