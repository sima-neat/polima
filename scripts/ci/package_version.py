#!/usr/bin/env python3
"""Print the PoLiMa package version for this CI ref.

Same scheme as LLiMa's published artifacts: a tag ``vX.Y.Z`` publishes
``X.Y.Z``; any branch publishes ``<package-version>+<branch>.<sha12>``.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def package_version(manifest: dict, env: dict) -> str:
    base = manifest["package-version"]
    if env.get("GITHUB_REF_TYPE") == "tag":
        tag = env["GITHUB_REF_NAME"].removeprefix("v")
        if tag != base:
            raise ValueError(f"tag {env['GITHUB_REF_NAME']} does not match package-version {base}")
        return base
    branch = env.get("GITHUB_HEAD_REF") or env.get("GITHUB_REF_NAME") or "local"
    branch = re.sub(r"[^a-z0-9.]+", ".", branch.lower()).strip(".") or "branch"
    commit = (env.get("GITHUB_SHA") or "0" * 12)[:12]
    return f"{base}+{branch}.{commit}"


if __name__ == "__main__":
    manifest = json.loads((ROOT / "deps/manifest.json").read_text(encoding="utf-8"))
    try:
        print(package_version(manifest, dict(os.environ)))
    except ValueError as error:
        sys.exit(str(error))
