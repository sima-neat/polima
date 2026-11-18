#!/usr/bin/env python3
"""Print a Markdown summary of one built lane: exact LLiMa and artifact checksums."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def summary(directory: Path) -> str:
    lock = json.loads((directory / "resolved-dependencies.lock.json").read_text(encoding="utf-8"))
    llima = lock["llima"]
    lines = [
        f"### PoLiMa {lock['lane']} lane",
        "",
        "| Identity | Value |",
        "|---|---|",
        f"| LLiMa requested | `{llima['requested']}` |",
        f"| LLiMa resolved | `{llima['repository']}@{llima['ref']}:{llima['resolved_spec']}` |",
        f"| LLiMa commit | `{llima['commit']}` |",
        f"| LLiMa version | `{llima['version']}` |",
    ]
    for name, digest in sorted(llima["packages"].items()):
        lines.append(f"| {name} | `{digest}` |")
    metadata = directory / "metadata.json"
    if metadata.is_file():
        lines.append(f"| PoLiMa package | `{json.loads(metadata.read_text())['version']}` |")
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix in (".gz", ".whl"):
            lines.append(f"| {path.name} | `{hashlib.sha256(path.read_bytes()).hexdigest()}` |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.stdout.write(summary(Path(sys.argv[1])))
