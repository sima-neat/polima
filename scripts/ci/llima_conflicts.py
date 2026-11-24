#!/usr/bin/env python3
"""List installed packages that would block installing one exact LLiMa version.

Other CI on the shared DevKit pool leaves branch builds behind, for example
NEAT Core's ``sima-neat`` pinned to ``sima-lmm-core (= 0.6.2+<branch>.<sha>)``.
Installing the LLiMa release then makes apt remove those packages, which
LLiMa's installer refuses. This reports every installed package with a
dependency on ``sima-lmm-*`` that the target version does not satisfy, as
``name<TAB>version<TAB>dependency`` lines. Versions are compared by dpkg.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

LISTING_FORMAT = "${db:Status-Abbrev}\t${Package}\t${Version}\t${Pre-Depends}, ${Depends}\n"
OPERATORS = {"<<": "lt", "<=": "le", "=": "eq", ">=": "ge", ">>": "gt", "<": "le", ">": "ge"}
RELATION = re.compile(r"(sima-lmm-[a-z0-9.+-]+?)(?::[a-z0-9-]+)?"
                      r"(?:\s*\(\s*(<<|<=|=|>=|>>|<|>)\s*([^)\s]+)\s*\))?")


def satisfies(version: str, operator: str, bound: str) -> bool:
    return subprocess.run(["dpkg", "--compare-versions", version, OPERATORS[operator], bound],
                          check=False).returncode == 0


def conflicts(listing: str, target: str) -> list[tuple[str, str, str]]:
    """Installed packages whose ``sima-lmm-*`` dependencies ``target`` breaks."""
    found = []
    for line in listing.splitlines():
        status, name, version, depends = (line.split("\t", 3) + ["", "", ""])[:4]
        if not status.startswith("ii") or name.startswith("sima-lmm-"):
            continue
        for clause in depends.split(","):
            alternatives = [part.strip() for part in clause.split("|") if part.strip()]
            matches = [RELATION.fullmatch(part) for part in alternatives]
            # A clause another package can satisfy is not this LLiMa's concern.
            if not alternatives or not all(matches):
                continue
            if not any(m[2] is None or satisfies(target, m[2], m[3]) for m in matches):
                found.append((name, version, clause.strip()))
                break
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", required=True, help="the LLiMa version to be installed")
    parser.add_argument("--listing", type=Path, help="dpkg-query output to read instead")
    args = parser.parse_args(argv)
    if args.listing:
        listing = args.listing.read_text(encoding="utf-8")
    else:
        listing = subprocess.run(["dpkg-query", "-W", "-f", LISTING_FORMAT], check=True,
                                 capture_output=True, text=True).stdout
    for row in conflicts(listing, args.target):
        print("\t".join(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
