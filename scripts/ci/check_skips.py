#!/usr/bin/env python3
"""Fail when pytest skipped a test that tests/allowed_skips.txt does not list.

Each allow-list line is ``<test file> | <reason text>``; a skip is allowed when
it comes from that file and its reason contains the text. A missing optional
dependency therefore cannot silently turn a test suite green.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def allowed_rules(text: str) -> list[tuple[str, str]]:
    rules = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            path, _, reason = (part.strip() for part in line.partition("|"))
            if not path or not reason:
                raise ValueError(f"allow-list line needs '<file> | <reason>': {line}")
            rules.append((path, reason))
    return rules


def skips(junit: str) -> list[tuple[str, str]]:
    found = []
    for case in ET.fromstring(junit).iter("testcase"):
        skipped = case.find("skipped")
        if skipped is None:
            continue
        module = case.get("classname", "").replace(".", "/") + ".py"
        reason = skipped.get("message", "") or (skipped.text or "")
        found.append((module, reason))
    return found


def unexpected(junit: str, allow: str) -> list[str]:
    rules = allowed_rules(allow)
    return [f"{module}: {reason}" for module, reason in skips(junit)
            if not any(module == path and text in reason for path, text in rules)]


if __name__ == "__main__":
    junit_path, allow_path = map(Path, sys.argv[1:3])
    bad = unexpected(junit_path.read_text(), allow_path.read_text())
    for line in bad:
        print(f"unexpected skip: {line}", file=sys.stderr)
    sys.exit(1 if bad else 0)
