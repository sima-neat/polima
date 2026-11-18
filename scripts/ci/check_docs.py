#!/usr/bin/env python3
"""Fail when customer documentation contains a broken local Markdown link."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
DOCS = (ROOT / "README.md", ROOT / "docs", ROOT / "examples")
LINK = re.compile(r"(?<!!)\[[^]]*]\(([^)\s]+)(?:\s+['\"][^)]*['\"])?\)")
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)


def markdown_files() -> list[Path]:
    files: list[Path] = []
    for source in DOCS:
        files.extend(source.rglob("*.md") if source.is_dir() else [source])
    return sorted(files)


def slug(text: str) -> str:
    """Approximate GitHub's stable heading identifier for repository docs."""
    text = re.sub(r"<[^>]+>", "", text).strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return re.sub(r"\s+", "-", text)


def anchors(path: Path) -> set[str]:
    found: set[str] = set()
    duplicates: dict[str, int] = {}
    for heading in HEADING.findall(path.read_text(encoding="utf-8")):
        base = slug(heading)
        index = duplicates.get(base, 0)
        found.add(base if index == 0 else f"{base}-{index}")
        duplicates[base] = index + 1
    return found


def local_target(source: Path, raw: str) -> tuple[Path, str] | None:
    raw = raw.removeprefix("<").removesuffix(">")
    parsed = urlsplit(raw)
    if parsed.scheme or parsed.netloc:
        return None
    path = source if not parsed.path else source.parent / unquote(parsed.path)
    return path.resolve(), unquote(parsed.fragment)


def main() -> int:
    failures: list[str] = []
    checked = 0
    cached_anchors: dict[Path, set[str]] = {}
    for source in markdown_files():
        for raw in LINK.findall(source.read_text(encoding="utf-8")):
            target = local_target(source, raw)
            if target is None:
                continue
            path, fragment = target
            checked += 1
            if not path.exists():
                failures.append(f"{source.relative_to(ROOT)}: missing {raw}")
                continue
            if fragment and path.suffix.lower() == ".md":
                headings = cached_anchors.setdefault(path, anchors(path))
                if fragment not in headings:
                    failures.append(f"{source.relative_to(ROOT)}: missing anchor {raw}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"checked {checked} local documentation links")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
