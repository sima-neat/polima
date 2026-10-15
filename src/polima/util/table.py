"""Tiny aligned table + status printer, for `polima doctor` and `polima list`.

Stdlib only -- this has to render on the Modalix board's bare python3.11 as well
as in a conda env.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Sequence

OK = "ok"
WARN = "warn"
FAIL = "fail"
SKIP = "skip"

_GLYPH = {OK: "+", WARN: "!", FAIL: "x", SKIP: "-"}
_COLOR = {OK: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m", SKIP: "\033[90m"}
_RESET = "\033[0m"


def _tty() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def status(kind: str, label: str, detail: str = "") -> str:
    glyph = _GLYPH.get(kind, "?")
    if _tty():
        glyph = f"{_COLOR.get(kind, '')}{glyph}{_RESET}"
    line = f" [{glyph}] {label}"
    if detail:
        line += f"  {detail}"
    return line


def render(rows: Sequence[Sequence[object]], headers: Sequence[str] | None = None) -> str:
    """Left-aligned fixed-width table. Empty input renders as an empty string."""
    body = [[("" if c is None else str(c)) for c in row] for row in rows]
    if not body and not headers:
        return ""
    columns = max(len(r) for r in ([list(headers)] if headers else []) + body)
    body = [r + [""] * (columns - len(r)) for r in body]

    widths = [0] * columns
    if headers:
        head = list(headers) + [""] * (columns - len(headers))
        for i, cell in enumerate(head):
            widths[i] = len(cell)
    for row in body:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(cells: Sequence[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

    out = []
    if headers:
        out.append(line(head))
        out.append("  ".join("-" * w for w in widths))
    out.extend(line(row) for row in body)
    return "\n".join(out)


def section(title: str) -> str:
    return f"\n{title}\n{'=' * len(title)}"


def bullets(items: Iterable[str], indent: str = "   ") -> str:
    return "\n".join(f"{indent}- {item}" for item in items)
