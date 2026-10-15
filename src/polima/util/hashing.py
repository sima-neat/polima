"""Content hashing for bundle identity and build-skip decisions.

Bundle ids are content-addressed (see polima.bundle.layout.compute_bundle_id), so
redeploying an identical build is a no-op and rollback is an `ln -sfn`. The board
currently carries 26 accreted deploy roots with no way to tell which is live;
this module is what replaces that.

Generalizes the `digest`/`sha256` helpers duplicated across the three compile
controllers.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

_CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_tree(root: str | Path, *, patterns: Iterable[str] = ("**/*",),
                exclude: Iterable[str] = ()) -> str:
    """Hash a directory's *content and layout*.

    Used for the `src_hash` that lets `polima deploy` skip the on-board cmake
    build when runtime/ has not changed. Paths are recorded relative to `root` and
    sorted, so the result is independent of filesystem iteration order.
    """
    root = Path(root)
    exclude = tuple(exclude)
    seen: set[Path] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file() and not any(path.match(rule) for rule in exclude):
                seen.add(path)

    digest = hashlib.sha256()
    for path in sorted(seen):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def short(digest: str, length: int = 8) -> str:
    return digest[:length]
