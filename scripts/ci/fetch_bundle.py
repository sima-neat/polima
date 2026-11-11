#!/usr/bin/env python3
"""Fetch one pinned reference bundle from Hugging Face into a runner cache.

Every bundle/ file is checked against the repository's SHA256SUMS; files already
in the cache with the right checksum are kept. Repository files outside bundle/
(model card, .gitattributes) are not fetched: the Hub rewrites some of them after
the checksums are written. Prints the bundle directory.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path, PurePosixPath

HUB = "https://huggingface.co"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sums(text: str) -> dict[str, str]:
    sums = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        if name.startswith("/") or ".." in Path(name).parts or len(digest) != 64:
            raise ValueError(f"unsafe SHA256SUMS entry: {line}")
        # `sha256sum ./...` writes "./bundle/x"; the Hub URL needs "bundle/x".
        sums[PurePosixPath(name).as_posix()] = digest
    return sums


def download(url: str, destination: Path) -> None:
    headers = {}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as stream:
        for chunk in iter(lambda: response.read(1 << 20), b""):
            stream.write(chunk)


def fetch(repo: str, revision: str, cache: Path) -> Path:
    target = cache / f"{repo.replace('/', '--')}@{revision[:12]}"
    target.mkdir(parents=True, exist_ok=True)
    base = f"{HUB}/{repo}/resolve/{revision}"
    sums_path = target / "SHA256SUMS"
    download(f"{base}/SHA256SUMS", sums_path)
    for name, digest in parse_sums(sums_path.read_text()).items():
        if not name.startswith("bundle/"):
            continue
        path = target / name
        if path.is_file() and sha256(path) == digest:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        download(f"{base}/{name}", partial)
        if sha256(partial) != digest:
            partial.unlink()
            raise ValueError(f"{repo}@{revision}: {name} does not match SHA256SUMS")
        partial.replace(path)
    bundle = target / "bundle"
    if not (bundle / "bundle.json").is_file():
        raise ValueError(f"{repo}@{revision} has no bundle/bundle.json")
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo")
    parser.add_argument("revision")
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.revision) != 40:
        parser.error("pin a full 40-character revision")
    print(fetch(args.repo, args.revision, args.cache))
    return 0


if __name__ == "__main__":
    sys.exit(main())
