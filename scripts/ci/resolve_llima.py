#!/usr/bin/env python3
"""Resolve, download and lock the exact LLiMa artifact a PoLiMa build uses.

This follows Neat Core's ``ensure_neat_llima``: read the LLiMa reference from
``deps/manifest.json``, resolve it with ``sima-cli neat install --json``, then
download that exact artifact. The released lane uses the manifest pin; the
LLiMa ``develop`` lane passes ``--ref develop:latest``. Every downloaded
package is checked against the published metadata, and the resolved identity
is written to a lock file that the build embeds and the package publishes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

REPOSITORY = "llima/debs"
REQUIRED_PACKAGES = ("core", "dev")


def manifest_selector(manifest: dict) -> str:
    """``ref:spec`` from the manifest's ``llima`` entry, as Core reads it."""
    value = manifest.get("llima")
    if not isinstance(value, dict):
        raise TypeError("deps/manifest.json must define an llima object")
    if value.get("policy"):
        raise ValueError("PoLiMa pins LLiMa by ref; llima.policy is not supported")
    ref = str(value.get("ref") or value.get("branch") or "").strip()
    spec = str(value.get("spec") or "latest").strip()
    if not ref:
        raise ValueError("llima.ref is required")
    return f"{ref}:{spec}"


def parse_resolution(text: str) -> dict:
    """The JSON object ``sima-cli neat install --json`` prints after its banner."""
    start = text.find("{")
    if start < 0:
        raise ValueError("sima-cli printed no JSON resolution")
    payload = json.loads(text[start:])
    for key in ("ref", "resolved_spec", "metadata_url", "environment"):
        if not str(payload.get(key, "")).strip():
            raise ValueError(f"sima-cli resolution has no {key}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_file(metadata: dict, kind: str) -> str:
    matches = [name for name in metadata.get("resources", [])
               if name.startswith("sima-lmm-") and name.endswith(f"-Linux-{kind}.deb")]
    if len(matches) != 1:
        raise ValueError(f"expected one LLiMa {kind} package, found {matches}")
    return matches[0]


def verify_download(directory: Path, metadata: dict, resolution: dict) -> dict[str, str]:
    """Check the download against the published metadata; return name -> sha256."""
    if metadata.get("branch") != resolution["ref"]:
        raise ValueError(f"metadata branch {metadata.get('branch')!r} != {resolution['ref']!r}")
    if not str(metadata.get("commit", "")).startswith(resolution["resolved_spec"]):
        raise ValueError("metadata commit does not match the resolved artifact")
    checksums = metadata.get("resources-checksum", {})
    verified = {}
    for kind in REQUIRED_PACKAGES:
        name = package_file(metadata, kind)
        path = directory / name
        if not path.is_file():
            raise ValueError(f"{name} was not downloaded")
        actual = sha256(path)
        if checksums.get(name) != actual:
            raise ValueError(f"{name} checksum {actual} does not match the published metadata")
        verified[name] = actual
    return verified


def build_lock(lane: str, requested: str, resolution: dict, metadata: dict,
               packages: dict[str, str]) -> dict:
    return {
        "schema_version": 1,
        "lane": lane,
        "llima": {
            "repository": REPOSITORY,
            "requested": requested,
            "environment": resolution["environment"],
            "ref": resolution["ref"],
            "resolved_spec": resolution["resolved_spec"],
            "commit": metadata["commit"],
            "version": metadata["version"],
            "metadata_url": resolution["metadata_url"],
            "packages": dict(sorted(packages.items())),
        },
    }


def run(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True,
                            env={**os.environ, "SIMA_CLI_CHECK_FOR_UPDATE": "0"})
    return result.stdout


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("deps/manifest.json"))
    parser.add_argument("--lane", choices=("release", "develop"), default="release")
    parser.add_argument("--ref", help="override the manifest, e.g. develop:latest")
    parser.add_argument("--env", default=os.environ.get("NEAT_VULCAN_ENV", "production"))
    parser.add_argument("--sima-cli", default=os.environ.get("SIMA_CLI_BIN", "sima-cli"))
    parser.add_argument("--output", type=Path, required=True, help="download directory")
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    requested = args.ref or manifest_selector(manifest)
    if args.lane == "release" and args.ref:
        raise SystemExit("the release lane uses the manifest pin; --ref is for the develop lane")

    base = [args.sima_cli, "neat", "install", "--env", args.env]
    resolution = parse_resolution(run([*base, f"{REPOSITORY}@{requested}", "--json"]))
    exact = f"{REPOSITORY}@{resolution['ref']}:{resolution['resolved_spec']}"
    metadata = fetch_json(resolution["metadata_url"])

    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise SystemExit(f"download directory is not empty: {args.output}")
    run([*base, "--download-only", "-d", str(args.output), exact])
    packages = verify_download(args.output, metadata, resolution)

    lock = build_lock(args.lane, requested, resolution, metadata, packages)
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    args.lock.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"LLiMa {lock['llima']['version']} from {exact} (commit {metadata['commit']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
