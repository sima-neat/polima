#!/usr/bin/env python3
"""Verify a PoLiMa package before it is published, independently of the build.

Checks the file set, metadata.json (with sima-cli's own validator when it is
importable), every checksum, the runtime archive layout and ELF properties,
and that the runtime, lock file and manifest record one consistent identity.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

RUNTIME_MEMBERS = ("runtime/bin/polima_cli", "runtime/bin/polima_server",
                   "runtime/tests/polima_kernel_tests", "runtime/runtime.json")
SOURCE_MEMBERS = ("pyproject.toml", "deps/manifest.json", "scripts/install_polima_modalix.sh",
                  "examples/so101/setup/install_lerobot_modalix.sh",
                  "scripts/systemd/polima-studio.service")
LOCK_FIELDS = ("ref", "resolved_spec", "commit", "metadata_url", "packages")


class PackageError(Exception):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise PackageError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def elf_properties(data: bytes) -> tuple[int, set[int]]:
    """(e_machine, dynamic tags) of a little-endian ELF64 file."""
    check(data[:4] == b"\x7fELF" and data[4] == 2 and data[5] == 1, "not a little-endian ELF64 file")
    machine = struct.unpack_from("<H", data, 18)[0]
    phoff, = struct.unpack_from("<Q", data, 32)
    phentsize, phnum = struct.unpack_from("<HH", data, 54)
    tags: set[int] = set()
    for index in range(phnum):
        kind, = struct.unpack_from("<I", data, phoff + index * phentsize)
        if kind != 2:  # PT_DYNAMIC
            continue
        offset, = struct.unpack_from("<Q", data, phoff + index * phentsize + 8)
        size, = struct.unpack_from("<Q", data, phoff + index * phentsize + 32)
        for entry in range(offset, offset + size, 16):
            tag, = struct.unpack_from("<q", data, entry)
            if tag == 0:
                break
            tags.add(tag)
    return machine, tags


def safe_members(archive: tarfile.TarFile, label: str) -> dict[str, tarfile.TarInfo]:
    members = {}
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        check(not path.is_absolute() and ".." not in path.parts, f"{label}: unsafe path {member.name}")
        check(member.isfile() or member.isdir(), f"{label}: unexpected member type {member.name}")
        members[member.name] = member
    return members


def verify(package: Path, commit: str | None = None, branch: str | None = None) -> dict:
    metadata = json.loads((package / "metadata.json").read_text(encoding="utf-8"))
    version = metadata.get("version", "")
    try:  # sima-cli's own schema, when its package is importable
        from sima_cli.install.metadata_validator import validate_metadata
    except ImportError:
        validate_metadata = None
    if validate_metadata:
        validate_metadata(metadata)

    lock = json.loads((package / "resolved-dependencies.lock.json").read_text(encoding="utf-8"))
    manifest = json.loads((package / "polima-package-manifest.json").read_text(encoding="utf-8"))
    runtime_name = f"polima-runtime-{version}-modalix.tar.gz"
    wheels = sorted(p.name for p in package.glob("polima-*.whl"))
    check(len(wheels) == 1, f"expected one wheel, found {wheels}")
    expected = {"metadata.json", "install_polima.sh", runtime_name, f"{runtime_name}.sha256",
                f"polima-source-{version}.tar.gz", wheels[0],
                "resolved-dependencies.lock.json", "polima-package-manifest.json"}
    actual = {p.name for p in package.iterdir()}
    check(actual == expected, f"package files {sorted(actual)} != {sorted(expected)}")

    check(metadata.get("name") == "gh:sima-neat/polima", "metadata name")
    check(set(metadata.get("resources", [])) == expected - {"metadata.json"}, "metadata resources")
    for name in expected - {"metadata.json"}:
        check(metadata.get("resources-checksum", {}).get(name) == sha256((package / name).read_bytes()),
              f"metadata checksum for {name}")
    check("install_polima.sh" in metadata.get("installation", {}).get("script", ""),
          "metadata installation script")
    platform = manifest["platform-version"]
    check(any(p.get("type") == "board" and "modalix" in p.get("compatible_with", [])
              and platform in p.get("version", "") for p in metadata.get("platforms", [])),
          f"metadata must target Modalix {platform}")
    if commit:
        check(metadata.get("commit") == commit and metadata.get("commit_folder") == commit[:12],
              "metadata commit")
    if branch:
        check(metadata.get("branch") == branch, "metadata branch")

    runtime_bytes = (package / runtime_name).read_bytes()
    recorded = (package / f"{runtime_name}.sha256").read_text().split()
    check(recorded == [sha256(runtime_bytes), runtime_name], "runtime archive .sha256")
    with tarfile.open(fileobj=io.BytesIO(runtime_bytes), mode="r:gz") as archive:
        members = safe_members(archive, runtime_name)
        check(all(name.startswith("runtime") for name in members), "runtime archive root")
        for name in RUNTIME_MEMBERS:
            check(name in members, f"runtime archive lacks {name}")
        for name in RUNTIME_MEMBERS[:3]:
            check(members[name].mode & 0o111, f"{name} is not executable")
            machine, tags = elf_properties(archive.extractfile(name).read())
            check(machine == 183, f"{name} is not an AArch64 executable")
            check(not tags & {15, 29}, f"{name} retains an RPATH/RUNPATH")
        runtime = json.load(archive.extractfile("runtime/runtime.json"))
    check(runtime.get("schema_version") == 2, "runtime.json schema")
    check(runtime.get("polima_version") == version, "runtime.json version != metadata version")
    check(runtime.get("platform_version") == platform, "runtime.json platform")
    check(runtime.get("lane") == lock["lane"] == "release", "only the release lane is published")
    check(runtime.get("llima_package_version") == lock["llima"]["version"], "runtime.json LLiMa version")
    check(runtime.get("llima") == {key: lock["llima"][key] for key in LOCK_FIELDS},
          "runtime.json LLiMa artifact != lock")
    if commit:
        check(runtime.get("source_commit") == commit, "runtime.json source commit")

    with tarfile.open(package / f"polima-source-{version}.tar.gz", "r:gz") as archive:
        members = safe_members(archive, "source archive")
        for name in SOURCE_MEMBERS:
            check(name in members, f"source archive lacks {name}")
        check(json.load(archive.extractfile("deps/manifest.json")) == manifest,
              "packaged manifest differs from the source manifest")

    with zipfile.ZipFile(package / wheels[0]) as wheel:
        names = wheel.namelist()
        info = Parser().parsestr(wheel.read(next(n for n in names if n.endswith(".dist-info/METADATA")))
                                 .decode())
        check(info["Name"] == "polima" and info["Version"] == manifest["package-version"],
              "wheel name/version")
        check("polima/__main__.py" in names, "wheel lacks the polima package")
        check(any(n.startswith("polima/studio/static/") for n in names), "wheel lacks Studio assets")

    return {"version": version, "runtime_sha256": sha256(runtime_bytes),
            "llima": lock["llima"]["version"], "llima_commit": lock["llima"]["commit"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--commit")
    parser.add_argument("--branch")
    args = parser.parse_args(argv)
    try:
        summary = verify(args.package, args.commit, args.branch)
    except (PackageError, KeyError, ValueError, OSError) as error:
        print(f"package verification failed: {error}", file=sys.stderr)
        return 1
    print("verified PoLiMa package " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
