"""Move bundles and files between the operator's browser and the board.

The build host is often not on the robot LAN, but the operator's browser reaches
Studio. Uploads are chunked and resumable: Waitress spools each request body to a
temporary file and refuses bodies over 1 GiB, and a GR00T bundle is larger than
that. Installation applies the checks `polima deploy` applies after rsync -- every
ELF's sha256 against bundle.json, and no archive inside the bundle.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import tarfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

CHUNK_MAX = 64 << 20
UPLOAD_MAX = 64 << 30
ARCHIVES = (".tar", ".tar.gz", ".tgz")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
CALIBRATION_FIELDS = ("id", "drive_mode", "homing_offset", "range_min", "range_max")
_UPLOAD_ID = re.compile(r"^[0-9a-f]{16}$")
_IO = 1 << 20

Progress = Callable[[str, int, int], None]


def safe_relpath(value: str) -> PurePosixPath:
    """A relative path that cannot leave the directory it is joined to."""
    path = PurePosixPath(str(value).replace("\\", "/"))
    if not str(value).strip() or path.is_absolute() or {"", ".", ".."} & set(path.parts):
        raise ValueError(f"unsafe path: {value!r}")
    return path


def is_archive(name: str) -> bool:
    return name.lower().endswith(ARCHIVES)


def confined(base: Path, relative: str = "") -> Path:
    base = base.resolve()
    target = (base / safe_relpath(relative)).resolve() if relative else base
    if target != base and base not in target.parents:
        raise ValueError("path leaves its download root")
    return target


class Uploads:
    """Staged, resumable uploads: <staging>/<id>/{upload.json,files/}."""

    def __init__(self, staging: Path) -> None:
        self.staging = staging

    def path(self, upload_id: str) -> Path:
        directory = self.staging / upload_id
        if not _UPLOAD_ID.match(upload_id or "") or not (directory / "upload.json").is_file():
            raise FileNotFoundError("unknown upload")
        return directory

    def create(self, files: list[dict[str, Any]], free_bytes: int) -> dict[str, Any]:
        if not isinstance(files, list) or not files:
            raise ValueError("choose a bundle folder or a .tar archive")
        entries = [{"path": str(safe_relpath(item.get("path", ""))),
                    "size": int(item.get("size", -1))} for item in files]
        paths = [entry["path"] for entry in entries]
        if len(set(paths)) != len(paths) or any(entry["size"] < 0 for entry in entries):
            raise ValueError("every file needs a unique path and a size")
        archive = len(entries) == 1 and is_archive(paths[0])
        if len(entries) > 1 and any(map(is_archive, paths)):
            raise ValueError("a bundle folder must not contain .tar archives")
        total = sum(entry["size"] for entry in entries)
        needed = total * (2.1 if archive else 1.1)  # an archive is also extracted
        if total > UPLOAD_MAX or (free_bytes and needed > free_bytes):
            raise ValueError(f"not enough disk space for {total / 1e9:.1f} GB")
        upload_id = secrets.token_hex(8)
        (self.staging / upload_id / "files").mkdir(parents=True)
        meta = {"id": upload_id, "created": time.time(), "archive": archive,
                "files": entries, "total": total}
        (self.staging / upload_id / "upload.json").write_text(json.dumps(meta))
        return self.status(upload_id)

    def status(self, upload_id: str) -> dict[str, Any]:
        directory = self.path(upload_id)
        meta = json.loads((directory / "upload.json").read_text())
        for entry in meta["files"]:
            target = directory / "files" / entry["path"]
            entry["received"] = target.stat().st_size if target.exists() else 0
        meta["received"] = sum(entry["received"] for entry in meta["files"])
        meta["complete"] = meta["received"] == meta["total"]
        return meta

    def pending(self) -> list[dict[str, Any]]:
        names = sorted(p.name for p in self.staging.iterdir()) if self.staging.is_dir() else []
        result = []
        for name in names:
            try:
                result.append(self.status(name))
            except (OSError, ValueError):
                pass
        return result

    def receive(self, upload_id: str, path: str, offset: int, stream: BinaryIO,
                length: int) -> dict[str, Any]:
        if not 0 < length <= CHUNK_MAX:
            raise ValueError(f"chunks are 1 byte to {CHUNK_MAX >> 20} MiB")
        path = str(safe_relpath(path))
        state = self.status(upload_id)
        entry = next((item for item in state["files"] if item["path"] == path), None)
        if entry is None:
            raise ValueError(f"{path} is not part of this upload")
        if offset != entry["received"]:  # the browser resumes from what is on disk
            return {"ok": False, "path": path, "received": entry["received"]}
        if offset + length > entry["size"]:
            raise ValueError(f"{path}: chunk runs past the declared size")
        target = self.path(upload_id) / "files" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("ab") as handle:
            while length and (block := stream.read(min(_IO, length))):
                handle.write(block)
                length -= len(block)
        if length:
            raise ValueError(f"{path}: connection closed mid-chunk; resume to continue")
        return {"ok": True, "path": path, "received": target.stat().st_size}

    def discard(self, upload_id: str) -> None:
        shutil.rmtree(self.path(upload_id), ignore_errors=True)

    def prune(self, max_age_s: float = 7 * 86400) -> None:
        """Abandoned partial uploads can be gigabytes; keep a week."""
        for item in self.pending():
            if item["created"] < time.time() - max_age_s:
                self.discard(item["id"])

    def tree(self, upload_id: str, progress: Progress) -> Path:
        """The uploaded bundle as a directory, extracting an archive if needed."""
        state, directory = self.status(upload_id), self.path(upload_id)
        if not state["complete"]:
            raise ValueError(f"upload is incomplete ({state['received']} of {state['total']})")
        if not state["archive"]:
            return directory / "files"
        archive = directory / "files" / state["files"][0]["path"]
        extract_archive(archive, directory / "tree", progress)
        archive.unlink()
        return directory / "tree"


def extract_archive(archive: Path, destination: Path, progress: Progress) -> None:
    """Extract regular files and directories only, never outside `destination`."""
    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        total, done = sum(m.size for m in members if m.isfile()), 0
        for member in members:
            name = re.sub(r"^(\./)+", "", member.name)
            if not name.strip("/."):
                continue
            target = destination / safe_relpath(name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or is_archive(name):
                raise ValueError(f"archive member {member.name!r} is not a bundle file")
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open("wb") as handle:
                while block := source.read(_IO):
                    handle.write(block)
                    done += len(block)
                    progress("extracting", done, total)
            target.chmod(0o755 if member.mode & 0o111 else 0o644)


def find_bundle_root(tree: Path) -> Path:
    """The single shallowest directory holding bundle.json."""
    for depth in range(3):
        found = sorted(p.parent for p in tree.glob("*/" * depth + "bundle.json"))
        if len(found) > 1:
            raise ValueError("the upload holds several bundles; install one at a time")
        if found:
            return found[0]
    raise ValueError("no bundle.json found: upload a PoLiMa bundle folder or a .tar of one")


def read_manifest(bundle: Path) -> dict[str, Any]:
    try:
        manifest = json.loads((bundle / "bundle.json").read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"bundle.json is unreadable: {error}") from error
    if not isinstance(manifest, dict) or not SAFE_ID.match(str(manifest.get("bundle_id"))):
        raise ValueError("bundle.json has no usable bundle_id")
    if not manifest.get("policy") or not manifest.get("graphs"):
        raise ValueError("bundle.json names no policy or compiled graphs")
    return manifest


def verify_elfs(bundle: Path, manifest: dict[str, Any], progress: Progress) -> list[str]:
    """Hash every ELF against bundle.json, as `polima deploy`'s verify stage does.

    A truncated transfer produces an ELF that loads and returns garbage, so a
    digest comparison is the only way to rule that out.
    """
    graphs = manifest["graphs"]
    total, done, problems = sum(int(g.get("elf_bytes") or 0) for g in graphs), 0, []
    for graph in graphs:
        path = bundle / safe_relpath(str(graph.get("elf", "")))
        if not path.is_file():
            problems.append(f"{graph.get('name')}: ELF missing")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(_IO):
                digest.update(block)
                done += len(block)
                progress("verifying", done, total)
        if digest.hexdigest() != graph.get("sha256"):
            problems.append(f"{graph.get('name')}: sha256 mismatch, transfer corrupted")
    return problems


def archives_in(bundle: Path) -> list[str]:
    return sorted(str(p.relative_to(bundle)) for p in bundle.rglob("*") if is_archive(p.name))


def tree_size(path: Path) -> tuple[int, int]:
    files = [p for p in path.rglob("*") if p.is_file() and not p.is_symlink()]
    return sum(p.stat().st_size for p in files), len(files)


def install_bundle(tree: Path, models: Path, scratch: Path, progress: Progress,
                   replaceable: Callable[[str], None]) -> dict[str, Any]:
    """Verify an uploaded bundle tree and move it into the model store.

    `replaceable(bundle_id)` raises when the installed copy is in use. A
    replacement keeps the board's calibration files: they describe the arm on
    this bench, which the uploaded copy cannot know.
    """
    bundle = find_bundle_root(tree)
    manifest = read_manifest(bundle)
    bundle_id = manifest["bundle_id"]
    problems = verify_elfs(bundle, manifest, progress) + archives_in(bundle)
    if problems:
        raise ValueError("bundle verification failed: " + "; ".join(problems))
    destination, kept = models / bundle_id, []
    replaced = destination.exists()
    if replaced:
        replaceable(bundle_id)
        for path in sorted(destination.glob("robot_client/calibration/*.json")):
            target = bundle / "robot_client/calibration" / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            kept.append(path.name)
    for path in bundle.rglob("*"):  # browsers drop Unix modes; rsync keeps them
        if path.is_file():
            with path.open("rb") as handle:
                if handle.read(2) == b"#!":
                    path.chmod(0o755)
    progress("installing", 0, 1)
    models.mkdir(parents=True, exist_ok=True)
    previous = scratch / "previous"
    if replaced:
        os.replace(destination, previous)
    os.replace(bundle, destination)
    shutil.rmtree(previous, ignore_errors=True)
    progress("installing", 1, 1)
    warnings = [] if (destination / "robot_client/start.sh").is_file() else [
        "no robot_client/start.sh: benchmarks work, robot runs need the robot client"]
    return {"bundle": bundle_id, "policy": manifest["policy"], "replaced": replaced,
            "kept_calibration": kept, "warnings": warnings, "graphs": len(manifest["graphs"])}


def validate_calibration(value: Any, existing: dict[str, Any] | None) -> dict[str, Any]:
    """A LeRobot motor calibration: motor name -> integer register values."""
    if not isinstance(value, dict) or not value:
        raise ValueError("calibration must be a JSON object of motors")
    for motor, fields in value.items():
        if not isinstance(fields, dict) or any(
            type(fields.get(name)) is not int for name in CALIBRATION_FIELDS
        ):
            raise ValueError(f"{motor}: needs integer {', '.join(CALIBRATION_FIELDS)}")
    if existing and set(existing) != set(value):
        raise ValueError("motor names differ from this bundle's calibration (another robot?)")
    return value


def download_roots(root: Path) -> dict[str, Path]:
    """Read-only folders the browser may list and download from.

    POLIMA_STUDIO_DOWNLOADS adds more, colon-separated, as name=path or path.
    """
    roots = {"logs": root / "var/log", "calibration-backups": root / "var/lib/calibration-backups",
             "models": root / "models"}
    for item in filter(None, os.environ.get("POLIMA_STUDIO_DOWNLOADS", "").split(":")):
        name, _, path = item.rpartition("=")
        roots[name or Path(path).name] = Path(path).expanduser()
    return roots


def list_directory(base: Path, relative: str = "") -> list[dict[str, Any]]:
    entries = []
    for path in confined(base, relative).iterdir():
        try:
            path.resolve().relative_to(base.resolve())
            stat = path.stat()
        except (OSError, ValueError):
            continue  # unreadable, or a symlink leading outside the root
        entries.append({"name": path.name, "dir": path.is_dir(), "size": stat.st_size,
                        "modified": stat.st_mtime})
    return sorted(entries, key=lambda e: (not e["dir"], -e["modified"]))


def stream_tar(directory: Path) -> Iterator[bytes]:
    """A folder as an uncompressed tar, streamed without a temporary file.

    Symlinks are skipped, so a link inside a download root cannot pull files
    from outside it. A file that grows while it is read (a live log) is cut at
    the size its header announced.
    """
    for current, dirs, files in os.walk(directory):
        dirs.sort()
        for path in sorted(Path(current) / name for name in files):
            if path.is_symlink() or not path.is_file():
                continue
            info = tarfile.TarInfo(str(directory.name / path.relative_to(directory)))
            stat = path.stat()
            info.size, info.mtime, info.mode = stat.st_size, stat.st_mtime, stat.st_mode & 0o755
            yield info.tobuf(tarfile.PAX_FORMAT)
            remaining = info.size
            with path.open("rb") as handle:
                while remaining and (block := handle.read(min(_IO, remaining))):
                    remaining -= len(block)
                    yield block
            yield b"\0" * (remaining + (-info.size % 512))
    yield b"\0" * 1024
