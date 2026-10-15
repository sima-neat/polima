"""Explode a SiMa model pack (*_mpk.tar.gz) into the direct model layout.

  * An archive with only an ELF is allowed: PoLiMa's direct-MLA runtime needs
    only the ELF, and ACT ships with no etc/ at all. `require_json` requires an
    etc/*.json as well, for the NEAT ModelPack runtime.
  * `rewrite_paths` is idempotent and reports what it changed, so a re-run on
    the DevKit is verifiable.

Path traversal is refused: absolute member names and `..` components are
skipped.
"""

from __future__ import annotations

import tarfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from polima.util.logging import get

log = get("bundle.unpack")

SUBDIRS = ("etc", "lib", "share")

#: file extension -> destination subdirectory
ROUTING = {
    ".elf": "share",
    ".so": "lib",
    ".json": "etc",
    ".yaml": "etc",
    ".yml": "etc",
}


@dataclass
class UnpackResult:
    root: Path
    elfs: list[Path] = field(default_factory=list)
    configs: list[Path] = field(default_factory=list)
    libs: list[Path] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.elfs)


def from_mpk(
    archive: str | Path,
    dest: str | Path,
    *,
    require_json: bool = False,
    rewrite: bool = True,
) -> UnpackResult:
    """Extract `archive` into `dest`/{etc,lib,share}."""
    archive = Path(archive)
    root = Path(dest).resolve()
    directories = {name: root / name for name in SUBDIRS}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    result = UnpackResult(root=root)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            source = PurePosixPath(member.name)
            if not member.isfile() or source.is_absolute() or ".." in source.parts:
                result.skipped.append(member.name)
                continue
            target_dir = directories.get(ROUTING.get(source.suffix.lower(), ""))
            if target_dir is None:
                result.skipped.append(member.name)
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            written = target_dir / source.name
            written.write_bytes(extracted.read())
            {"share": result.elfs, "lib": result.libs, "etc": result.configs}[
                target_dir.name
            ].append(written)

    if rewrite:
        rewrite_paths(root)

    if not result.elfs:
        raise RuntimeError(f"{archive} contained no .elf -- not a usable model pack")
    if require_json and not result.configs:
        raise RuntimeError(f"{archive} contained no etc/*.json (NEAT runtime needs one)")

    log.debug(
        "unpacked %s -> %s (%d elf, %d lib, %d cfg)",
        archive.name, root, len(result.elfs), len(result.libs), len(result.configs),
    )
    return result


def rewrite_paths(root: str | Path) -> list[str]:
    """Point each etc/*.json at this directory's absolute share/ and lib/ paths.

    The compiler bakes build-machine paths into `simaai__params.model_path` and
    `model_info.path`; they must be rewritten wherever the model actually lands.
    So this re-runs on the DevKit after the model is copied there.

    Returns the list of files changed, so a no-op re-run is visible.
    """
    root = Path(root).resolve()
    directories = {name: root / name for name in SUBDIRS}
    changed: list[str] = []

    from polima.util.jsonio import read_json

    for config_path in sorted(directories["etc"].glob("*.json")):
        try:
            config = read_json(config_path)
        except Exception:  # noqa: BLE001 - a malformed sidecar must not abort deploy
            log.warning("skipping unparsable %s", config_path)
            continue

        before = _dumps(config)
        params = config.get("simaai__params")
        if isinstance(params, dict) and isinstance(params.get("model_path"), str):
            params["model_path"] = str(directories["share"] / Path(params["model_path"]).name)
        info = config.get("model_info")
        if isinstance(info, dict) and isinstance(info.get("path"), str):
            info["path"] = str(directories["lib"] / Path(info["path"]).name)

        after = _dumps(config)
        if after != before:
            # Written only on change and in one stable format (indent=4), so a
            # re-run produces no spurious diff.
            config_path.write_text(after, encoding="utf-8")
            changed.append(config_path.name)
    return changed


def _dumps(config: dict) -> str:
    import json

    return json.dumps(config, indent=4) + "\n"


def find_mpk(build_dir: str | Path, graph: str) -> Path | None:
    """Locate `<graph>_mpk.tar.gz` anywhere under a compiler build tree."""
    build_dir = Path(build_dir)
    matches = sorted(build_dir.rglob(f"{graph}_mpk.tar.gz"))
    return matches[0] if matches else None


def mpk_has_elf(archive: str | Path) -> bool:
    """Cheap validity probe, used to decide whether a build can resume."""
    try:
        with tarfile.open(archive, "r:gz") as tar:
            return any(
                member.isfile() and member.name.lower().endswith(".elf") for member in tar
            )
    except (tarfile.TarError, OSError):
        return False
