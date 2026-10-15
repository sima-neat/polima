"""Find and unpack SiMa model packs (`*_mpk.tar.gz`).

An mpk is the compiler's shippable artifact: a gzipped tar holding the MLA ELF,
its `.so` helpers, and json config that points at both by whatever path the
compile host happened to use. Unpacking is therefore not just extraction -- the
json has to be rewritten to the new location or the runtime looks for the model
where the build machine kept it.

The layout produced (`etc/`, `lib/`, `share/`) is what the on-device loader
expects, and matches `models_uncompressed/<graph>/` in a compiler build tree.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path, PurePosixPath

#: Where each file type lands, keyed by suffix.
_DESTINATIONS = {
    ".elf": "share",
    ".so": "lib",
    ".json": "etc",
    ".yaml": "etc",
    ".yml": "etc",
}


def find(root: str | Path, stem: str) -> Path | None:
    """Locate `<stem>_mpk.tar.gz` anywhere under `root`."""
    matches = sorted(Path(root).glob(f"**/{stem}_mpk.tar.gz"))
    return matches[0] if matches else None


def has_elf(archive: str | Path) -> bool:
    """Whether an mpk actually contains an ELF.

    A compile can exit 0 and still produce an mpk with no ELF when the graph
    fell back to the APU. A compile is accepted only after this check, because
    the failure is otherwise invisible until the DevKit cannot load the model.
    """
    try:
        with tarfile.open(archive, "r:gz") as handle:
            return any(m.isfile() and m.name.endswith(".elf") for m in handle.getmembers())
    except (tarfile.TarError, OSError):
        return False


def rewrite_paths(root: str | Path) -> None:
    """Repoint the extracted json config at this directory."""
    root = Path(root).resolve()
    for config_path in (root / "etc").glob("*.json"):
        config = json.loads(config_path.read_text())
        params = config.get("simaai__params")
        if isinstance(params, dict) and isinstance(params.get("model_path"), str):
            params["model_path"] = str(root / "share" / Path(params["model_path"]).name)
        info = config.get("model_info")
        if isinstance(info, dict) and isinstance(info.get("path"), str):
            info["path"] = str(root / "lib" / Path(info["path"]).name)
        config_path.write_text(json.dumps(config, indent=4) + "\n")


def unpack(archive: str | Path, destination: str | Path) -> Path:
    """Extract an mpk into `etc/`, `lib/`, `share/` and fix up its config."""
    archive = Path(archive)
    root = Path(destination).resolve()
    for name in ("etc", "lib", "share"):
        (root / name).mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            source = PurePosixPath(member.name)
            # Refuse absolute paths and parent traversal: an mpk is an archive
            # from a build host, and extracting one should not be able to write
            # outside the destination.
            if not member.isfile() or source.is_absolute() or ".." in source.parts:
                continue
            target = _DESTINATIONS.get(source.suffix.lower())
            if target is None:
                continue
            extracted = handle.extractfile(member)
            if extracted is not None:
                (root / target / source.name).write_bytes(extracted.read())

    rewrite_paths(root)
    if not any((root / "etc").glob("*.json")) or not any((root / "share").glob("*.elf")):
        raise RuntimeError(
            f"{archive} did not yield a usable model directory "
            f"(need etc/*.json and share/*.elf under {root})"
        )
    return root
