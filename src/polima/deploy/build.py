"""Build the native runtime on the DevKit from source, skipping when nothing changed.

There is one binary for every policy, so a policy change does not touch the
C++ at all. Reuse is keyed by both the runtime sources and the installed
SDK/LLiMa packages, so an SDK or LLiMa upgrade cannot reuse an old binary.
A DevKit installed from a PoLiMa package keeps its prebuilt runtime.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import dataclass

from polima.config.base import BoardConfig
from polima.deploy.ssh import BoardSession
from polima.util.hashing import sha256_tree, short
from polima.util.logging import get
from polima.util.paths import native_dir

log = get("deploy.build")

BINARIES = ("polima_server", "polima_cli")


@dataclass
class BuildResult:
    source_hash: str
    build_dir: str
    skipped: bool
    binaries: list[str]
    toolchain_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "source_hash": self.source_hash, "build_dir": self.build_dir,
            "skipped": self.skipped, "binaries": self.binaries,
            "toolchain_hash": self.toolchain_hash,
        }


def source_hash(source_dir=None) -> str:
    """Content hash of runtime/ -- the build-skip key."""
    return short(
        sha256_tree(
            source_dir or native_dir(),
            patterns=("**/*.cpp", "**/*.hpp", "**/*.h", "**/CMakeLists.txt"),
        ),
        12,
    )


def build_key(source: str, toolchain: str) -> str:
    """Content key for sources compiled against one exact target toolchain."""
    return hashlib.sha256(f"{source}\0{toolchain}".encode()).hexdigest()[:16]


def toolchain_identity(session: BoardSession) -> str:
    """Fingerprint the platform and the matching LLiMa runtime and development packages.

    A version mismatch is an installation error: compiling against one package
    and loading another is unsafe.
    """
    result = session.run(
        r'''set -eu
core=$(dpkg-query -W -f='${Version}' sima-lmm-core 2>/dev/null) || exit 1
dev=$(dpkg-query -W -f='${Version}' sima-lmm-dev 2>/dev/null) || exit 1
[ "$core" = "$dev" ] || exit 2
printf 'platform\n'
sed -nE 's/^[[:space:]]*(MACHINE|DISTRO_VERSION|DISTRO_BUILD)[[:space:]]*=[[:space:]]*(.*)$/\1=\2/p' \
    /etc/buildinfo 2>/dev/null | LC_ALL=C sort
printf 'packages\nsima-lmm-core=%s\nsima-lmm-dev=%s\n' "$core" "$dev"
printf 'files\n'
for package in sima-lmm-core sima-lmm-dev; do
    cat /var/lib/dpkg/info/"$package".md5sums /var/lib/dpkg/info/"$package":*.md5sums 2>/dev/null || true
done
''',
        check=False,
    )
    if result.returncode == 2:
        raise RuntimeError(
            "LLiMa development and runtime packages differ; install one matching LLiMa artifact"
        )
    if result.returncode != 0:
        raise RuntimeError("LLiMa 0.6 (sima-lmm-core and sima-lmm-dev) is not installed on the DevKit")
    return result.stdout.strip()


def packaged_runtime(session: BoardSession, board: BoardConfig) -> bool:
    """Does the DevKit run a prebuilt runtime installed from a PoLiMa package?"""
    text = session.capture(f"cat {shlex.quote(board.path('runtime.json'))} 2>/dev/null || true")
    try:
        return json.loads(text).get("schema_version") == 2
    except ValueError:
        return False


def toolchain_value(identity: str, name: str) -> str:
    """Read one ``name=value`` line from a captured toolchain identity."""
    for line in identity.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == name:
            return value.strip()
    return ""


def build_native(
    session: BoardSession,
    board: BoardConfig,
    *,
    force: bool = False,
    jobs: int | None = None,
) -> BuildResult:
    """Sync runtime/ and build it for an explicit source deployment."""
    digest = source_hash()
    identity = toolchain_identity(session)
    platform_version = toolchain_value(identity, "DISTRO_VERSION")
    llima_version = toolchain_value(identity, "sima-lmm-core")
    toolchain_hash = short(hashlib.sha256(identity.encode()).hexdigest(), 12)
    key = build_key(digest, identity)
    build_dir = board.path("build", key)
    sentinel = f"{build_dir}/.ok"
    jobs = jobs or board.build_jobs

    if not force and session.exists(sentinel):
        log.info("build: skipped (source and LLiMa toolchain unchanged: %s)", key)
        _link_binaries(session, board, build_dir)
        return BuildResult(digest, build_dir, True, list(BINARIES), toolchain_hash)

    log.info("build: source hash %s, compiling with -j%d", digest, jobs)
    session.mkdirs(board.native_src_dir, build_dir)
    session.rsync(native_dir(), board.native_src_dir, delete=True, echo=False)

    session.run(
        f"cmake -S {shlex.quote(board.native_src_dir)} -B {shlex.quote(build_dir)} "
        f"-DCMAKE_BUILD_TYPE=Release -DPOLIMA_SOURCE_COMMIT={shlex.quote(digest)}"
        f" -DPOLIMA_PLATFORM_VERSION={shlex.quote(platform_version)}"
        f" -DPOLIMA_LLIMA_PACKAGE_VERSION={shlex.quote(llima_version)}",
        echo=False,
    )
    session.run(f"cmake --build {shlex.quote(build_dir)} -j{jobs}", echo=True)

    missing = [
        name for name in BINARIES
        if not session.exists(f"{build_dir}/{name}")
    ]
    if missing:
        raise RuntimeError(f"build finished but produced no {', '.join(missing)}")

    session.run(f"touch {shlex.quote(sentinel)}")
    _link_binaries(session, board, build_dir)
    return BuildResult(digest, build_dir, False, list(BINARIES), toolchain_hash)


def _link_binaries(session: BoardSession, board: BoardConfig, build_dir: str) -> None:
    """Atomically install the one active native binary pair under ``bin/``."""
    for name in BINARIES:
        session.run(
            f"install -m 0755 {shlex.quote(build_dir + '/' + name)} "
            f"{shlex.quote(board.path('bin', name + '.new'))} && "
            f"mv -f {shlex.quote(board.path('bin', name + '.new'))} "
            f"{shlex.quote(board.path('bin', name))}"
        )


#: Alias -> binary, installed onto PATH. Dashes match the console scripts
#: (`polima run`, `polima doctor`); the binaries keep their underscored names.
PATH_ALIASES = {
    "polima": "polima_cli",
    "polima-server": "polima_server",
    "polima-cli": "polima_cli",
}

#: Where LLiMa lives on the DevKit (`/usr/bin/llima`), so this is the convention
#: the devkit already follows. /usr/local/bin is preferred because it is ahead
#: of /usr/bin on PATH and is not owned by a package.
PATH_DIRS = ("/usr/local/bin", "/usr/bin")


def link_onto_path(session: BoardSession, board: BoardConfig) -> dict[str, str]:
    """Put `polima-server` / `polima-cli` on PATH, the way `llima` already is.

    Without this the binaries sit in `/media/nvme/polima/bin`, which is on no
    PATH, so every invocation needs the full path. LLiMa installs `/usr/bin/llima`
    and is simply `llima` from any shell; matching that is the whole point.

    The links point at the atomically installed pair under `bin/`. CMake's
    content-addressed directories are build caches, not additional installed
    runtimes; callers and PATH never depend on their names.

    Best-effort by design. A convenience alias must never fail a deploy, so a
    directory that needs root is tried with `sudo -n` and skipped if that would
    prompt. The caller reports what was linked and what was not.
    """
    linked: dict[str, str] = {}
    for directory in PATH_DIRS:
        if session.run(f"test -d {shlex.quote(directory)}", check=False).returncode != 0:
            continue
        writable = session.run(f"test -w {shlex.quote(directory)}", check=False).returncode == 0
        prefix = "" if writable else "sudo -n "
        if not writable and session.run("sudo -n true", check=False).returncode != 0:
            continue

        for alias, binary in PATH_ALIASES.items():
            target = board.path("bin", binary)
            result = session.run(
                f"{prefix}ln -sfn {shlex.quote(target)} "
                f"{shlex.quote(directory + '/' + alias)}",
                check=False,
            )
            if result.returncode == 0:
                linked[alias] = f"{directory}/{alias}"
        if linked:
            break
    return linked


def prune_builds(session: BoardSession, board: BoardConfig, keep: int = 2) -> list[str]:
    """Drop old build trees, newest `keep` retained. Never touches `bin` links."""
    listing = session.capture(
        f"ls -1t {shlex.quote(board.path('build'))} 2>/dev/null"
    ).splitlines()
    stale = [name for name in listing[keep:] if name.strip()]
    for name in stale:
        session.run(f"rm -rf {shlex.quote(board.path('build', name))}", check=False)
    if stale:
        log.info("pruned %d old build tree(s)", len(stale))
    return stale
