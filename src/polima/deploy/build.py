"""Build the native runtime on the DevKit, skipping when nothing changed.

There is one binary for every policy, so a policy change does not touch the
C++ at all, and the build tree is keyed by a content hash of runtime/, so an
unchanged source tree skips configure and compile entirely.
"""

from __future__ import annotations

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

    def to_dict(self) -> dict:
        return {
            "source_hash": self.source_hash, "build_dir": self.build_dir,
            "skipped": self.skipped, "binaries": self.binaries,
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


def build_native(
    session: BoardSession,
    board: BoardConfig,
    *,
    force: bool = False,
    jobs: int | None = None,
) -> BuildResult:
    """Sync runtime/ to the DevKit and build it if the source changed."""
    digest = source_hash()
    build_dir = board.path("build", digest)
    sentinel = f"{build_dir}/.ok"
    jobs = jobs or board.build_jobs

    if not force and session.exists(sentinel):
        log.info("build: skipped (source hash %s unchanged)", digest)
        _link_binaries(session, board, build_dir)
        return BuildResult(digest, build_dir, True, list(BINARIES))

    log.info("build: source hash %s, compiling with -j%d", digest, jobs)
    session.mkdirs(board.native_src_dir, build_dir)
    session.rsync(native_dir(), board.native_src_dir, delete=True, echo=False)

    session.run(
        f"cmake -S {shlex.quote(board.native_src_dir)} -B {shlex.quote(build_dir)} "
        f"-DCMAKE_BUILD_TYPE=Release",
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
    return BuildResult(digest, build_dir, False, list(BINARIES))


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
