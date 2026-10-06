"""Board-side layout, preflight and safety guards."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field

from polima.bundle.layout import Bundle
from polima.config.base import BoardConfig
from polima.deploy.ssh import BoardError, BoardSession
from polima.util.logging import get

log = get("deploy.board")

#: Directories every deploy expects to exist under BoardConfig.root.
LAYOUT = ("bin", "src", "build", "robot", "var/log", "var/run", "etc")


@dataclass
class Preflight:
    ok: bool = True
    machine: str = ""
    cores: int = 0
    free_bytes: int = 0
    required_bytes: int = 0
    cmake: str = ""
    simalmm: str = ""
    problems: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.ok = False
        self.problems.append(message)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok, "machine": self.machine, "cores": self.cores,
            "free_bytes": self.free_bytes, "required_bytes": self.required_bytes,
            "cmake": self.cmake, "simalmm": self.simalmm, "problems": self.problems,
        }


def preflight(session: BoardSession, board: BoardConfig, bundle: Bundle | None) -> Preflight:
    """Check the board can take this deploy, before anything is transferred."""
    report = Preflight()

    facts = _probe(session, board)
    report.machine = facts.get("ARCH", "")
    report.cores = int(facts.get("CORES") or 0)
    report.cmake = facts.get("CMAKE", "")
    report.simalmm = facts.get("SIMALMM", "")

    if report.machine != "aarch64":
        report.fail(f"expected an aarch64 board, found {report.machine or 'unknown'}")
    if not report.cmake:
        report.fail("cmake is not installed; the native runtime is built on the board")
    if not report.simalmm:
        report.fail(
            "no SimaLMM cmake package (/usr/lib/*/cmake/SimaLMM); "
            "find_package(SimaLMM REQUIRED) will fail"
        )
    if not facts.get("JSONCPP"):
        report.fail(
            "jsoncpp headers missing; SimaLMM's mla_model.hpp includes <json/json.h>"
        )

    # Measure at the deploy root itself (or its nearest existing parent), not at
    # the first path component: /media is the root filesystem, /media/nvme is the
    # 458 GB disk the bundles actually land on.
    report.free_bytes = session.free_bytes(board.root)
    if bundle is not None:
        report.required_bytes = int(bundle.total_elf_bytes * board.free_space_factor)
        if report.free_bytes and report.free_bytes < report.required_bytes:
            report.fail(
                f"{board.root} has {report.free_bytes / 1e9:.1f} GB free but this deploy "
                f"needs {report.required_bytes / 1e9:.1f} GB "
                f"({board.free_space_factor}x the bundle)"
            )
    return report


def _probe(session: BoardSession, board: BoardConfig) -> dict[str, str]:
    command = (
        "echo ARCH=$(uname -m); "
        "echo CORES=$(nproc); "
        "echo CMAKE=$(cmake --version 2>/dev/null | head -1 | awk '{print $3}'); "
        "echo SIMALMM=$(ls -d /usr/lib/*/cmake/SimaLMM 2>/dev/null | head -1); "
        "echo JSONCPP=$(ls /usr/include/jsoncpp/json/json.h 2>/dev/null)"
    )
    output = session.capture(command)
    return dict(
        line.split("=", 1) for line in output.splitlines() if "=" in line
    )


def ensure_layout(session: BoardSession, board: BoardConfig) -> None:
    session.mkdirs(*(board.path(part) for part in LAYOUT), board.bundles_dir)


def assert_no_archives(session: BoardSession, remote_dir: str) -> None:
    """Refuse to leave a .tar.gz on the SoM.

    Both legacy deploy scripts carry this guard. The direct-MLA runtime never
    opens an archive, so one arriving means something shipped the compiler's
    output instead of the extracted ELF -- which would silently waste gigabytes
    and, worse, suggest the wrong artifact was selected.
    """
    found = session.capture(
        f"find {shlex.quote(remote_dir)} -name '*.tar.gz' -print 2>/dev/null | head -5"
    )
    if found:
        raise BoardError(
            "archives reached the board, which the direct-MLA runtime cannot use:\n  "
            + "\n  ".join(found.splitlines())
        )


def verify_bundle(session: BoardSession, board: BoardConfig, bundle: Bundle) -> list[str]:
    """Confirm every ELF arrived intact, by hashing on the board.

    A truncated rsync produces an ELF that loads and returns garbage; comparing
    digests is the only way to rule that out.
    """
    remote_root = f"{board.bundles_dir}/{bundle.bundle_id}"
    problems: list[str] = []
    commands = " ; ".join(
        f"sha256sum {shlex.quote(remote_root + '/' + artifact.elf)} 2>/dev/null || "
        f"echo 'MISSING {artifact.elf}'"
        for artifact in bundle.graphs
    )
    output = session.capture(commands)

    digests: dict[str, str] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] != "MISSING":
            digests[parts[1].rsplit("/", 1)[-1]] = parts[0]

    for artifact in bundle.graphs:
        name = artifact.elf.rsplit("/", 1)[-1]
        actual = digests.get(name)
        if actual is None:
            problems.append(f"{artifact.name}: ELF missing on the board")
        elif actual != artifact.sha256:
            problems.append(
                f"{artifact.name}: sha256 {actual[:12]} on board, "
                f"expected {artifact.sha256[:12]} -- transfer corrupted"
            )
    return problems


def activate(session: BoardSession, board: BoardConfig, bundle_id: str) -> None:
    """Point `current` at a bundle. Atomic, so rollback needs no file transfer."""
    target = f"{board.bundles_dir}/{bundle_id}"
    session.run(f"ln -sfn {shlex.quote(target)} {shlex.quote(board.current_link)}")
    log.info("activated %s", bundle_id)


def current_bundle(session: BoardSession, board: BoardConfig) -> str | None:
    """The activated bundle id, or None.

    `readlink -f` echoes the path back when the link does not exist, so read the
    link itself and require that it actually is one -- otherwise a board with no
    deploy reports its current bundle as "current".
    """
    resolved = session.capture(
        f"test -L {shlex.quote(board.current_link)} && "
        f"readlink {shlex.quote(board.current_link)} 2>/dev/null"
    )
    return resolved.rsplit("/", 1)[-1] if resolved else None


def list_bundles(session: BoardSession, board: BoardConfig) -> list[str]:
    output = session.capture(f"ls -1 {shlex.quote(board.bundles_dir)} 2>/dev/null")
    return [line for line in output.splitlines() if line.strip()]


def bundle_identity(
    session: BoardSession, board: BoardConfig, bundle_id: str
) -> tuple[str, str]:
    """Return the policy and robot recorded by one deployed bundle.

    ``polima deploy --list`` used to call every managed bundle's *kind*
    ``polima``. That described its storage format, but operators reasonably
    read the column as the policy kind. Read the two small manifests so the
    list distinguishes ACT/SmolVLA/GR00T and SO-101/MyCobot explicitly.

    Older bundles may lack one of these fields, so retain conservative name
    fallbacks rather than making a listing command fail on historical data.
    """
    root = f"{board.bundles_dir}/{bundle_id}"

    def load(remote_path: str) -> dict:
        try:
            value = json.loads(
                session.capture(f"cat {shlex.quote(remote_path)} 2>/dev/null") or "{}"
            )
        except (json.JSONDecodeError, TypeError):
            return {}
        return value if isinstance(value, dict) else {}

    manifest = load(f"{root}/bundle.json")
    plan = load(f"{root}/plan.json")
    policy = str(manifest.get("policy") or "").strip().lower()
    robot_block = plan.get("robot") if isinstance(plan.get("robot"), dict) else {}
    profile = plan.get("profile") if isinstance(plan.get("profile"), dict) else {}
    robot = str(
        robot_block.get("calibration_id") or profile.get("calibration_id") or ""
    ).strip()

    lowered = bundle_id.lower()
    if not policy:
        policy = next(
            (name for name in ("act", "smolvla", "groot") if lowered.startswith(name)),
            "-",
        )
    if not robot:
        robot = "mycobot280" if "mycobot" in lowered else "-"
    return policy, robot


def managed_bundles(session: BoardSession, board: BoardConfig) -> set[str]:
    """Of the trees in the model store, the ones PoLiMa built.

    The store also holds hand-built deploy roots (e.g. ACT_<task>_100000).
    Those are perfectly valid to serve, but
    `--activate` and the content-hash guarantees only apply to trees carrying a
    bundle.json, so the listing says which is which rather than implying every
    directory is interchangeable.
    """
    found = session.capture(
        f"find {shlex.quote(board.bundles_dir)} -maxdepth 2 -name bundle.json 2>/dev/null"
    )
    prefix = board.bundles_dir.rstrip("/") + "/"
    return {
        line[len(prefix):].split("/")[0]
        for line in found.splitlines()
        if line.startswith(prefix)
    }
