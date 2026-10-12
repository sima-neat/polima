"""Deploy a bundle to a Modalix DevKit.

Every stage is individually skippable and individually recorded, so a failed
deploy says which step failed rather than leaving the DevKit half-updated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from polima.bundle.layout import Bundle
from polima.config.base import BoardConfig
from polima.deploy import board as board_ops
from polima.deploy.build import BuildResult, build_native, link_onto_path
from polima.deploy.service import ServiceStatus, start
from polima.deploy.ssh import BoardError, BoardSession
from polima.util.logging import get
from polima.util.paths import package_root

log = get("deploy")

STAGES = (
    "preflight", "layout", "sync-bundle", "python-runtime", "build", "verify", "guard",
    "activate", "service", "smoke",
)


@dataclass
class DeployReport:
    bundle_id: str
    host: str
    port: int
    stages: dict = field(default_factory=dict)
    build: BuildResult | None = None
    service: ServiceStatus | None = None
    started_at: float = 0.0
    duration_s: float = 0.0
    ok: bool = True

    def record(self, stage: str, status: str, **detail) -> None:
        self.stages[stage] = {"status": status, **detail}
        if status == "failed":
            self.ok = False

    def to_dict(self) -> dict:
        return {
            "ok": self.ok, "bundle_id": self.bundle_id, "host": self.host,
            "port": self.port, "stages": self.stages,
            "build": self.build.to_dict() if self.build else None,
            "service": self.service.to_dict() if self.service else None,
            "duration_s": round(self.duration_s, 2),
        }


def _metadata_matches(session, bundle, remote_bundle: str) -> bool:
    """Whether the board's copy of the non-ELF files is current.

    ELF identity is guaranteed by the bundle id; everything else is not.
    """
    import hashlib

    robot_sidecars = tuple(
        name
        for name in bundle.sidecars
        if name.startswith("robot_client/") and ".." not in Path(name).parts
    )
    for name in ("plan.json", "bundle.json", *robot_sidecars):
        local = bundle.root / name
        if not local.is_file():
            continue
        expected = hashlib.sha256(local.read_bytes()).hexdigest()
        actual = session.capture(
            f"sha256sum {remote_bundle}/{name} 2>/dev/null | cut -d' ' -f1"
        ).strip()
        if actual != expected:
            log.info("%s differs from the board's copy; re-syncing", name)
            return False
    return True


def deploy(
    bundle: Bundle,
    board_config: BoardConfig,
    *,
    port: int | None = None,
    build: bool = True,
    start_service: bool = False,
    activate: bool = True,
    force: bool = False,
    verbose_server: bool = False,
    dry_run: bool = False,
) -> DeployReport:
    """Push `bundle` to the board and (optionally) serve it."""
    resolved_port = port or board_config.port or 0
    report = DeployReport(
        bundle_id=bundle.bundle_id, host=board_config.host, port=resolved_port,
        started_at=time.time(),
    )
    remote_bundle = f"{board_config.bundles_dir}/{bundle.bundle_id}"

    with BoardSession(board_config, dry_run=dry_run) as session:
        # --- preflight: fail before transferring 100+ MiB ------------------
        checks = board_ops.preflight(session, board_config, bundle)
        report.record("preflight", "ok" if checks.ok else "failed", **checks.to_dict())
        if not checks.ok:
            raise BoardError(
                "board preflight failed:\n  " + "\n  ".join(checks.problems)
            )
        log.info(
            "board %s: %s, %d cores, %.0f GB free",
            board_config.host, checks.machine, checks.cores, checks.free_bytes / 1e9,
        )

        board_ops.ensure_layout(session, board_config)
        report.record("layout", "ok")

        # --- sync: content-addressed on the ELFs, so an identical bundle is
        # a no-op. But the id covers ELFs ONLY: plan.json, the wire description
        # and the fixtures can all change without moving it, so skipping on
        # "the directory exists" would leave an edited plan behind. rsync is
        # cheap when the ELFs match, so compare the small files and let rsync
        # decide about the big ones.
        already = session.exists(f"{remote_bundle}/bundle.json") and _metadata_matches(
            session, bundle, remote_bundle
        )
        if already and not force:
            log.info("bundle %s already on the board; skipping transfer", bundle.bundle_id)
            report.record("sync-bundle", "skipped", reason="already present")
        else:
            log.info("syncing %.1f MiB to %s", bundle.total_elf_bytes / 1048576, remote_bundle)
            session.mkdirs(remote_bundle)
            session.rsync(
                bundle.root, remote_bundle, delete=True,
                excludes=("*.tar.gz", "*.onnx", "__pycache__"), echo=False,
            )
            report.record("sync-bundle", "ok", bytes=bundle.total_elf_bytes)

        launcher = bundle.root / "robot_client/start.sh"
        if launcher.is_file() and "polima.robot.so101" in launcher.read_text():
            log.info("syncing robot Python runtime to %s", board_config.python_src_dir)
            session.mkdirs(board_config.python_src_dir)
            session.rsync(
                package_root() / "src/polima", board_config.python_src_dir,
                excludes=("__pycache__", "*.pyc"), echo=False,
            )
            report.record("python-runtime", "ok")
        else:
            report.record("python-runtime", "skipped", reason="bundle uses standalone client")

        # --- build: skipped when runtime/ is unchanged ----------------------
        if build:
            result = build_native(session, board_config, force=force)
            report.build = result
            report.record(
                "build", "skipped" if result.skipped else "ok", source_hash=result.source_hash
            )
            # Put the binaries on PATH, the way /usr/bin/llima already is. Best
            # effort: an alias is a convenience and must never fail a deploy.
            aliases = link_onto_path(session, board_config)
            report.record(
                "link-bins", "ok" if aliases else "skipped",
                aliases=sorted(aliases),
                reason="" if aliases else "no writable PATH directory (try sudo)",
            )
        else:
            report.record("build", "skipped", reason="--no-build")

        # --- verify: hash every ELF on the board ---------------------------
        problems = board_ops.verify_bundle(session, board_config, bundle)
        report.record("verify", "ok" if not problems else "failed", problems=problems)
        if problems:
            raise BoardError("bundle verification failed:\n  " + "\n  ".join(problems))

        board_ops.assert_no_archives(session, remote_bundle)
        report.record("guard", "ok")

        if activate:
            board_ops.activate(session, board_config, bundle.bundle_id)
            report.record("activate", "ok", current=bundle.bundle_id)
        else:
            report.record("activate", "skipped")

        # --- service -------------------------------------------------------
        if start_service:
            if not resolved_port:
                raise BoardError("no port given and the board config declares none")
            status = start(
                session, board_config, port=resolved_port,
                bundle_path=board_config.current_link if activate else remote_bundle,
                verbose=verbose_server,
            )
            report.service = status
            report.record("service", "ok", pid=status.pid, port=resolved_port)
        else:
            report.record("service", "skipped", reason="not requested (use --start)")

    report.duration_s = time.time() - report.started_at
    return report
