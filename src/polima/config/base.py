"""Core configuration objects.

One typed place for the DevKit address, ports, paths and runtime settings.

The DevKit address has no default: every network is different, so it comes from
`--board`, `$POLIMA_BOARD` or `board.host` in `.polima.yaml` (see
`.polima.yaml.example`). The other defaults match a stock Modalix DevKit:
aarch64, 16 cores, the NVMe drive mounted at /media/nvme.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# No default DevKit: set it with --devkit, $POLIMA_DEVKIT or .polima.yaml.
DEFAULT_BOARD_HOST = ""
NO_BOARD_MESSAGE = (
    "no DevKit address: pass --devkit user@host, set POLIMA_DEVKIT, "
    "or set board.host in .polima.yaml (see .polima.yaml.example)"
)
DEFAULT_BOARD_ROOT = "/media/nvme/polima"


@dataclass
class BoardConfig:
    """A Modalix DevKit target."""

    host: str = DEFAULT_BOARD_HOST
    root: str = DEFAULT_BOARD_ROOT
    port: int | None = None                 # None -> PolicySpec.wire.default_port
    # accept-new: a new DevKit is trusted on first contact; a changed identity is refused.
    ssh_options: tuple[str, ...] = ("-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new")
    control_master: bool = True             # one multiplexed connection per deploy
    # The DevKit has 16 cores.
    build_jobs: int = 8
    # Refuse to deploy unless free space is this multiple of the bundle size.
    free_space_factor: float = 1.5
    health_timeout_s: float = 30.0
    #: SIGTERM grace before `stop` escalates to SIGKILL. Generous on purpose:
    #: a killed server never releases its DMA-coherent buffers, so every
    #: escalation fragments the MLA's CMA pool a little more, and a large
    #: bundle unloading several models does not finish in 10s.
    stop_timeout_s: float = 30.0
    connect_timeout_s: float = 10.0
    #: Directory under `root` holding deployed model trees. `models` rather than
    #: `bundles` so PoLiMa bundles sit alongside any hand-built model trees
    #: instead of creating a second place to look. Bundle names stay
    #: content-addressed, so they cannot collide with hand-named trees.
    bundles_subdir: str = "models"

    @property
    def user(self) -> str:
        return self.host.split("@")[0] if "@" in self.host else ""

    @property
    def address(self) -> str:
        return self.host.split("@")[-1]

    def path(self, *parts: str) -> str:
        return "/".join([self.root.rstrip("/"), *(p.strip("/") for p in parts)])

    @property
    def bundles_dir(self) -> str:
        return self.path(self.bundles_subdir)

    @property
    def current_link(self) -> str:
        return self.path("current")

    @property
    def native_src_dir(self) -> str:
        return self.path("src", "runtime")

    @property
    def python_src_dir(self) -> str:
        return self.path("python", "polima")

    @property
    def bin_dir(self) -> str:
        return self.path("bin")

    @property
    def venv_dir(self) -> str:
        return self.path("venv")

    @property
    def log_dir(self) -> str:
        return self.path("var", "log")

    @property
    def run_dir(self) -> str:
        return self.path("var", "run")


@dataclass
class PathsConfig:
    """Host-side locations. The training host keeps its existing layout --
    /ml_datasets and the conda envs are NOT relocated to /media/nvme; that
    applies to the DevKit only."""

    dataset_parent: Path = Path("/ml_datasets")
    outputs: Path | None = None             # None -> polima/outputs
    build_root: Path | None = None          # None -> <outputs>/build
    # $HOME/sima-sdk-extensions/model-compiler/bin -- overridable by
    # $MODEL_COMPILER_BIN.
    model_compiler_bin: Path | None = None
    lerobot_dir: Path | None = None         # host clone used for training
    # GR00T compile (`polima compile --policy groot`): each model family's export
    # interpreter and its Isaac-GR00T checkout. $POLIMA_GROOT16_PYTHON etc.
    groot16_python: Path | None = None
    groot16_source: Path | None = None
    groot17_python: Path | None = None
    groot17_source: Path | None = None
    groot17_backbone: str = ""             # Hub repository id or local directory


@dataclass
class SiteConfig:
    """Everything not owned by a single policy."""

    board: BoardConfig = field(default_factory=BoardConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    conda_base: Path | None = None
    offline: bool = False                   # sets the HF_*_OFFLINE trio
    dry_run: bool = False
    log_level: str = "INFO"

    def board_port(self, fallback: int) -> int:
        return self.board.port if self.board.port is not None else fallback
