#!/usr/bin/env bash
# Install the two native PoLiMa binaries and the LeRobot control environment on Modalix.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="${POLIMA_NATIVE_SOURCE:-$(cd -- "$SCRIPT_DIR/../runtime" && pwd)}"
POLIMA_SOURCE="${POLIMA_SOURCE:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
ROOT="${POLIMA_ROOT:-/media/nvme/polima}"
LEROBOT_VENV="${LEROBOT_VENV:-/media/nvme/lerobot}"
LEROBOT_INSTALLER="${LEROBOT_INSTALLER:-$POLIMA_SOURCE/examples/so101/setup/install_lerobot_modalix.sh}"
JOBS="${JOBS:-$(nproc)}"
STUDIO_USER="${POLIMA_STUDIO_USER:-$(id -un)}"
STUDIO_GROUP="$(id -gn "$STUDIO_USER")"

fail() { echo "[fail] $*" >&2; exit 1; }
log() { echo "==> $*"; }

[[ "$(uname -m)" == aarch64 ]] || fail "this installer must run on the Modalix devkit"
[[ -f "$SOURCE/CMakeLists.txt" ]] || fail "native source not found at $SOURCE"
for tool in cmake c++ python3 install; do
    command -v "$tool" >/dev/null 2>&1 || fail "$tool is required; install the DevKit SDK build prerequisites"
done
for path in "$ROOT" "$LEROBOT_VENV"; do
    [[ "$path" =~ ^/[a-zA-Z0-9_./-]+$ ]] || fail "installation paths must be absolute and contain only letters, digits, /, ., _ or -"
done
if command -v systemctl >/dev/null 2>&1 && [[ ! -w /etc/systemd/system ]]; then
    sudo -n true 2>/dev/null || fail "service installation requires passwordless sudo; configure installation privileges before continuing"
fi
mkdir -p "$ROOT/bin" "$ROOT/build" "$ROOT/var/log" "$ROOT/var/run"

# An interrupted LeRobot install leaves a venv without lerobot; finish it instead of reusing it.
if [[ ! -x "$LEROBOT_VENV/bin/python" ]] || ! "$LEROBOT_VENV/bin/python" -I -c 'import lerobot' >/dev/null 2>&1; then
    [[ "${SKIP_LEROBOT:-0}" != 1 ]] || fail "LeRobot is not installed in $LEROBOT_VENV and SKIP_LEROBOT=1"
    [[ -x "$LEROBOT_INSTALLER" ]] || fail "LeRobot installer not found at $LEROBOT_INSTALLER; initialize a complete PoLiMa checkout or set LEROBOT_INSTALLER"
    log "Installing the standalone LeRobot environment"
    VENV="$LEROBOT_VENV" "$LEROBOT_INSTALLER"
else
    log "Reusing LeRobot environment at $LEROBOT_VENV"
fi

if ! "$LEROBOT_VENV/bin/python" -c '
import importlib.metadata as metadata
import flask, grpc, waitress, google.protobuf.runtime_version
assert metadata.version("protobuf") == "6.31.1"
assert metadata.version("wandb") == "0.24.2"
assert metadata.version("grpcio") == "1.80.0"
' >/dev/null 2>&1; then
    log "Installing Studio dependencies and compatible protobuf and gRPC runtimes"
    "$LEROBOT_VENV/bin/python" -m pip install \
        flask waitress "wandb==0.24.2" "protobuf==6.31.1" "grpcio==1.80.0"
fi

log "Installing the PoLiMa DevKit package and Studio assets"
# setuptools reuses build/lib from an earlier install, which would carry deleted modules along.
rm -rf "$POLIMA_SOURCE"/build/lib "$POLIMA_SOURCE"/build/bdist.*
"$LEROBOT_VENV/bin/python" -m pip install --no-deps --upgrade "$POLIMA_SOURCE"

# Native commands include ROOT/python on their import path. Retire old source
# overlays so the CLI and Studio resolve the same installed package.
"$LEROBOT_VENV/bin/python" - "$ROOT" <<'PY_PACKAGE'
from pathlib import Path
import importlib.metadata
import sys
import time
root = Path(sys.argv[1])
installed = (Path(sys.argv[2]) if len(sys.argv) > 2 else
             Path(importlib.metadata.distribution("polima").locate_file("polima"))).resolve()
if not (installed / "__main__.py").is_file():
    raise RuntimeError(f"Installed PoLiMa package is missing: {installed}")
shadow = root / "python/polima"
shadow.parent.mkdir(parents=True, exist_ok=True)
if shadow.resolve() != installed:
    if shadow.exists() or shadow.is_symlink():
        backup = root / "var/backups" / f"python-polima-{time.time_ns()}"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shadow.rename(backup)
    shadow.symlink_to(installed, target_is_directory=True)
PY_PACKAGE

# Build in a staging tree, then atomically replace the one active binary pair.
# CMake intermediates are not an installed runtime and can be removed afterward.
STAGE="$ROOT/build/install-staging"
log "Building polima_server and polima_cli with $JOBS jobs"
cmake -E remove_directory "$STAGE"
cmake -S "$SOURCE" -B "$STAGE" -DCMAKE_BUILD_TYPE=Release
cmake --build "$STAGE" -j"$JOBS"
install -m 0755 "$STAGE/polima_server" "$ROOT/bin/polima_server.new"
install -m 0755 "$STAGE/polima_cli" "$ROOT/bin/polima_cli.new"
mv -f "$ROOT/bin/polima_server.new" "$ROOT/bin/polima_server"
mv -f "$ROOT/bin/polima_cli.new" "$ROOT/bin/polima_cli"

link_dir=""
for candidate in /usr/local/bin /usr/bin; do
    if [[ -d "$candidate" && -w "$candidate" ]]; then
        link_dir="$candidate"
        break
    fi
    if [[ -d "$candidate" ]] && sudo -n true 2>/dev/null; then
        sudo -n ln -sfn "$ROOT/bin/polima_cli" "$candidate/polima"
        sudo -n ln -sfn "$ROOT/bin/polima_cli" "$candidate/polima-cli"
        sudo -n ln -sfn "$ROOT/bin/polima_server" "$candidate/polima-server"
        link_dir="$candidate"
        break
    fi
done
if [[ -n "$link_dir" && -w "$link_dir" ]]; then
    ln -sfn "$ROOT/bin/polima_cli" "$link_dir/polima"
    ln -sfn "$ROOT/bin/polima_cli" "$link_dir/polima-cli"
    ln -sfn "$ROOT/bin/polima_server" "$link_dir/polima-server"
fi

if command -v systemctl >/dev/null 2>&1; then
    systemd_dir=/etc/systemd/system
    default_dir=/etc/default
    studio_service_was_installed=0
    [[ -e "$systemd_dir/polima-studio.service" ]] && studio_service_was_installed=1
    systemctl_cmd=(systemctl)
    install_cmd=(install)
    if [[ ! -w "$systemd_dir" ]]; then
        sudo -n true 2>/dev/null || fail "sudo is required to install polima-studio.service"
        systemctl_cmd=(sudo -n systemctl)
        install_cmd=(sudo -n install)
    fi
    log "Installing polima-studio.service"
    rendered_service="$(mktemp)"
    trap 'rm -f "$rendered_service"' EXIT
    python3 - "$POLIMA_SOURCE/scripts/systemd/polima-studio.service" "$rendered_service" \
        "$ROOT" "$LEROBOT_VENV" "$STUDIO_USER" "$STUDIO_GROUP" <<'PY_SERVICE'
from pathlib import Path
import sys
source, destination, root, venv, user, group = sys.argv[1:]
text = Path(source).read_text()
replacements = {
    "ExecStart=": f'ExecStart=/usr/bin/env POLIMA_ROOT={root} LEROBOT_VENV={venv} {venv}/bin/python -m polima.studio.main',
    "User=": f"User={user}", "Group=": f"Group={group}",
    "ReadWritePaths=": f"ReadWritePaths={root}",
}
Path(destination).write_text("\n".join(
    next((value for key, value in replacements.items() if line.startswith(key)), line)
    for line in text.splitlines()) + "\n")
PY_SERVICE
    "${install_cmd[@]}" -m 0644 "$rendered_service" "$systemd_dir/polima-studio.service"
    if [[ ! -e "$default_dir/polima-studio" ]]; then
        "${install_cmd[@]}" -m 0644 "$POLIMA_SOURCE/scripts/systemd/polima-studio.default" \
            "$default_dir/polima-studio"
    fi
    "${systemctl_cmd[@]}" daemon-reload
    if [[ "$studio_service_was_installed" == 0 ]]; then
        log "Leaving PoLiMa Studio disabled and stopped by default"
        "${systemctl_cmd[@]}" disable --now polima-studio.service
    else
        log "Preserving the existing PoLiMa Studio enable/running state"
    fi
fi

log "Verifying"
"$ROOT/bin/polima_cli" --help >/dev/null
"$ROOT/bin/polima_server" --help >/dev/null
"$LEROBOT_VENV/bin/python" -c 'import importlib.metadata as m; import lerobot, cv2, flask, waitress, torch, polima.studio; print("lerobot=%s torch=%s cv2=%s flask=%s waitress=%s" % (lerobot.__version__, torch.__version__, cv2.__version__, m.version("flask"), m.version("waitress")))'

cat <<EOF

Installed:
  $ROOT/bin/polima_cli
  $ROOT/bin/polima_server
  $LEROBOT_VENV

For custom paths, export POLIMA_ROOT=$ROOT and LEROBOT_VENV=$LEROBOT_VENV
in your terminal before using the native command.

Next:
  polima activate        # select and optionally start an installed bundle
  polima server          # interactive start/stop
  polima robot           # inspect hardware and confirm startup
  polima studio start    # start Studio for this boot
  polima studio enable   # optionally enable Studio at boot
  polima studio status   # print state and URL
EOF
