#!/usr/bin/env bash
# Provide a private LeRobot environment for DevKit qualification without
# touching the runner's shared installation.
#
# Usage: prepare_lerobot.sh <candidate-venv>
#
# When the runner has a working shared environment, it is copied to
# <candidate-venv>; otherwise a fresh environment is built there. Ephemeral
# DevKit runners are not guaranteed to carry the shared environment. Either
# way the caller owns <candidate-venv> and installs the candidate PoLiMa into it.
set -euo pipefail

dest="${1:?usage: prepare_lerobot.sh <candidate-venv>}"
shared="${POLIMA_CI_SHARED_LEROBOT:-/media/nvme/lerobot}"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

log() { echo "==> $*"; }

mkdir -p "$(dirname -- "$dest")"
if [[ -x "$shared/bin/python" ]] && "$shared/bin/python" -I -c 'import lerobot' >/dev/null 2>&1; then
    log "Copying the shared LeRobot environment at $shared"
    cp -a --reflink=auto "$shared" "$dest"
    exit 0
fi

log "No usable LeRobot environment at $shared; building one at $dest"
# SKIP_DIALOUT keeps the installer from prompting for sudo; CI never opens the arms.
VENV="$dest" SKIP_DIALOUT=1 "$root/examples/so101/setup/install_lerobot_modalix.sh"
# The Studio and robot-client dependencies that install_polima_modalix.sh adds.
"$dest/bin/python" -m pip install --disable-pip-version-check \
    waitress "tokenizers==${TOKENIZERS_VERSION:-0.22.2}"
