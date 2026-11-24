#!/usr/bin/env bash
# Provide a LeRobot environment for DevKit qualification without touching the
# runner's shared installation.
#
# Usage: prepare_lerobot.sh [--copy] <candidate-venv>
#
# When the runner has a working shared environment, it is reused: printed as-is,
# or with --copy cloned to <candidate-venv> so the caller may install into it.
# Otherwise a fresh environment is built at <candidate-venv>. Ephemeral DevKit
# runners are not guaranteed to carry the shared environment.
#
# Prints the venv path to use on stdout; progress goes to stderr.
set -euo pipefail

copy=0
if [[ "${1:-}" == --copy ]]; then
    copy=1
    shift
fi
dest="${1:?usage: prepare_lerobot.sh [--copy] <candidate-venv>}"
shared="${POLIMA_CI_SHARED_LEROBOT:-/media/nvme/lerobot}"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

log() { echo "==> $*" >&2; }

if [[ -x "$shared/bin/python" ]] && "$shared/bin/python" -I -c 'import lerobot' >/dev/null 2>&1; then
    if [[ "$copy" == 1 ]]; then
        log "Copying the shared LeRobot environment at $shared"
        mkdir -p "$(dirname -- "$dest")"
        cp -a --reflink=auto "$shared" "$dest"
        echo "$dest"
    else
        log "Reusing the shared LeRobot environment at $shared"
        echo "$shared"
    fi
    exit 0
fi

log "No usable LeRobot environment at $shared; building one at $dest"
mkdir -p "$(dirname -- "$dest")"
# SKIP_DIALOUT keeps the installer from prompting for sudo; CI never opens the arms.
VENV="$dest" SKIP_DIALOUT=1 "$root/examples/so101/setup/install_lerobot_modalix.sh" >&2
# The Studio and robot-client dependencies that install_polima_modalix.sh adds.
"$dest/bin/python" -m pip install --disable-pip-version-check \
    waitress "tokenizers==${TOKENIZERS_VERSION:-0.22.2}" >&2
echo "$dest"
