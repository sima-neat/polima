#!/usr/bin/env bash
# Run the SO-101 control loop for the bundle this script lives in.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${LEROBOT_VENV:-/media/nvme/lerobot}"
export PYTHONPATH="${POLIMA_PYTHON_ROOT:-/media/nvme/polima/python}:$ROOT/robot_client:${PYTHONPATH:-}"
exec "$VENV/bin/python" -m polima.robot.so101 --bundle "$ROOT" "$@"
