#!/usr/bin/env bash
# Reuse the established SO101 loop with bundle-driven GR00T transport.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${LEROBOT_VENV:-/media/nvme/lerobot}"
export PYTHONPATH="${POLIMA_PYTHON_ROOT:-/media/nvme/polima/python}:$ROOT/robot_client:${PYTHONPATH:-}"
exec "$VENV/bin/python" -m polima.robot.so101 --bundle "$ROOT" "$@"
