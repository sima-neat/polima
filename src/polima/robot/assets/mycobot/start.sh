#!/usr/bin/env bash
# Board-side MyCobot launcher; arguments are supplied by native polima robot run.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${LEROBOT_VENV:-/media/nvme/lerobot}"
export PYTHONPATH="${POLIMA_PYTHON_ROOT:-/media/nvme/polima/python}:$ROOT/robot_client:${PYTHONPATH:-}"
ROBOT_PORT=""
FRONT_CAMERA=""
WRIST_CAMERA=""
SERVER=""
FPS="20"
ACTIONS="100"
MAX_TARGET="12"
CALIBRATION="${MYCOBOT_CALIBRATION_PATH:-}"
TASK="${POLIMA_TASK_ID:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --robot-port) ROBOT_PORT="$2"; shift 2 ;;
        --front-camera) FRONT_CAMERA="$2"; shift 2 ;;
        --wrist-camera) WRIST_CAMERA="$2"; shift 2 ;;
        --server-address) SERVER="$2"; shift 2 ;;
        --fps) FPS="$2"; shift 2 ;;
        --actions-per-chunk) ACTIONS="$2"; shift 2 ;;
        --max-relative-target) MAX_TARGET="$2"; shift 2 ;;
        *) echo "unknown MyCobot launcher argument: $1" >&2; exit 2 ;;
    esac
done

CAMERA_FOURCC="${POLIMA_CAMERA_FOURCC:-YUYV}"
FRONT_CAMERA_FPS="${POLIMA_FRONT_CAMERA_FPS:-30}"
WRIST_CAMERA_FPS="${POLIMA_WRIST_CAMERA_FPS:-25}"
MAX_CYCLES="${POLIMA_MAX_CYCLES:-}"
NO_AUTO_COMPLETE="${POLIMA_NO_AUTO_COMPLETE:-}"
DIAGNOSTIC_DIR="${POLIMA_DIAGNOSTIC_DIR:-}"
if [[ "$DIAGNOSTIC_DIR" == "auto" ]]; then
    RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
    SAFE_TASK="${TASK//[^a-zA-Z0-9_.-]/_}"
    DIAGNOSTIC_DIR="/media/nvme/polima/diagnostics/${RUN_STAMP}-task-${SAFE_TASK}-$$"
fi

for name in ROBOT_PORT FRONT_CAMERA WRIST_CAMERA SERVER; do
    value="${!name}"
    if [[ -z "$value" ]]; then
        echo "missing required MyCobot launcher value: $name" >&2
        exit 2
    fi
done

args=(
    --bundle "$ROOT"
    --robot-port "$ROBOT_PORT"
    --front-camera "$FRONT_CAMERA"
    --wrist-camera "$WRIST_CAMERA"
    --server "$SERVER"
    --fps "$FPS"
    --camera-fourcc "$CAMERA_FOURCC"
    --front-camera-fps "$FRONT_CAMERA_FPS"
    --wrist-camera-fps "$WRIST_CAMERA_FPS"
    --actions-per-chunk "$ACTIONS"
    --max-initial-target-error-deg "$MAX_TARGET"
    --task "$TASK"
    --execute
)

if [[ -z "$CALIBRATION" && -f "$ROOT/robot_client/calibration/mycobot280.json" ]]; then
    CALIBRATION="$ROOT/robot_client/calibration/mycobot280.json"
fi
if [[ -n "$CALIBRATION" ]]; then
    args+=(--calibration-path "$CALIBRATION")
fi
if [[ -n "$MAX_CYCLES" ]]; then
    args+=(--max-cycles "$MAX_CYCLES")
fi
if [[ -n "$NO_AUTO_COMPLETE" ]]; then
    args+=(--no-auto-complete)
fi
if [[ -n "$DIAGNOSTIC_DIR" ]]; then
    args+=(--diagnostic-dir "$DIAGNOSTIC_DIR")
fi

exec "$VENV/bin/python" -m polima.robot.mycobot "${args[@]}"
