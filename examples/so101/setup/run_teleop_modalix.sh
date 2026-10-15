#!/usr/bin/env bash
# Safely preflight and start SO-101 leader/follower teleoperation on Modalix.
set -euo pipefail

VENV="${VENV:-/media/nvme/lerobot}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  ./run_teleop_modalix.sh
  ./run_teleop_modalix.sh LEADER_PORT FOLLOWER_PORT

Roles:
  LEADER   = input/controller arm; YOU move this arm by hand.
  FOLLOWER = output/robot arm; its powered motors COPY the leader.

Use stable /dev/serial/by-id/... paths when available. If both boards have the
same USB serial number, use /dev/serial/by-path/... and keep each arm in the
same physical USB socket.

Example:
  ./run_teleop_modalix.sh \
    /dev/serial/by-path/LEADER_LINK \
    /dev/serial/by-path/FOLLOWER_LINK

Set SKIP_PAIR_CHECK=1 only when intentionally bypassing the read-only preflight.

With no ports, the script interactively identifies each arm by asking you to
disconnect and reconnect it. Supplying two ports skips that identification.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 0 && $# -ne 2 ]]; then
  usage >&2
  exit 2
fi

python_bin="$VENV/bin/python"
teleop_bin="$VENV/bin/lerobot-teleoperate"

[[ -x "$python_bin" ]] || { echo "FAIL: Python not found at $python_bin" >&2; exit 1; }
[[ -x "$teleop_bin" ]] || { echo "FAIL: lerobot-teleoperate not found at $teleop_bin" >&2; exit 1; }

if [[ $# -eq 0 ]]; then
  ports=()
  while IFS= read -r port; do
    ports+=("$port")
  done < <("$python_bin" "$SCRIPT_DIR/discover_teleop_ports.py")
  if [[ ${#ports[@]} -ne 2 ]]; then
    echo "FAIL: port identification did not return a leader/follower pair" >&2
    exit 1
  fi
  leader_port="${ports[0]}"
  follower_port="${ports[1]}"
else
  leader_port="$1"
  follower_port="$2"
fi

echo "ROLE MAP"
echo "  LEADER   [you move it] : $leader_port"
echo "  FOLLOWER [robot moves]  : $follower_port"
echo

if [[ "${SKIP_PAIR_CHECK:-0}" != "1" ]]; then
  "$python_bin" "$SCRIPT_DIR/test_teleop_pair.py" \
    --leader-port "$leader_port" --follower-port "$follower_port"
fi

cat <<'EOF'

Starting LeRobot.

If calibration for modalix_leader or modalix_follower does not exist, LeRobot will
first guide you through calibration. Follow the role shown in each calibration
prompt and do not swap the ports. Teleoperation starts only after calibration.

Keep clear of the FOLLOWER arm and have its power switch or emergency stop within
reach. Press Ctrl-C to stop.
EOF

exec "$teleop_bin" \
  --robot.type=so101_follower \
  --robot.port="$follower_port" \
  --robot.id=modalix_follower \
  --teleop.type=so101_leader \
  --teleop.port="$leader_port" \
  --teleop.id=modalix_leader
