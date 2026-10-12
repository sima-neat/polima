#!/usr/bin/env bash
# Minimal LeRobot install for the SiMa.ai Modalix SOM (arm64 / eLxr), teleoperation only.
#
# Builds a venv on the SOM's SYSTEM python3 with --system-site-packages, so the venv can
# still import vendor packages from /usr/lib/python3/dist-packages (SiMa Palette runtime,
# ROS 2 rclpy, system cv2). Those are cp311 C-extensions and are ONLY importable by the
# system interpreter -- which is why this pins LeRobot 0.4.4:
#
#   lerobot <=0.4.4  requires-python >=3.10   <- works on eLxr/bookworm's python3.11
#   lerobot >=0.5.0  requires-python >=3.12   <- would force a standalone interpreter
#                                                and lose system-site-packages entirely
#
# No uv, no conda, no apt. Stock venv + pip.
#
# Usage:
#   ./install_lerobot_modalix.sh                    # SO-100/SO-101/Moss (Feetech)
#   MOTORS=dynamixel ./install_lerobot_modalix.sh   # Koch v1.1
#   MOTORS=both ./install_lerobot_modalix.sh
#
# Env knobs:
#   VENV=<path>          venv location             (default: /media/nvme/lerobot)
#   MOTORS=feetech|dynamixel|both                  (default: feetech)
#   LEROBOT_VERSION=<v>  pin override              (default: 0.4.4)
#   ISOLATED=1           drop --system-site-packages (fully isolated venv)
#   SKIP_DIALOUT=1       don't touch group membership

set -euo pipefail

VENV="${VENV:-/media/nvme/lerobot}"
MOTORS="${MOTORS:-feetech}"
LEROBOT_VERSION="${LEROBOT_VERSION:-0.4.4}"
# LeRobot's permitted W&B range is broad enough that pip can select a build
# whose generated protobufs require a newer runtime than Debian provides.
# Pin this matching pair in the venv so --system-site-packages cannot expose
# the incompatible system protobuf instead.
WANDB_VERSION="${WANDB_VERSION:-0.24.2}"
PROTOBUF_VERSION="${PROTOBUF_VERSION:-6.31.1}"
# The asynchronous PoLiMa client connects to the on-device policy service over
# gRPC. LeRobot 0.4.4 does not make grpcio a dependable base dependency.
GRPCIO_VERSION="${GRPCIO_VERSION:-1.80.0}"
PYTHON="${PYTHON:-python3}"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight --
log "Preflight"

arch="$(uname -m)"
[ "$arch" = "aarch64" ] || die "expected aarch64 (Modalix SOM), got $arch"

# torch/opencv publish manylinux_2_28 aarch64 wheels; bookworm ships glibc 2.36.
glibc="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || echo 0)"
awk -v g="$glibc" 'BEGIN{split(g,v,"."); exit !(v[1]>2 || (v[1]==2 && v[2]>=28))}' \
  || die "glibc $glibc < 2.28; the prebuilt aarch64 torch/opencv wheels will not load"

command -v "$PYTHON" >/dev/null 2>&1 || die "$PYTHON not found"
pyver="$($PYTHON -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
"$PYTHON" -c 'import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)' \
  || die "lerobot $LEROBOT_VERSION needs python >=3.10, system python is $pyver"

# Debian splits venv out of the stdlib; we are pip-only so surface the fix, don't run it.
"$PYTHON" -c 'import venv, ensurepip' 2>/dev/null \
  || die "venv/ensurepip missing. Run: sudo apt-get install -y python3-venv"

printf '    arch=%s glibc=%s python=%s os=%s\n' "$arch" "$glibc" "$pyver" \
  "$(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-unknown}")"

# 0.4.4's base already bundles pynput/pyserial/deepdiff/opencv/rerun, so the motor
# SDK is the only extra teleop actually needs. ("hardware" does not exist until 0.5.0.)
case "$MOTORS" in
  feetech)   EXTRAS="feetech" ;;
  dynamixel) EXTRAS="dynamixel" ;;
  both)      EXTRAS="feetech,dynamixel" ;;
  *) die "MOTORS must be feetech|dynamixel|both (got '$MOTORS')" ;;
esac

# ------------------------------------------------------------------ venv -----
SSP="--system-site-packages"
[ "${ISOLATED:-0}" = "1" ] && SSP=""

# The SOM's eMMC is small and this install unpacks to ~2-3 GB, so the venv goes on the
# NVMe. Verify the target is real, writable and roomy BEFORE pip starts -- if the NVMe
# isn't mounted, /media/nvme is just a directory on the rootfs and we'd fill the eMMC.
venv_parent="$(dirname "$VENV")"
[ -d "$venv_parent" ] \
  || die "$venv_parent does not exist -- mount the NVMe first, or override with VENV=<path>"
[ -w "$venv_parent" ] \
  || die "$venv_parent is not writable by $USER"

if [ "$(stat -c %d "$venv_parent" 2>/dev/null || echo x)" = "$(stat -c %d / 2>/dev/null || echo y)" ]; then
  warn "$venv_parent is on the SAME filesystem as / -- the NVMe is not mounted there."
  warn "Verify with 'findmnt $venv_parent'. Continuing will consume eMMC space."
fi

avail_kb="$(df -Pk "$venv_parent" | awk 'NR==2{print $4}')"
avail_gb=$((avail_kb / 1024 / 1024))
[ "$avail_kb" -ge 4194304 ] \
  || warn "only ${avail_gb} GB free on $venv_parent; this install needs roughly 3 GB"

log "Creating venv at $VENV ${SSP:-(isolated)} -- ${avail_gb} GB free"
# shellcheck disable=SC2086
"$PYTHON" -m venv $SSP "$VENV"

# Record what the system interpreter sees, before the venv can shadow anything.
sys_numpy="$("$PYTHON" -c 'import numpy;print(numpy.__version__)' 2>/dev/null || echo none)"

# ---------------------------------------------------------------- install ----
# aarch64 torch wheels on PyPI are already CPU-only (~146 MB, no bundled CUDA),
# so no --extra-index-url is needed. torchcodec self-excludes on aarch64 via its
# environment marker, and lerobot falls back to pyav for video decode.
log "Installing lerobot[$EXTRAS]==$LEROBOT_VERSION (this pulls ~1 GB, be patient on the SOM)"
export PIP_DISABLE_PIP_VERSION_CHECK=1
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV/bin/python" -m pip install \
  "lerobot[$EXTRAS]==$LEROBOT_VERSION" \
  "wandb==$WANDB_VERSION" \
  "protobuf==$PROTOBUF_VERSION" \
  "grpcio==$GRPCIO_VERSION"

# The PoLiMa robot clients expose their already-captured camera observations as
# a tokenized MJPEG page. Flask is intentionally installed in this standalone
# LeRobot environment because the native PoLiMa package itself is not installed
# on the board.
log "Installing Flask for the PoLiMa live camera preview"
"$VENV/bin/python" -m pip install flask

# scipy is an UNDECLARED dependency of the teleop entry point in 0.4.4: the metadata
# only lists it under the `wallx` extra, but policies/pi0_fast/modeling_pi0_fast.py
# does a top-level `from scipy.fftpack import idct`, and lerobot_teleoperate reaches
# that module through robots -> unitree_g1 -> envs.factory -> policies. So every
# lerobot-teleoperate run imports scipy no matter which robot you actually drive.
#
# Without this, the install breaks in two different ways and neither is obvious:
#   --system-site-packages -> falls through to the distro scipy, which is built against
#                             numpy 1.x while rerun-sdk has forced numpy 2.x into the
#                             venv: "ValueError: numpy.dtype size changed"
#   ISOLATED=1             -> nothing to fall through to: "ModuleNotFoundError: scipy"
# Pinning >=1.13 because that is the first release with numpy 2 support.
log "Installing scipy (undeclared dependency of lerobot-teleoperate in $LEROBOT_VERSION)"
"$VENV/bin/python" -m pip install "scipy>=1.13"

# ------------------------------------------------------------------ verify ---
log "Verifying"
"$VENV/bin/python" - <<'PY'
import importlib.metadata as md, platform, sys
print(f"    python  {sys.version.split()[0]} ({platform.machine()})")
print(f"    lerobot {md.version('lerobot')}")
import torch; print(f"    torch   {torch.__version__}")
import flask; print(f"    flask  {md.version('flask')}")
import grpc; print(f"    grpcio {md.version('grpcio')}")
import google.protobuf.runtime_version
print(f"    protobuf {md.version('protobuf')}")
print(f"    wandb    {md.version('wandb')}")
for p in ("feetech-servo-sdk", "dynamixel-sdk"):
    try: print(f"    {p:<18} {md.version(p)}")
    except md.PackageNotFoundError: pass
PY

# The entry point is the real smoke test: --help imports the whole teleop stack
# (rerun, every robot and teleoperator config) in one go, so it fails for reasons the
# metadata checks above cannot see. Two failures look identical from the exit status
# alone but need opposite fixes -- the script is MISSING (package installed, console
# scripts landed elsewhere) versus the script CRASHES (scripts fine, an import blew
# up) -- so report which one happened, and never discard the interpreter's own error.
teleop_bin="$VENV/bin/lerobot-teleoperate"

if [ ! -x "$teleop_bin" ]; then
  warn "lerobot-teleoperate is not in $VENV/bin, but pip reported lerobot installed."
  warn "The package landed somewhere its console scripts did not follow."
  echo "    lerobot-* scripts present in $VENV/bin:"
  found=0
  for f in "$VENV"/bin/lerobot-*; do
    [ -e "$f" ] || continue
    printf '      %s\n' "${f##*/}"
    found=$((found + 1))
  done
  if [ "$found" -eq 0 ]; then
    echo "      (none)"
    # pip configured with `user = true` installs modules importable via
    # --system-site-packages while diverting scripts to ~/.local/bin, which produces
    # exactly this split: `import lerobot` works, every lerobot-* command does not.
    for f in "$HOME"/.local/bin/lerobot-*; do
      [ -e "$f" ] || continue
      warn "Found lerobot scripts in $HOME/.local/bin instead -- pip is installing --user."
      warn "Check for 'user = true':  $VENV/bin/python -m pip config list; cat /etc/pip.conf"
      warn "Then reinstall into the venv:  PIP_USER=0 $VENV/bin/python -m pip install \\"
      warn "    --force-reinstall --no-deps 'lerobot[$EXTRAS]==$LEROBOT_VERSION'"
      break
    done
  fi
  die "lerobot-teleoperate entry point missing"
fi

if teleop_help="$("$teleop_bin" --help 2>&1)"; then
  echo "    lerobot-teleoperate OK"
else
  teleop_rc=$?
  warn "lerobot-teleoperate exists but exited $teleop_rc. Its own output, last 20 lines:"
  printf '%s\n' "$teleop_help" | tail -20 | sed 's/^/      /'
  case "$teleop_help" in
    *_ARRAY_API*|*numpy.dtype\ size\ changed*)
      warn "This is the numpy ABI break described below -- a C extension was built against"
      warn "numpy 1.x but the venv now has numpy 2.x. Re-run with ISOLATED=1." ;;
    *"No module named 'rerun'"*)
      warn "rerun-sdk is missing. It is imported unconditionally by lerobot_teleoperate,"
      warn "so teleop cannot start without it:  $VENV/bin/python -m pip install rerun-sdk" ;;
  esac
  die "lerobot-teleoperate entry point is installed but not runnable"
fi

# --system-site-packages is one-way: venv packages take precedence over system ones.
# If pip had to upgrade numpy past the system copy, vendor cp311 extensions compiled
# against the old ABI can fail to import *inside this venv* (they stay fine outside).
if [ "${ISOLATED:-0}" != "1" ]; then
  venv_numpy="$("$VENV/bin/python" -c 'import numpy;print(numpy.__version__)' 2>/dev/null || echo none)"
  if [ "$sys_numpy" != "none" ] && [ "$sys_numpy" != "$venv_numpy" ]; then
    warn "numpy shadowed: system $sys_numpy -> venv $venv_numpy"
    warn "This is forced, not incidental: lerobot $LEROBOT_VERSION depends on rerun-sdk,"
    warn "which requires numpy>=2, so numpy<2 is unsatisfiable. Inside this venv the new"
    warn "numpy shadows the system copy, and any system C-extension built against numpy 1.x"
    warn "(system cv2 is the usual casualty) will fail to import with '_ARRAY_API not found'."
    warn "Pure-python vendor modules are only at risk if their own code is numpy-1 specific."
    warn "Test what you actually need:  $VENV/bin/python -c 'import cv2, <vendor_module>'"
    warn "If a module you need is broken this way, --system-site-packages is not buying you"
    warn "much: re-run with ISOLATED=1 and reach the vendor runtime out-of-process instead."
  fi

  log "System packages still reachable from the venv:"
  "$VENV/bin/python" - <<'PY'
import importlib.util as u
for m in ("rclpy", "cv2", "sima", "simaai", "afe"):
    s = None
    try: s = u.find_spec(m)
    except Exception: pass
    if s: print(f"    {m:<10} {getattr(s,'origin',None)}")
PY
fi

# ------------------------------------------------------- serial inventory ----
# ttyACM* indices follow USB *enumeration order*, not the device, so leader and follower
# swap across a replug, a reboot, or a different power-on order. Because --robot.id keys
# the calibration files, a swap silently applies one arm's homing offsets to the other --
# both ports open, both arms respond, only the mapping is wrong. So pin a stable path
# BEFORE the first calibration.
#
# Advisory only: the arms may not be plugged in during install, so nothing here is fatal.
# Dirs are overridable purely so this logic can be exercised against a fixture tree.
BY_ID_DIR="${BY_ID_DIR:-/dev/serial/by-id}"
BY_PATH_DIR="${BY_PATH_DIR:-/dev/serial/by-path}"
DEV_DIR="${DEV_DIR:-/dev}"

# prints "<stable path>\t<resolved tty>" per line
scan_serial() {
  local d="$1" link target
  [ -d "$d" ] || return 0
  for link in "$d"/*; do
    [ -e "$link" ] || continue
    target="$(readlink -f "$link" 2>/dev/null)" || continue
    case "$target" in
      */ttyACM*|*/ttyUSB*) printf '%s\t%s\n' "$link" "$target" ;;
    esac
  done
  return 0
}

# Counted with a glob, not `ls`: an unmatched glob makes ls exit 2, and under
# `set -euo pipefail` that aborts the whole install. ttyUSB* is normally absent.
count_tty() {
  local n=0 f
  for f in "$DEV_DIR"/ttyACM* "$DEV_DIR"/ttyUSB*; do
    [ -e "$f" ] && n=$((n + 1))
  done
  printf '%s' "$n"
}

count_lines() {
  if [ -z "$1" ]; then printf '0'; else printf '%s\n' "$1" | wc -l | tr -d ' '; fi
}

tty_count="$(count_tty)"
by_id="$(scan_serial "$BY_ID_DIR")"
by_id_count="$(count_lines "$by_id")"

stable_list=""
PORT_A=""; PORT_B=""

log "Serial devices"
if [ "$tty_count" -eq 0 ]; then
  warn "No ttyACM*/ttyUSB* present -- arms not connected right now."
  warn "Plug both arms in, then re-check with:  ls -l $BY_ID_DIR"
elif [ "$by_id_count" -ge "$tty_count" ] && [ "$by_id_count" -gt 0 ]; then
  stable_list="$by_id"
  echo "    stable path (by-id, follows the device across replug):"
  printf '%s\n' "$by_id" | while IFS="$(printf '\t')" read -r l t; do
    [ -n "$l" ] && printf '      %s  ->  %s\n' "$l" "$t"
  done
else
  # Boards report a duplicate or absent USB iSerial, so udev could not mint one
  # by-id link per device. by-path keys on the physical connector instead.
  warn "Only $by_id_count by-id link(s) for $tty_count serial device(s) -- the driver"
  warn "boards likely share (or omit) a USB serial number. Falling back to by-path."
  warn "by-path is stable ONLY if each arm always goes into the same USB connector."
  stable_list="$(scan_serial "$BY_PATH_DIR")"
  if [ -n "$stable_list" ]; then
    echo "    stable path (by-path, follows the USB connector):"
    printf '%s\n' "$stable_list" | while IFS="$(printf '\t')" read -r l t; do
      [ -n "$l" ] && printf '      %s  ->  %s\n' "$l" "$t"
    done
  else
    warn "No $BY_PATH_DIR entries either; fall back to raw /dev/ttyACM* and re-check after every replug."
  fi
fi

if [ -n "$stable_list" ]; then
  PORT_A="$(printf '%s\n' "$stable_list" | sed -n '1s/\t.*//p')"
  PORT_B="$(printf '%s\n' "$stable_list" | sed -n '2s/\t.*//p')"
fi

# -------------------------------------------------------------------- next ---
cat <<EOF

$(log "Done")

  Activate:      source $VENV/bin/activate
  PoLiMa server: polima server                 # interactive start/stop
  PoLiMa robot:  polima robot                  # inspect + confirm startup
  Find ports:    lerobot-find-port          # unplug/replug each arm when prompted
  Teleoperate:   lerobot-teleoperate \\
                   --robot.type=so101_follower --robot.port=${PORT_A:-/dev/ttyACM0} --robot.id=follower \\
                   --teleop.type=so101_leader --teleop.port=${PORT_B:-/dev/ttyACM1} --teleop.id=leader

  ${PORT_A:+NOTE: the two ports above are the stable paths found on this machine, but which
  one is the leader and which is the follower CANNOT be inferred from the path. Run
  lerobot-find-port and unplug one arm when prompted to identify it, then swap the two
  --*.port values above if needed.

  }First run auto-triggers calibration. Calibration is keyed to --robot.id, so a leader/follower
  swap loads one arm's homing offsets onto the other -- it will move to the wrong joint angles
  and nothing will error. Settle the port assignment BEFORE calibrating, and reuse the same ids
  afterwards. These paths are symlinks; pyserial opens them transparently, so pass them directly.

Platform notes:
  * Pinned to lerobot $LEROBOT_VERSION on purpose: 0.5.0+ requires python >=3.12 and would
    break --system-site-packages against eLxr's python $pyver. Drop the pin only if you
    also drop system-site-packages (ISOLATED=1 + a 3.12 interpreter).
  * Leader-arm teleop is headless-safe over SSH. Keyboard teleop (--teleop.type=keyboard)
    needs a global X11 key backend and will NOT work on a headless SOM.
  * torchcodec is skipped on aarch64 by its own marker; video decode falls back to pyav.
  * torch here is CPU-only. The Modalix NPU is driven by SiMa's Palette runtime, not torch.
EOF

# --------------------------------------------------- serial perms (last) -----
# Deliberately the very last step: it is the only part needing root, so the long
# unattended pip install never stalls waiting on a password prompt. By the time sudo
# asks, everything else has already succeeded and been reported.
#
# The password is intentionally NOT embedded here. A device credential in a file that
# lives in a project directory ends up in git, and `sudo -S <<<"pw"` additionally leaks
# it into the process table. For a fully unattended run, pre-authorize sudo's cached
# timestamp instead -- same effect, nothing written down:
#
#     sudo -v && ./install_lerobot_modalix.sh
#
# This step then consumes the cached credential and never prompts.
if [ "${SKIP_DIALOUT:-0}" != "1" ]; then
  if id -nG "$USER" | tr ' ' '\n' | grep -qx dialout; then
    log "Serial access: $USER is already in 'dialout' -- nothing left to do"
  elif ! command -v sudo >/dev/null 2>&1; then
    warn "sudo not found. As root, run:  usermod -aG dialout $USER"
  else
    echo
    log "One step remains, and it needs root."
    echo "    /dev/ttyACM* is crw-rw---- root:dialout, so $USER cannot open the arms"
    echo "    until it joins that group. Enter your sudo password below,"
    echo "    or press Ctrl-C and run the command yourself later."
    echo
    if sudo usermod -aG dialout "$USER"; then
      log "Added $USER to 'dialout'."
      warn "Group changes only apply to NEW logins -- run 'newgrp dialout' in this shell,"
      warn "or log out and back in, before teleoperating."
    else
      echo
      warn "Not applied. Before teleoperating, run:"
      warn "    sudo usermod -aG dialout $USER && newgrp dialout"
    fi
  fi
fi
