#!/usr/bin/env bash
# One Vulcan build lane, run inside the Palette SDK container at /workspace.
#   release  manifest-pinned LLiMa; produces the publishable package
#   develop  latest published LLiMa develop; produces compatibility evidence
#
# usage: vulcan_build.sh release|develop   (output: dist/<lane>/)
set -euo pipefail

lane="${1:?release or develop}"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
case "$lane" in
  release) ref_args=() ;;
  develop) ref_args=(--ref "${POLIMA_LLIMA_DEVELOP_REF:-develop:latest}") ;;
  *) echo "unknown lane: $lane" >&2; exit 2 ;;
esac

# The shared resolver selected this sysroot; build only in the matching SDK.
if [[ -n "${SYSROOT_VERSION:-}" ]]; then
  grep -qx "Platform Version = $SYSROOT_VERSION" /etc/sdk-release || {
    echo "SDK container is not platform $SYSROOT_VERSION:" >&2; cat /etc/sdk-release >&2; exit 2; }
fi
source /opt/bin/simaai-init-build-env modalix >/dev/null

POLIMA_VERSION="$(python3 scripts/ci/package_version.py)"
export POLIMA_VERSION POLIMA_DISPOSABLE_CI=1
work="dist/work-$lane"
out="dist/$lane"
lock="$work/resolved-dependencies.lock.json"
rm -rf "$work" "$out"

python3 scripts/ci/resolve_llima.py --lane "$lane" "${ref_args[@]}" \
  --env "${NEAT_VULCAN_ENV:-prod}" --output "$work/llima" --lock "$lock"
scripts/ci/prepare_sysroot.sh "$lock" "$work/llima"
scripts/ci/build_runtime.sh "$lock" "$work/runtime"

if [[ "$lane" == release ]]; then
  scripts/ci/package_polima.sh "$lock" "$work/runtime" "$out"
else
  mkdir -p "$out"
  cp "$lock" "$work/runtime"/polima-runtime-*-modalix.tar.gz* "$out/"
fi
rm -rf "$work"
