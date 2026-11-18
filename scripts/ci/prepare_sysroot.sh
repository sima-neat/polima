#!/usr/bin/env bash
# Install the exact LLiMa core/dev packages from a lock file, and the SDK
# packages they need, into a disposable CI SDK sysroot. Same pattern as Neat
# Core's ensure_neat_llima and LLiMa's sysroot payloads: apt-get download and
# dpkg-deb -x. Afterwards every LLiMa file in the sysroot must be byte-identical
# to the locked packages, so a LLiMa left in a cached SDK cannot be used.
#
# usage: POLIMA_DISPOSABLE_CI=1 prepare_sysroot.sh LOCK_JSON LLIMA_DEB_DIR
set -euo pipefail

lock="${1:?lock file}"
debs="${2:?LLiMa package directory}"
sysroot="${SYSROOT:-/opt/toolchain/aarch64/modalix}"
libdir=usr/lib/aarch64-linux-gnu
record="$sysroot/var/lib/polima/resolved-dependencies.lock.json"

[[ "${POLIMA_DISPOSABLE_CI:-}" == 1 ]] ||
  { echo "Only a disposable CI SDK container may be modified (set POLIMA_DISPOSABLE_CI=1)" >&2; exit 2; }
[[ -d "$sysroot/usr/include" ]] || { echo "SDK sysroot not found: $sysroot" >&2; exit 2; }

# Same as LLiMa's build.sh: the SDK container user may not be root.
run_as_root() {
  if [[ "$(id -u)" -eq 0 ]]; then "$@"; return; fi
  command -v sudo >/dev/null 2>&1 || { echo "root permissions are required and sudo is not available" >&2; exit 1; }
  sudo "$@"
}

work="$(mktemp -d "${RUNNER_TEMP:-/tmp}/polima-sysroot.XXXXXX")"
trap 'rm -rf "$work"' EXIT

mapfile -t llima < <(python3 - "$lock" <<'PY'
import json, sys
lock = json.load(open(sys.argv[1], encoding="utf-8"))["llima"]
print(lock["version"])
for name, digest in sorted(lock["packages"].items()):
    print(f"{name}\t{digest}")
PY
)
version="${llima[0]}"
payload="$work/llima"
for entry in "${llima[@]:1}"; do
  name="${entry%%$'\t'*}"
  echo "${entry#*$'\t'}  $debs/$name" | sha256sum --check --strict --quiet
  [[ "$(dpkg-deb -f "$debs/$name" Version)" == "$version" ]] ||
    { echo "$name does not report LLiMa $version" >&2; exit 2; }
  dpkg-deb -x "$debs/$name" "$payload"
done

# SimaLMM's CMake package names <prefix>/include/eigen3; PoLiMa includes
# nlohmann/json; libsima_lmm_runtime needs libfftw3, which the 3.0 SDK sysroot
# lacks. Only what is missing is added.
present() { [[ -e "$sysroot/$1" ]]; }
packages=()
present usr/include/eigen3/Eigen/Core || packages+=(libeigen3-dev)
present usr/include/nlohmann/json.hpp || packages+=(nlohmann-json3-dev)
for library in $(readelf -d "$payload/$libdir/libsima_lmm_runtime.so" |
                 sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p'); do
  present "$libdir/$library" || present "lib/aarch64-linux-gnu/$library" || case "$library" in
    libfftw3.so.3) packages+=(libfftw3-double3:arm64) ;;
    *) echo "LLiMa needs $library, which the SDK sysroot does not provide" >&2; exit 2 ;;
  esac
done
if (( ${#packages[@]} )); then
  echo "Adding SDK packages to the sysroot: ${packages[*]}"
  mkdir -p "$work/apt"
  run_as_root apt-get update -qq  # a fresh SDK container has no package lists
  (cd "$work/apt" && apt-get download "${packages[@]}")
  for deb in "$work"/apt/*.deb; do run_as_root dpkg-deb -x "$deb" "$sysroot"; done
fi

for entry in "${llima[@]:1}"; do run_as_root dpkg-deb -x "$debs/${entry%%$'\t'*}" "$sysroot"; done
(cd "$payload" && find . -type f -print0) | while IFS= read -r -d '' file; do
  cmp -s "$payload/$file" "$sysroot/$file" ||
    { echo "sysroot ${file#.} differs from the locked LLiMa package" >&2; exit 2; }
done
test -f "$sysroot/usr/include/eigen3/Eigen/Core"
test -f "$sysroot/usr/include/nlohmann/json.hpp"
run_as_root install -D -m 0644 "$lock" "$record"
echo "LLiMa $version installed in $sysroot"
