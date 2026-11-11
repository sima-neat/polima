#!/usr/bin/env bash
# Build polima_cli/polima_server against the exact LLiMa in a lock file and
# write a reproducible runtime archive that records that identity.
#
# Runs in the Palette SDK container after prepare_sysroot.sh installed the
# locked LLiMa into $SYSROOT.
#
# usage: build_runtime.sh LOCK_JSON OUTPUT_DIR
#   POLIMA_VERSION   package version (default: manifest package-version)
#   POLIMA_REQUIRE_KERNEL_TESTS=1  fail if the kernel tests cannot run here
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
lock="${1:?lock file}"
output="${2:?output directory}"
sysroot="${SYSROOT:?run inside the SDK build environment (SYSROOT)}"
jobs="${POLIMA_BUILD_JOBS:-$(nproc)}"
source_commit="${GITHUB_SHA:-$(git -C "$root" rev-parse HEAD 2>/dev/null || echo unknown)}"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git -C "$root" log -1 --format=%ct 2>/dev/null || echo 0)}"

mapfile -t identity < <(python3 - "$root/deps/manifest.json" "$lock" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
lock = json.load(open(sys.argv[2], encoding="utf-8"))
print(manifest["package-version"])
print(manifest["platform-version"])
print(lock["lane"])
print(lock["llima"]["version"])
PY
)
polima_version="${POLIMA_VERSION:-${identity[0]}}"
platform_version="${identity[1]}"
lane="${identity[2]}"
llima_version="${identity[3]}"
cmp -s "$lock" "$sysroot/var/lib/polima/resolved-dependencies.lock.json" ||
  { echo "$sysroot was not prepared from $lock (run prepare_sysroot.sh)" >&2; exit 2; }

work="$(mktemp -d "${RUNNER_TEMP:-/tmp}/polima-runtime.XXXXXX")"
trap 'rm -rf "$work"' EXIT
build="$work/build"

cmake_args=(
  -S "$root/runtime" -B "$build"
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_TOOLCHAIN_FILE="$root/runtime/cmake/toolchain-sima.cmake"
  -DPOLIMA_PLATFORM_VERSION="$platform_version"
  -DPOLIMA_LLIMA_PACKAGE_VERSION="$llima_version"
  -DPOLIMA_SOURCE_COMMIT="$source_commit"
  "-DCMAKE_CXX_FLAGS=-ffile-prefix-map=$root=/polima -ffile-prefix-map=$work=/build"
)
[[ "$lane" != release ]] || cmake_args+=(-DPOLIMA_REQUIRED_LLIMA_VERSION="${llima_version%%+*}")
readelf=aarch64-linux-gnu-readelf

# The SDK environment script exports -I$SYSROOT/usr/include in CXXFLAGS, which
# would let sysroot headers shadow the locked LLiMa; the toolchain file already
# supplies --sysroot. Match the DevKit's native compiler defaults otherwise.
env -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS cmake "${cmake_args[@]}"
simalmm_dir="$(sed -n 's/^SimaLMM_DIR:PATH=//p' "$build/CMakeCache.txt")"
[[ "$simalmm_dir" == "$sysroot/usr/lib/aarch64-linux-gnu/cmake/SimaLMM" ]] ||
  { echo "SimaLMM came from $simalmm_dir, not the locked LLiMa in $sysroot" >&2; exit 2; }
env -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS cmake --build "$build" -j"$jobs"

for binary in polima_cli polima_server polima_kernel_tests; do
  if "$readelf" -d "$build/$binary" | grep -Eq '\((RPATH|RUNPATH)\)'; then
    echo "$binary must not retain a build-machine RPATH" >&2
    exit 2
  fi
done

# The kernel tests use only libstdc++; run them with the target's own loader.
loader="$sysroot/lib/ld-linux-aarch64.so.1"
if [[ "$(uname -m)" == aarch64 && -x "$loader" ]]; then
  "$loader" --library-path "$sysroot/lib/aarch64-linux-gnu:$sysroot/usr/lib/aarch64-linux-gnu" \
    "$build/polima_kernel_tests"
elif [[ "${POLIMA_REQUIRE_KERNEL_TESTS:-0}" == 1 ]]; then
  echo "kernel tests are required but cannot run on $(uname -m)" >&2
  exit 2
else
  echo "kernel tests not run on $(uname -m); the DevKit test runs them"
fi

stage="$work/runtime"
install -d "$stage/bin" "$stage/tests"
install -m 0755 "$build/polima_cli" "$build/polima_server" "$stage/bin/"
install -m 0755 "$build/polima_kernel_tests" "$stage/tests/"
python3 - "$stage/runtime.json" "$lock" "$polima_version" "$source_commit" "$platform_version" <<'PY'
import json, sys
path, lock_path, polima, source, platform = sys.argv[1:]
lock = json.load(open(lock_path, encoding="utf-8"))
llima = lock["llima"]
json.dump({
    "schema_version": 2,
    "polima_version": polima,
    "source_commit": source,
    "platform_version": platform,
    "llima_package_version": llima["version"],
    "lane": lock["lane"],
    "llima": {key: llima[key] for key in
              ("ref", "resolved_spec", "commit", "metadata_url", "packages")},
}, open(path, "w", encoding="utf-8"), indent=2, sort_keys=True)
open(path, "a", encoding="utf-8").write("\n")
PY

mkdir -p "$output"
archive="$output/polima-runtime-$polima_version-modalix.tar.gz"
[[ ! -e "$archive" ]] || { echo "$archive already exists" >&2; exit 2; }
touch -d "@$SOURCE_DATE_EPOCH" "$stage" "$stage"/* "$stage"/*/*
tar --sort=name --mtime="@$SOURCE_DATE_EPOCH" --owner=0 --group=0 --numeric-owner \
  -C "$work" -cf - runtime | gzip -n > "$archive"
(cd "$output" && sha256sum "$(basename -- "$archive")" > "$(basename -- "$archive").sha256")
echo "$archive"
