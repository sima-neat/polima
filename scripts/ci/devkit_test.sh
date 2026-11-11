#!/usr/bin/env bash
# Test one exact PoLiMa package on a Modalix DevKit, without installing it
# system-wide: runtime identity, kernel tests, then every reference bundle
# served by polima_server and checked by `polima run --fixture` (the bundle's
# own accuracy envelope) from the package's wheel. Nothing is built here.
#
# usage: devkit_test.sh PACKAGE_DIR RESULTS_DIR
#   POLIMA_CI_MODEL_CACHE  bundle cache (default /media/nvme/polima-ci/models)
#   POLIMA_CI_BUNDLES      space-separated subset of reference_bundles.json names
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
package="$(cd -- "${1:?package directory}" && pwd)"
results="${2:?results directory}"
cache="${POLIMA_CI_MODEL_CACHE:-/media/nvme/polima-ci/models}"
mkdir -p "$results" "$cache"
results="$(cd -- "$results" && pwd)"

work="$(mktemp -d "${RUNNER_TEMP:-/tmp}/polima-devkit.XXXXXX")"
server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -INT "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  rm -rf "$work"
}
trap cleanup EXIT

[[ "$(uname -m)" == aarch64 ]] || { echo "run on the Modalix DevKit" >&2; exit 2; }
version="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$package/metadata.json")"
archive="$package/polima-runtime-$version-modalix.tar.gz"
(cd "$package" && sha256sum --check --strict "$(basename -- "$archive").sha256")
tar -xzf "$archive" -C "$work"
cli="$work/runtime/bin/polima_cli"
server="$work/runtime/bin/polima_server"

"$cli" --runtime-info | tee "$results/runtime-info.txt"
"$work/runtime/tests/polima_kernel_tests" | tee "$results/kernel-tests.txt"

python3 -m venv "$work/venv"
"$work/venv/bin/python" -m pip install --quiet --no-cache-dir numpy
"$work/venv/bin/python" -m pip install --quiet --no-cache-dir --no-deps "$package"/polima-*.whl

mapfile -t bundles < <(python3 - "$root/scripts/ci/reference_bundles.json" "${POLIMA_CI_BUNDLES:-}" <<'PY'
import json, sys
wanted = set(sys.argv[2].split())
for bundle in json.load(open(sys.argv[1]))["bundles"]:
    if not wanted or bundle["name"] in wanted:
        print(f"{bundle['name']}\t{bundle['repo']}\t{bundle['revision']}")
PY
)
(( ${#bundles[@]} )) || { echo "no reference bundles selected" >&2; exit 2; }

status=0
for entry in "${bundles[@]}"; do
  IFS=$'\t' read -r name repo revision <<< "$entry"
  bundle="$(python3 "$root/scripts/ci/fetch_bundle.py" "$repo" "$revision" --cache "$cache")"
  port="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')"
  "$server" --bundle "$bundle" --port "$port" > "$results/$name-server.log" 2>&1 &
  server_pid=$!
  for _ in $(seq 1 300); do
    kill -0 "$server_pid" 2>/dev/null || break
    python3 -c 'import socket,sys; socket.create_connection(("127.0.0.1",int(sys.argv[1])),1)' "$port" \
      2>/dev/null && break
    sleep 1
  done
  if "$work/venv/bin/python" -m polima run --bundle "$bundle" --fixture --devkit 127.0.0.1 \
      --port "$port" --repeat 10 --json > "$results/$name.json"; then
    echo "$name ($repo@${revision:0:12}): fixture passed"
  else
    echo "$name ($repo@${revision:0:12}): fixture FAILED" >&2
    status=1
  fi
  kill -INT "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  server_pid=""
done
exit "$status"
