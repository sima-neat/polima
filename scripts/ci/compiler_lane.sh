#!/usr/bin/env bash
# Resolve and exercise one exact Model Compiler lane on an isolated Vulcan runner.
set -euo pipefail

lane="${1:?release or develop}"
output="${2:?evidence directory}"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ "$lane" == release || "$lane" == develop ]] || { echo "invalid lane: $lane" >&2; exit 2; }
mkdir -p "$output"

readarray -t compiler < <(python3 - "$root/deps/manifest.json" "$lane" <<'PY'
import json, sys
entry = json.load(open(sys.argv[1]))["model-compiler"]
lane = entry[sys.argv[2]]
print(entry["repository"])
print(lane["ref"])
print(lane["spec"])
PY
)
repository="${compiler[0]}"
ref="${compiler[1]}"
spec="${compiler[2]}"
requested="${repository}@${ref}:${spec}"

"$root/scripts/ci/install_sima_cli_main.sh"
sima_cli="${HOME}/.sima-cli/.venv/bin/sima-cli"
"$sima_cli" neat install --env "${NEAT_VULCAN_ENV:?}" --json "$requested" \
  > "$output/resolution.json"
"$sima_cli" neat install --env "$NEAT_VULCAN_ENV" "$requested"

compiler_dir=""
for candidate in /sdk-extensions/model-compiler /sdk-add-on/model-compiler \
                 "$HOME/sdk-extensions/model-compiler"; do
  if [[ -f "$candidate/bin/activate" ]]; then
    compiler_dir="$candidate"
    break
  fi
done
[[ -n "$compiler_dir" ]] || { echo "Model Compiler installation not found" >&2; exit 1; }

export POLIMA_COMPILER_ROOT="$root"
export POLIMA_COMPILER_OUTPUT="$output/smoke"
export POLIMA_COMPILER_RESOLUTION="$output/resolution.json"
export MODEL_COMPILER_DIR="$compiler_dir"
bash --noprofile --rcfile "$HOME/.bashrc" -i <<'BASH'
set -euo pipefail
activate-model-compiler
test "$VIRTUAL_ENV" = "$MODEL_COMPILER_DIR"
python - <<'PY'
import afe
import importlib.metadata
print("sima-frontend", importlib.metadata.version("sima-frontend"), afe.__file__)
PY
PYTHONPATH="$POLIMA_COMPILER_ROOT/src" python \
  "$POLIMA_COMPILER_ROOT/scripts/ci/model_compiler_smoke.py" \
  --output "$POLIMA_COMPILER_OUTPUT" --resolution "$POLIMA_COMPILER_RESOLUTION"
BASH
