#!/usr/bin/env bash
# Install on the DevKit the exact LLiMa a PoLiMa package was built against.
#
# Usage: install_exact_llima.sh LOCK_FILE WORK_DIR [MANIFEST]
#
# The shared DevKit pool carries whatever LLiMa and NEAT Core builds last ran on
# each board (their own CI installs branch builds system-wide), so qualification
# cannot assume the release is present. This reads a package's
# resolved-dependencies.lock.json and:
#
# - changes nothing when sima-lmm-core, -cli and -dev already match;
# - when installed NEAT Core (sima-neat*) builds pin a different LLiMa, reinstalls
#   the NEAT Core release pinned in deps/manifest.json ("neat-core"), which ships
#   its matching LLiMa, so the board is left with a coherent NEAT and LLiMa;
# - otherwise installs the LLiMa root artifact with sima-cli.
#
# Both installers simulate the apt transaction first and refuse unexpected removals.
# They use passwordless sudo, else DEVKIT_PASSWORD, else the DevKit default.
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lock="${1:?resolved-dependencies.lock.json of the package}"
work="${2:?work directory}"
manifest="${3:-$here/../../deps/manifest.json}"
[[ -f "$lock" ]] || { echo "no LLiMa lock at $lock" >&2; exit 1; }

read -r ref spec env version < <(python3 - "$lock" <<'PY'
import json, sys
llima = json.load(open(sys.argv[1]))["llima"]
print(llima["ref"], llima["resolved_spec"], llima["environment"], llima["version"])
PY
)
target="llima@$ref:$spec"

llima_matches() {
    local name installed status=0
    for name in sima-lmm-core sima-lmm-cli sima-lmm-dev; do
        installed="$(dpkg-query -W -f='${Version}' "$name" 2>/dev/null || true)"
        echo "installed $name: ${installed:-<missing>}"
        [[ "$installed" == "$version" ]] || status=1
    done
    return "$status"
}

cli() { SIMA_CLI_CHECK_FOR_UPDATE=0 sima-cli "$@"; }

if llima_matches; then
    echo "LLiMa $version is already installed; nothing to change"
    exit 0
fi

mapfile -t blocking < <(python3 "$here/llima_conflicts.py" --target "$version")
if (( ${#blocking[@]} )); then
    for row in "${blocking[@]}"; do
        IFS=$'\t' read -r name installed relation <<< "$row"
        echo "$name $installed requires $relation, which LLiMa $version does not satisfy"
        [[ "$name" == sima-neat* ]] ||
            { echo "refusing to replace $name; only NEAT Core (sima-neat*) is reinstalled" >&2; exit 1; }
    done
    read -r core_ref core_spec < <(python3 - "$manifest" <<'PY'
import json, sys
core = json.load(open(sys.argv[1]))["neat-core"]
print(core["ref"], core["spec"])
PY
)
    echo "Reinstalling NEAT Core core@$core_ref:$core_spec with its matching LLiMa"
    rm -rf -- "$work/neat-core"
    mkdir -p "$work/neat-core"
    # A CI board: no agent skills in the runner's home and no paired-DevKit sync.
    NEAT_INSTALLER_INSTALL_CODEX_SKILL=OFF NEAT_INSTALLER_INSTALL_CLAUDE_SKILL=OFF \
    NEAT_INSTALLER_SKIP_DEVKIT_SYNC=ON \
        cli install --neat --env "$env" -d "$work/neat-core" "core@$core_ref:$core_spec" -t all
    if llima_matches; then
        echo "LLiMa $version installed with NEAT Core $core_ref"
        exit 0
    fi
    echo "NEAT Core $core_ref did not bring LLiMa $version; installing $target"
fi

echo "Installing LLiMa $version from $target (Vulcan env $env)"
for attempt in 1 2 3; do
    if cli neat install --env "$env" --json "$target" >/dev/null; then
        break
    fi
    (( attempt < 3 )) || { echo "could not resolve $target" >&2; exit 1; }
    sleep $((attempt * 10))
done
rm -rf -- "$work/llima"
mkdir -p "$work/llima"
cli neat install --env "$env" -d "$work/llima" "$target"
