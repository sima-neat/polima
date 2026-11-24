#!/usr/bin/env bash
# Install on the DevKit the exact LLiMa a PoLiMa package was built against.
#
# Usage: install_exact_llima.sh LOCK_FILE WORK_DIR
#
# The shared DevKit pool carries whatever LLiMa build last ran on each board
# (LLiMa's own CI installs its branch builds system-wide), so qualification
# cannot assume the release is present. This reads a package's
# resolved-dependencies.lock.json and installs that LLiMa's root artifact with
# sima-cli. LLiMa's installer simulates the apt transaction, refuses to remove
# packages, allows the downgrade from a branch build, and checks the installed
# versions. Nothing is changed when core, cli and dev already match.
#
# LLiMa's installer uses passwordless sudo, else DEVKIT_PASSWORD, else the DevKit default.
set -euo pipefail

lock="${1:?resolved-dependencies.lock.json of the package}"
work="${2:?work directory}"
[[ -f "$lock" ]] || { echo "no LLiMa lock at $lock" >&2; exit 1; }

read -r ref spec env version < <(python3 - "$lock" <<'PY'
import json, sys
llima = json.load(open(sys.argv[1]))["llima"]
print(llima["ref"], llima["resolved_spec"], llima["environment"], llima["version"])
PY
)
target="llima@$ref:$spec"

matches=1
for name in sima-lmm-core sima-lmm-cli sima-lmm-dev; do
    installed="$(dpkg-query -W -f='${Version}' "$name" 2>/dev/null || true)"
    echo "installed $name: ${installed:-<missing>}"
    [[ "$installed" == "$version" ]] || matches=0
done
if [[ "$matches" == 1 ]]; then
    echo "LLiMa $version is already installed; nothing to change"
    exit 0
fi

echo "Installing LLiMa $version from $target (Vulcan env $env)"
for attempt in 1 2 3; do
    if SIMA_CLI_CHECK_FOR_UPDATE=0 sima-cli neat install --env "$env" --json "$target" >/dev/null; then
        break
    fi
    (( attempt < 3 )) || { echo "could not resolve $target" >&2; exit 1; }
    sleep $((attempt * 10))
done
rm -rf -- "$work"
mkdir -p "$work"
SIMA_CLI_CHECK_FOR_UPDATE=0 sima-cli neat install --env "$env" -d "$work" "$target"
