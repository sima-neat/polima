#!/usr/bin/env bash
# Assemble the publishable PoLiMa package: the Vulcan-built runtime, the wheel,
# the source tree the DevKit installer needs, the exact dependency lock and a
# sima-cli metadata.json (written by `sima-cli packages build`, as LLiMa does).
#
# usage: package_polima.sh LOCK_JSON RUNTIME_DIR OUTPUT_DIR
#   POLIMA_VERSION  package version (see package_version.py)
#   GITHUB_SHA, GITHUB_REF_NAME/GITHUB_HEAD_REF  recorded in metadata.json
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
lock="${1:?lock file}"
runtime_dir="${2:?runtime directory}"
output="${3:?output directory}"
version="${POLIMA_VERSION:?set POLIMA_VERSION (scripts/ci/package_version.py)}"
commit="${GITHUB_SHA:-$(git -C "$root" rev-parse HEAD)}"
branch="${GITHUB_HEAD_REF:-${GITHUB_REF_NAME:-$(git -C "$root" branch --show-current)}}"
sima_cli="${SIMA_CLI_BIN:-sima-cli}"
python="${POLIMA_PYTHON:-python3}"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git -C "$root" log -1 --format=%ct 2>/dev/null || echo 0)}"

[[ ! -e "$output" ]] || { echo "$output already exists" >&2; exit 2; }
work="$(mktemp -d "${RUNNER_TEMP:-/tmp}/polima-package.XXXXXX")"
trap 'rm -rf "$work"' EXIT
mkdir -p "$output"

archive="polima-runtime-$version-modalix.tar.gz"
install -m 0644 "$runtime_dir/$archive" "$runtime_dir/$archive.sha256" "$output/"
(cd "$output" && sha256sum --check --strict --quiet "$archive.sha256")

# The tracked source tree (CI checkouts have no .git: every file is tracked),
# without the documentation images (17 MB the DevKit never uses).
if git -C "$root" rev-parse --git-dir >/dev/null 2>&1; then
  git -C "$root" ls-files -z --cached --others --exclude-standard
else
  (cd "$root" && find . -type f ! -path './dist/*' ! -path './.git/*' -printf '%P\0')
fi | grep -zv '^docs/images/' | LC_ALL=C sort -z > "$work/files"
tar -C "$root" --null -T "$work/files" --sort=name --mtime="@$SOURCE_DATE_EPOCH" \
  --owner=0 --group=0 --numeric-owner -cf - | gzip -n > "$output/polima-source-$version.tar.gz"

# Build the wheel from an unpacked copy so the checkout stays untouched.
mkdir "$work/src"
tar -xzf "$output/polima-source-$version.tar.gz" -C "$work/src"
"$python" -m pip wheel --quiet --no-deps --wheel-dir "$work/wheel" "$work/src"
install -m 0644 "$work"/wheel/polima-*.whl "$output/"

install -m 0644 "$lock" "$output/resolved-dependencies.lock.json"
install -m 0644 "$root/deps/manifest.json" "$output/polima-package-manifest.json"
install -m 0755 "$root/scripts/ci/install_polima.sh" "$output/install_polima.sh"

platform="$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["platform-version"])' \
  "$root/deps/manifest.json")"
SIMA_CLI_CHECK_FOR_UPDATE=0 "$sima_cli" packages build "$output" \
  --name gh:sima-neat/polima --version "$version" \
  --description "PoLiMa robot-policy runtime for SiMa.ai Modalix" \
  --install-script install_polima.sh \
  --board-platform "modalix@==$platform"
"$python" - "$output/metadata.json" "$commit" "$branch" <<'PY'
import json, sys
path, commit, branch = sys.argv[1:]
metadata = json.load(open(path, encoding="utf-8"))
metadata.update({
    "repository": "sima-neat/polima",
    "branch": branch,
    "commit": commit,
    "commit_folder": commit[:12],
    "artifact": {"type": "devkit-package", "repository": "polima",
                 "package_path": ".", "profile": "devkit-install"},
})
open(path, "w", encoding="utf-8").write(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
PY
"$python" "$root/scripts/ci/verify_package.py" "$output" --commit "$commit" --branch "$branch"
