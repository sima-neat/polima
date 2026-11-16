#!/usr/bin/env bash
# Installation entry point of the published PoLiMa package. `sima-cli neat
# install polima@<ref>` downloads the package, verifies every file against its
# metadata.json, and runs this script from the download folder.
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
shopt -s nullglob
runtime=("$here"/polima-runtime-*-modalix.tar.gz)
wheel=("$here"/polima-*.whl)
source=("$here"/polima-source-*.tar.gz)
if (( ${#runtime[@]} != 1 || ${#wheel[@]} != 1 || ${#source[@]} != 1 )); then
  echo "incomplete PoLiMa package in $here" >&2
  exit 1
fi

source_dir="$here/polima-source"
extracted_dirs=("${runtime[0]%.tar.gz}" "${source[0]%.tar.gz}")
cleanup() { rm -rf -- "$source_dir" "${extracted_dirs[@]}"; }
trap cleanup EXIT
cleanup
mkdir "$source_dir"
tar -xzf "${source[0]}" -C "$source_dir" --no-same-owner
POLIMA_SOURCE="$source_dir" \
POLIMA_RUNTIME_ARCHIVE="${runtime[0]}" \
POLIMA_WHEEL="${wheel[0]}" \
  bash "$here/polima-source/scripts/install_polima_modalix.sh" "$@"
