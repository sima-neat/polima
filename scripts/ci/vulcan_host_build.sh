#!/usr/bin/env bash
# Vulcan build_command (runs on the Vulcan runner, in /mnt/vulcan/workspace):
# select the SDK container the way LLiMa's vulcan-ci.yml does, configure only
# that disposable container, and run one lane inside it.
#
# usage: vulcan_host_build.sh release|develop
set -euo pipefail

lane="${1:?release or develop}"
sdk_container="$(docker ps --filter 'ancestor=ghcr.io/sima-neat/sdk:latest' --format '{{.Names}}' | head -n 1)"
if [[ -z "$sdk_container" ]]; then
  sdk_container="$(docker ps --filter 'name=sdk' --format '{{.Names}}' | head -n 1)"
fi
if [[ -z "$sdk_container" ]]; then
  docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
  echo "Could not find running SDK container." >&2
  exit 1
fi
"${SDK_PACKAGE_HELPER:?Shared SDK package helper required}" "$sdk_container" configure

# The workspace copy has no .git; take the commit time from the checkout.
epoch="$(git -C "${GITHUB_WORKSPACE:?}" log -1 --format=%ct)"
docker exec \
  -e GITHUB_SHA -e GITHUB_REF_NAME -e GITHUB_REF_TYPE -e GITHUB_HEAD_REF \
  -e SYSROOT_VERSION -e NEAT_VULCAN_ENV -e SOURCE_DATE_EPOCH="$epoch" \
  -e SIMA_CLI_CHECK_FOR_UPDATE=0 \
  "$sdk_container" bash -lc '
    set -euo pipefail
    /workspace/scripts/ci/install_sima_cli_main.sh
    export PATH="$HOME/.sima-cli/.venv/bin:$PATH"
    sima-cli --version
    /workspace/scripts/ci/vulcan_build.sh '"$lane"'
  '
