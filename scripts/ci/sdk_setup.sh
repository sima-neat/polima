#!/usr/bin/env bash
# Set up the Vulcan runner's SDK container from its one tagged sima-neat SDK image.
set -euo pipefail
mapfile -t sdk_images < <(
  docker images --format '{{.Repository}}:{{.Tag}}' |
    awk '/^ghcr[.]io\/sima-neat\/sdk/ && $0 !~ /:<none>$/ { print }'
)
if [[ "${#sdk_images[@]}" -ne 1 ]]; then
  printf 'Expected exactly one tagged SDK image; found %d:\n' "${#sdk_images[@]}" >&2
  printf '  %s\n' "${sdk_images[@]}" >&2
  exit 1
fi
/mnt/vulcan/bin/sima-cli sdk setup -y -n --minimal \
  --workspace /mnt/vulcan/workspace --no-model-sdk --no-insight \
  --image "${sdk_images[0]}"
