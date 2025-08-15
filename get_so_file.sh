#!/usr/bin/env bash
# save as: copy_plugins_from_image.sh
# usage: ./copy_plugins_from_image.sh [image[:tag]] [dest-dir]
#example : ./get_so_file.sh mozzamp out
set -euo pipefail

IMG="${1:-mozza:latest}"      # whatever you tagged your build
DEST="${2:-./mp-out}"         # where to dump the artifacts

CID="$(docker create "$IMG" true)"
trap 'docker rm -f "$CID" >/dev/null 2>&1 || true' EXIT

mkdir -p "$DEST/plugins" "$DEST/lib"

# Copy the plugins and the (stub) runtime out of the image
docker cp "$CID":/out/plugins/libgstfacelandmarks.so "$DEST/plugins/" || true
docker cp "$CID":/out/plugins/libgstmozzamp.so       "$DEST/plugins/" || true
docker cp "$CID":/out/lib/libmp_runtime.so           "$DEST/lib/"      || true

echo "Wrote:"
ls -l "$DEST/plugins" "$DEST/lib" 2>/dev/null || true