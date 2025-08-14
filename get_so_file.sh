#!/usr/bin/env bash
# save as: copy_plugin_from_image.sh
set -euo pipefail
IMG="${1:-gst-facelandmarks:ducksoup}"
DEST="${2:-./dist}"
SRC="/app/plugins/libgstfacelandmarks.so"

mkdir -p "$DEST"
CID="$(docker create "$IMG" true)"
trap 'docker rm -f "$CID" >/dev/null 2>&1 || true' EXIT

docker cp "$CID":"$SRC" "$DEST"/libgstfacelandmarks.so
echo "Copied to $DEST/libgstfacelandmarks.so"