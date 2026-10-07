#!/usr/bin/env bash
# Copy the built plugins and runtime library out of the plugins image.
# usage: ./scripts/get_so_file.sh [image[:tag]] [dest-dir]   (default dest: mp-out/ in the repository)
set -euo pipefail

IMG="${1:-mp_plugins:latest}"      # whatever you tagged your build
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${2:-$ROOT_DIR/mp-out}"  # where to dump the artifacts

CID="$(docker create "$IMG" true)"
trap 'docker rm -f "$CID" >/dev/null 2>&1 || true' EXIT

mkdir -p "$DEST/plugins" "$DEST/lib"

# Copy the plugins and the (stub) runtime out of the image
docker cp "$CID":/usr/local/lib/gstreamer-1.0/libgstfacelandmarks.so "$DEST/plugins/" || true
docker cp "$CID":/usr/local/lib/gstreamer-1.0/libgstmozzamp.so       "$DEST/plugins/" || true
docker cp "$CID":/usr/local/lib/gstreamer-1.0/libgstmozzamp_gpu.so   "$DEST/plugins/" || true
docker cp "$CID":/usr/local/lib/gstreamer-1.0/libgstmozzamesh.so     "$DEST/plugins/" || true
docker cp "$CID":/usr/local/lib/libmp_runtime.so                     "$DEST/lib/"      || true

echo "Wrote:"
ls -l "$DEST/plugins" "$DEST/lib" 2>/dev/null || true