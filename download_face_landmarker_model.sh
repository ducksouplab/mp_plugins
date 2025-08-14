#!/usr/bin/env bash
# download_face_landmarker_model.sh

set -euo pipefail

# Default download URL (Float16 v1)
DEFAULT_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

MODEL_URL="${1:-$DEFAULT_URL}"
DEST_DIR="${2:-.}"
DEST="${DEST_DIR}/$(basename "$MODEL_URL")"

echo "Downloading Face Landmarker model..."
echo "URL:      $MODEL_URL"
echo "To file:  $DEST"

mkdir -p "$DEST_DIR"
curl -L --progress-bar --output "$DEST" "$MODEL_URL"

echo "Download complete: $DEST"