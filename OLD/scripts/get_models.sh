# scripts/get_models.sh
#!/usr/bin/env bash
set -euo pipefail
OUTDIR="${1:-/opt/models}"
mkdir -p "${OUTDIR}"
echo "Downloading face_landmarker.task to ${OUTDIR}"
curl -L -o "${OUTDIR}/face_landmarker.task" \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task