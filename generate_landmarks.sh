#!/usr/bin/env bash
# generate_landmarks.sh
# Runs both CPU and GPU plugins on a test image, exports 478 landmarks per frame,
# then compares them with compare_landmarks.py.
#
# Usage:
#   ./generate_landmarks.sh [image] [face_task] [frames]
#
# Defaults:
#   image      = assets/test_image.jpg
#   face_task  = env/face_landmarker.task
#   frames     = 30  (duplicate the still image this many times for stable detection)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="${1:-$SCRIPT_DIR/assets/test_image.jpg}"
FRAMES="${2:-30}"

CPU_OUT="$SCRIPT_DIR/landmarks_cpu.txt"
GPU_OUT="$SCRIPT_DIR/landmarks_gpu.txt"

# Base image (has GStreamer + CPU plugin only; no CUDA runtime)
BASE_IMAGE="ducksouplab/debian-gstreamer:deb12-with-plugins-cuda12.2-gst1.28.0"
BASE_PLUGIN_PATH="/work/mp-out/plugins"
BASE_LIB_PATH="/work/mp-out/lib:/opt/gstreamer/lib/x86_64-linux-gnu"

# Test image (has CUDA + TensorRT runtime; plugins pre-installed inside image)
# Rebuilt with: DOCKER_BUILDKIT=1 docker build -f Dockerfile.test -t mp_plugins_test:fixed \
#   --build-context mp_plugins_gpu=docker-image://mp_plugins_gpu:fixed .
TEST_IMAGE="mp_plugins_test:fixed"

# ── helpers ──────────────────────────────────────────────────────────────────

die() { echo "ERROR: $*" >&2; exit 1; }

check_prereqs() {
  [ -f "$IMAGE" ] || die "Test image not found: $IMAGE"
  [ -f "$SCRIPT_DIR/env/face_landmarker.task" ] || die "Model not found: env/face_landmarker.task"
  docker info &>/dev/null || die "Docker not available"
  [ -f "$SCRIPT_DIR/mp-out/plugins/libgstmozzamp.so" ]     || die "libgstmozzamp.so missing — rebuild first"
  [ -f "$SCRIPT_DIR/mp-out/plugins/libgstmozzamp_gpu.so" ] || die "libgstmozzamp_gpu.so missing — rebuild first"
  docker image inspect "$TEST_IMAGE" &>/dev/null || die "$TEST_IMAGE not found — rebuild with Dockerfile.test"
}

run_cpu_plugin() {
  local out_file="$1"
  echo "→ Running mozza_mp / CPU (${FRAMES} frames)..."
  rm -f "$out_file"

  docker run --rm \
    -v "$SCRIPT_DIR:/work" \
    -v "$IMAGE:/work_image:ro" \
    -e GST_PLUGIN_PATH="$BASE_PLUGIN_PATH" \
    -e LD_LIBRARY_PATH="$BASE_LIB_PATH" \
    -e LANDMARK_OUTPUT_FILE="/work/$(basename "$out_file")" \
    -e GST_DEBUG="2" \
    "$BASE_IMAGE" \
    gst-launch-1.0 -q \
    filesrc location=/work_image \
    '!' jpegdec \
    '!' imagefreeze num-buffers=${FRAMES} \
    '!' "video/x-raw,framerate=30/1" \
    '!' videoconvert \
    '!' "video/x-raw,format=RGBA" \
    '!' mozza_mp \
        model="/work/env/face_landmarker.task" \
        no-warp=true \
    '!' fakesink sync=false

  [ -f "$out_file" ] || die "mozza_mp produced no landmark file"
  local nframes
  nframes=$(grep -c "^Frame" "$out_file" 2>/dev/null || echo 0)
  echo "  ✓  $nframes frame(s) written to $(basename "$out_file")"
}

run_gpu_plugin() {
  local out_file="$1"
  # mozza_mp_gpu needs TensorRT/CUDA runtime; use the pre-built test image.
  # The test image has plugins in /usr/local/lib/gstreamer-1.0 and LD paths set.
  # We still mount $SCRIPT_DIR for model files and the output landmark file.
  echo "→ Running mozza_mp_gpu / GPU (${FRAMES} frames)..."
  rm -f "$out_file"

  # mozza_mp_gpu uses model= property (path to face_landmarker.task);
  # ONNX models (face_detector.onnx, face_landmarks.onnx) must be in same dir.
  local model_path="/work/env/face_landmarker.task"

  docker run --rm --gpus all \
    -v "$SCRIPT_DIR:/work" \
    -v "$IMAGE:/work_image:ro" \
    -e LANDMARK_OUTPUT_FILE="/work/$(basename "$out_file")" \
    -e GST_DEBUG="2" \
    "$TEST_IMAGE" \
    gst-launch-1.0 -q \
    filesrc location=/work_image \
    '!' jpegdec \
    '!' imagefreeze num-buffers=${FRAMES} \
    '!' "video/x-raw,framerate=30/1" \
    '!' videoconvert \
    '!' "video/x-raw,format=RGBA" \
    '!' mozza_mp_gpu \
        model="$model_path" \
        no-warp=true \
    '!' fakesink sync=false

  [ -f "$out_file" ] || die "mozza_mp_gpu produced no landmark file"
  local nframes
  nframes=$(grep -c "^Frame" "$out_file" 2>/dev/null || echo 0)
  echo "  ✓  $nframes frame(s) written to $(basename "$out_file")"
}

# ── main ─────────────────────────────────────────────────────────────────────

check_prereqs

echo "=== Landmark Generation ==="
echo "  Image:  $IMAGE"
echo "  Frames: $FRAMES"
echo ""

run_cpu_plugin "$CPU_OUT"
run_gpu_plugin "$GPU_OUT"

echo ""
echo "=== Comparison ==="
cd "$SCRIPT_DIR"
python3 compare_landmarks.py
