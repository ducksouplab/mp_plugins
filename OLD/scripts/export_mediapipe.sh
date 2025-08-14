#!/usr/bin/env bash
set -euo pipefail

# ---- Config ----
MP_REF="${MEDIAPIPE_REF:-v0.10.14}"
SRC_DIR="/opt/gst-facelandmarks/mediapipe-src"
OUT_DIR="/opt/gst-facelandmarks/third_party/mediapipe-export"
BAZEL_BIN="${BAZEL_BIN:-/usr/local/bin/bazel}"

echo "== Exporting MediaPipe @ ${MP_REF} with Bazel =="

# Minimal build deps used *only* for exporting headers/libs
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
  python3 python3-numpy git rsync pkg-config ca-certificates \
  libopencv-dev libgoogle-glog-dev libgflags-dev libeigen3-dev

mkdir -p "$(dirname "$SRC_DIR")" "$OUT_DIR"

# Checkout MediaPipe at the requested ref
if [ ! -d "$SRC_DIR/.git" ]; then
  git clone https://github.com/google/mediapipe.git "$SRC_DIR"
fi
git -C "$SRC_DIR" fetch --tags origin "$MP_REF" --depth 1
git -C "$SRC_DIR" checkout -f "$MP_REF"

# Ensure Bazel 6.1.1 is present (known-good for MP 0.10.x)
if ! command -v "$BAZEL_BIN" >/dev/null 2>&1; then
  curl -fsSL -o /usr/local/bin/bazel \
    https://releases.bazel.build/6.1.1/release/bazel-6.1.1-linux-x86_64
  chmod +x /usr/local/bin/bazel
  BAZEL_BIN=/usr/local/bin/bazel
fi

cd "$SRC_DIR"

echo ">> Building MediaPipe targets (this will take a bit)…"
"$BAZEL_BIN" build -c opt --cxxopt='-std=c++17' \
  //mediapipe/tasks/cc/vision/face_landmarker:face_landmarker \
  //mediapipe/tasks/metadata:metadata_schema_cc_srcs

# --- HERE IS THE IMPORTANT CHANGE ---
# Ask Bazel for the ABSOLUTE execution root so we don't depend on relative symlinks.
EXEC_ROOT="$("$BAZEL_BIN" info execution_root)"
EXTERNAL_DIR="${EXEC_ROOT}/external"
echo ">> Bazel execroot: ${EXEC_ROOT}"

# Resolve the FlatBuffers external repo (name varies across TF/MP)
FB_INC=""
for cand in \
  "${EXTERNAL_DIR}/flatbuffers/include" \
  "${EXTERNAL_DIR}/com_google_flatbuffers/include" \
  "${EXTERNAL_DIR}/com_github_google_flatbuffers/include"
do
  if [ -d "${cand}/flatbuffers" ]; then
    FB_INC="$cand"
    break
  fi
done
if [ -z "$FB_INC" ]; then
  echo ">> Could not locate FlatBuffers headers under ${EXTERNAL_DIR}" >&2
  ls -1 "${EXTERNAL_DIR}" | sed 's/^/   - /' >&2 || true
  exit 5
fi
echo ">> Found FlatBuffers include at: ${FB_INC}"

# Clean and (re)create export dir
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/include" "$OUT_DIR/lib"

echo ">> Staging into ${OUT_DIR}"

# Abseil headers
rsync -a "${EXTERNAL_DIR}/com_google_absl/absl"                "${OUT_DIR}/include/"

# Protobuf headers (google/*)
rsync -a "${EXTERNAL_DIR}/com_google_protobuf/src/google"      "${OUT_DIR}/include/"

# FlatBuffers headers that MATCH the generated code version
rsync -a "${FB_INC}/flatbuffers"                                "${OUT_DIR}/include/"

# Eigen headers (system)
rsync -a /usr/include/eigen3/Eigen                              "${OUT_DIR}/include/" || true

# MediaPipe public headers
rsync -a mediapipe                                             "${OUT_DIR}/include/"

# TensorFlow Lite public headers
rsync -a "${EXTERNAL_DIR}/org_tensorflow/tensorflow/lite"       "${OUT_DIR}/include/tensorflow/"

# Libraries (PIC static archives we’ll link against)
find bazel-bin/mediapipe/tasks/cc/vision/face_landmarker \
  -maxdepth 1 -name '*.pic.a' -exec cp -t "${OUT_DIR}/lib" {} +

echo ">> Sanity checks"
test -f "${OUT_DIR}/include/flatbuffers/flatbuffers.h"
test -f "${OUT_DIR}/include/tensorflow/lite/schema/schema_generated.h"
test -f "${OUT_DIR}/include/mediapipe/tasks/metadata/metadata_schema_generated.h"

echo ">> Export complete."