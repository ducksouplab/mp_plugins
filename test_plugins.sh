#!/bin/bash
# test_plugins.sh - Automated verification for Mediapipe GStreamer plugins
#
# Usage:
#   ./test_plugins.sh [plugin_path]
#
# If plugin_path is omitted, it assumes they are in the default GStreamer path
# or GST_PLUGIN_PATH is already set.

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

MODEL="face_landmarker.task"
INPUT="test_face.jpg"
DEFORM="smile.dfm"

if [ ! -f "$MODEL" ]; then
    echo -e "${RED}Error: $MODEL not found. Run download_face_landmarker_model.sh first.${NC}"
    exit 1
fi

if [ ! -f "$INPUT" ]; then
    echo -e "${RED}Error: $INPUT not found.${NC}"
    exit 1
fi

if [ ! -f "$DEFORM" ]; then
    echo -e "${RED}Error: $DEFORM not found.${NC}"
    exit 1
fi

# Set GST_PLUGIN_PATH if a path was provided
if [ ! -z "$1" ]; then
    export GST_PLUGIN_PATH=$1:$GST_PLUGIN_PATH
fi

echo "--- GStreamer Plugin Verification ---"

# 1. Inspect plugins
for p in facelandmarks mozza_mp mozza_mp_gpu; do
    echo -n "Checking $p... "
    if gst-inspect-1.0 "$p" > /dev/null 2>&1; then
        echo -e "${GREEN}FOUND${NC}"
    else
        echo -e "${RED}NOT FOUND${NC}"
        # Don't exit yet, might be expected if GPU is missing
    fi
done

echo ""
echo "--- Running Functional Tests ---"

# Test facelandmarks (CPU)
echo -n "Testing facelandmarks (CPU)... "
gst-launch-1.0 -q filesrc location="$INPUT" ! jpegdec ! videoconvert ! video/x-raw,format=RGBA ! \
  facelandmarks model="$MODEL" ! \
  videoconvert ! pngenc ! filesink location="test_out_landmarks.png"
if [ -s "test_out_landmarks.png" ]; then
    echo -e "${GREEN}PASSED${NC} (output: test_out_landmarks.png)"
else
    echo -e "${RED}FAILED${NC}"
fi

# Test mozza_mp (CPU)
echo -n "Testing mozza_mp (CPU)... "
gst-launch-1.0 -q filesrc location="$INPUT" ! jpegdec ! videoconvert ! video/x-raw,format=RGBA ! \
  mozza_mp model="$MODEL" deform="$DEFORM" alpha=2.0 ! \
  videoconvert ! pngenc ! filesink location="test_out_mozza_cpu.png"
if [ -s "test_out_mozza_cpu.png" ]; then
    echo -e "${GREEN}PASSED${NC} (output: test_out_mozza_cpu.png)"
else
    echo -e "${RED}FAILED${NC}"
fi

# Test mozza_mp_gpu (GPU)
echo -n "Testing mozza_mp_gpu (GPU)... "
# Check if GPU is available
if nvidia-smi > /dev/null 2>&1; then
    # Ensure ONNX models exist
    if [ ! -f "face_detector.onnx" ] || [ ! -f "face_landmarks.onnx" ]; then
        echo -e "${RED}FAILED${NC} (Missing ONNX models. Run python3 convert_models.py $MODEL)"
    else
        gst-launch-1.0 -q filesrc location="$INPUT" ! jpegdec ! videoconvert ! video/x-raw,format=RGBA ! \
          mozza_mp_gpu model_path="$MODEL" deform="$DEFORM" alpha=2.0 ! \
          videoconvert ! pngenc ! filesink location="test_out_mozza_gpu.png"
        if [ -s "test_out_mozza_gpu.png" ]; then
            echo -e "${GREEN}PASSED${NC} (output: test_out_mozza_gpu.png)"
        else
            echo -e "${RED}FAILED${NC}"
        fi
    fi
else
    echo -e "\033[0;33mSKIPPED\033[0m (No NVIDIA GPU detected)"
fi

echo ""
echo "Verification complete."
