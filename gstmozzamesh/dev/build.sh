#!/bin/bash
# Development build of mozza_mesh with g++, in the mozza-mesh-dev image
# (gstmozzamesh/dev/Dockerfile, same base image as the official build).
# Uses the same OpenCV (Debian 4.6, /usr/include/opencv4) and GStreamer
# (/opt/gstreamer) as the official Bazel build. Run from the repo root:
#
#   docker build --platform linux/amd64 -t mozza-mesh-dev gstmozzamesh/dev
#   docker run --rm --platform linux/amd64 -v "$PWD":/src -w /src mozza-mesh-dev gstmozzamesh/dev/build.sh
#
# Outputs gstmozzamesh/out/libgstmozzamesh.so and out/facewarp_cli.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p out
CV_INC="-I/usr/include/opencv4"
CV_LIB="-L/usr/lib/x86_64-linux-gnu -l:libopencv_core.so.406 -l:libopencv_imgproc.so.406"
GLIB_INC="$(pkg-config --cflags glib-2.0) -I/opt/gstreamer/include/json-glib-1.0"
GLIB_LIB="-L/opt/gstreamer/lib/x86_64-linux-gnu -ljson-glib-1.0 $(pkg-config --libs gobject-2.0 glib-2.0)"
GST_INC="-I/opt/gstreamer/include/gstreamer-1.0"
GST_LIB="-lgstvideo-1.0 -lgstbase-1.0 -lgstreamer-1.0"
CXXFLAGS="-std=c++17 -O3 -fPIC -Wall -Wextra"

g++ $CXXFLAGS $CV_INC $GLIB_INC facewarp.cpp tools/facewarp_cli.cpp -o out/facewarp_cli \
    $CV_LIB -l:libopencv_imgcodecs.so.406 $GLIB_LIB
g++ $CXXFLAGS -shared $CV_INC $GLIB_INC $GST_INC -I../gstshared -DPACKAGE='"mozza_mesh"' \
    gstmozzamesh.cpp facewarp.cpp ../gstshared/mp_runtime_loader.cc \
    -o out/libgstmozzamesh.so -Wl,-z,defs $CV_LIB $GST_LIB $GLIB_LIB -ldl
echo "built: $(ls out)"
