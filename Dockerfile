# Build MediaPipe Mozza MP (MLS deformation) plugin.
# Produces /app/plugins/libgstmozzamp.so

FROM ducksouplab/debian-gstreamer:deb12-cuda12.2-plugins-gst1.24.10 AS builder

ARG MEDIAPIPE_TAG=v0.10.26
ARG BAZEL_VERSION=6.5.0
ENV DEBIAN_FRONTEND=noninteractive

# Toolchain + OpenCV (imgwarp needs core/imgproc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git pkg-config build-essential \
    python3 python3-numpy libglib2.0-dev \
    libopencv-core-dev libopencv-imgproc-dev \
 && rm -rf /var/lib/apt/lists/*

# Bazel
RUN curl -fsSL -o /usr/local/bin/bazel \
    https://releases.bazel.build/${BAZEL_VERSION}/release/bazel-${BAZEL_VERSION}-linux-x86_64 \
 && chmod +x /usr/local/bin/bazel

WORKDIR /opt/gst-mozzamp

# MediaPipe sources
RUN git clone --depth=1 --branch ${MEDIAPIPE_TAG} \
    https://github.com/google-ai-edge/mediapipe.git mediapipe-src

# -----------------------------------------------------------------------------
# Bring in your plugin sources (must exist in build context)
# -----------------------------------------------------------------------------
COPY src/gstmozzamp.cpp            mediapipe-src/gstmozzamp/gstmozzamp.cpp
COPY src/dfm.hpp                   mediapipe-src/gstmozzamp/dfm.hpp
COPY src/dfm.cpp                   mediapipe-src/gstmozzamp/dfm.cpp
COPY src/deform_utils.hpp          mediapipe-src/gstmozzamp/deform_utils.hpp
COPY src/deform_utils.cpp          mediapipe-src/gstmozzamp/deform_utils.cpp
# Already unzipped MLS library:
COPY imgwarp/                      mediapipe-src/gstmozzamp/imgwarp/

# -----------------------------------------------------------------------------
# Vendor GStreamer/GLib headers into a Bazel-friendly package
# -----------------------------------------------------------------------------
RUN bash -eux <<'BASH'
cd mediapipe-src
mkdir -p third_party/sysroot_gst/include \
         third_party/sysroot_gst/lib/glib-2.0/include

# Headers from the Ducksoup base image
cp -a /opt/gstreamer/include/gstreamer-1.0 third_party/sysroot_gst/include/
cp -a /usr/include/glib-2.0               third_party/sysroot_gst/include/
cp -a /usr/lib/x86_64-linux-gnu/glib-2.0/include/* \
      third_party/sysroot_gst/lib/glib-2.0/include/

# Define Bazel targets for GLib and GStreamer
cat > third_party/sysroot_gst/BUILD <<'EOF'
load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "glib",
    hdrs = glob([
        "include/glib-2.0/**/*.h",
        "lib/glib-2.0/include/**/*.h",
    ]),
    includes = [
        "include/glib-2.0",
        "lib/glib-2.0/include",
    ],
    linkopts = ["-lglib-2.0", "-lgobject-2.0"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "gstreamer",
    hdrs = glob(["include/gstreamer-1.0/**/*.h"]),
    includes = [
        "include/gstreamer-1.0",
        "include/glib-2.0",
        "lib/glib-2.0/include",
    ],
    deps = [":glib"],
    linkopts = ["-L/opt/gstreamer/lib/x86_64-linux-gnu",
                "-lgstvideo-1.0", "-lgstbase-1.0", "-lgstreamer-1.0"],
    visibility = ["//visibility:public"],
)
EOF
BASH

# -----------------------------------------------------------------------------
# Create the BUILD file for gstmozzamp inside the image
# Use a cc_library for headers so Bazel stages them into the sandbox.
# -----------------------------------------------------------------------------
RUN bash -eux <<'BASH'
cd mediapipe-src/gstmozzamp

cat > BUILD <<'EOF'
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary")

# imgwarp MLS library from local folder
cc_library(
    name = "imgwarp",
    srcs = glob([
        "imgwarp/**/*.cc",
        "imgwarp/**/*.cpp",
        "imgwarp/**/*.c",
    ]),
    hdrs = glob([
        "imgwarp/**/*.h",
        "imgwarp/**/*.hpp",
    ]),
    includes = ["imgwarp"],
    copts = [
        "-fPIC",
        "-O2",
        "-I/usr/include/opencv4",
    ],
    visibility = ["//visibility:public"],
)

# Core library exposing headers to dependents
cc_library(
    name = "mozzamp_core",
    srcs = [
        "dfm.cpp",
        "deform_utils.cpp",
    ],
    hdrs = [
        "dfm.hpp",
        "deform_utils.hpp",
    ],
    includes = ["."],  # allow #include "deform_utils.hpp"
    copts = [
        "-std=gnu++17",
        "-fPIC",
        "-O2",
        "-I/usr/include/opencv4",
    ],
    deps = [
        ":imgwarp",
        "@com_google_absl//absl/status:statusor",
    ],
    visibility = ["//visibility:public"],
)

# The plugin shared object
cc_binary(
    name = "libgstmozzamp.so",
    srcs = [
        "gstmozzamp.cpp",
    ],
    linkshared = 1,
    copts = [
        "-std=gnu++17",
        "-fPIC",
        "-O2",
        "-Wno-deprecated-declarations",
        "-I/usr/include/opencv4",
    ],
    deps = [
        ":mozzamp_core",
        "//third_party/sysroot_gst:gstreamer",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker",
        "@com_google_absl//absl/status:statusor",
    ],
    linkopts = [
        "-Wl,-rpath,/opt/gstreamer/lib/x86_64-linux-gnu",
        "-Wl,-rpath,/usr/local/lib",
        "-Wl,-rpath,/usr/lib/x86_64-linux-gnu",
        "-lopencv_core",
        "-lopencv_imgproc",
    ],
    visibility = ["//visibility:public"],
)
EOF
BASH

WORKDIR /opt/gst-mozzamp/mediapipe-src

# Bazel config: CPU-only, C++17, OpenCV headers available
RUN printf '%s\n' \
  'common --experimental_repo_remote_exec' \
  'common --repo_env=HERMETIC_PYTHON_VERSION=3.11' \
  'build --define MEDIAPIPE_DISABLE_GPU=1' \
  'build --define xnn_enable_avxvnni=false' \
  'build --define xnn_enable_avxvnniint8=false' \
  'build --cxxopt=-std=gnu++17 --host_cxxopt=-std=gnu++17' \
  'build --cxxopt=-I/usr/include/opencv4 --host_cxxopt=-I/usr/include/opencv4' \
  > .bazelrc

# Build and install the plugin into /app/plugins
RUN set -eux; \
    bazel clean --expunge; \
    bazel build //gstmozzamp:libgstmozzamp.so; \
    bbin="$(bazel info bazel-bin)"; \
    real="${bbin}/gstmozzamp/libgstmozzamp.so"; \
    install -D -m 0755 "$real" /app/plugins/libgstmozzamp.so; \
    # quick self-check (non-fatal)
    GST_PLUGIN_PATH=/app/plugins \
    GST_DEBUG=GST_PLUGIN_LOADING:4 \
    gst-inspect-1.0 mozza_mp || true

# -----------------------------------------------------------------------------
# Runtime stage: ship plugin + OpenCV runtime libs
# -----------------------------------------------------------------------------
FROM ducksouplab/debian-gstreamer:deb12-cuda12.2-plugins-gst1.24.10 AS runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core4.6 libopencv-imgproc4.6 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/plugins/libgstmozzamp.so /app/plugins/libgstmozzamp.so
ENV GST_PLUGIN_PATH=/app/plugins:$GST_PLUGIN_PATH