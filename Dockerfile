# Build both plugins *inside* the MediaPipe workspace so labels like
#   //mediapipe/... work and we avoid Bzlmod/local_repository mismatches.
# This Dockerfile expects these directories in your repo root:
#   gstfacelandmarks/ , gstmozzamp/ , gstshared/ , (optional) imgwarp/
# Each contains its own BUILD file.

FROM ducksouplab/debian-gstreamer:deb12-cuda12.2-plugins-gst1.24.10 AS builder

ARG MEDIAPIPE_TAG=v0.10.26
ARG BAZEL_VERSION=6.5.0
ENV DEBIAN_FRONTEND=noninteractive

# Toolchain + headers we need at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git pkg-config build-essential \
    python3 python3-numpy libglib2.0-dev \
    libopencv-dev \
    libegl1-mesa-dev libgles2-mesa-dev libgl1-mesa-dev \
 && rm -rf /var/lib/apt/lists/*

# Bazel
RUN curl -fsSL -o /usr/local/bin/bazel \
      https://releases.bazel.build/${BAZEL_VERSION}/release/bazel-${BAZEL_VERSION}-linux-x86_64 \
 && chmod +x /usr/local/bin/bazel

# 1) Clone MediaPipe (we build *inside* this workspace)
WORKDIR /opt
RUN git clone --depth=1 --branch ${MEDIAPIPE_TAG} \
      https://github.com/google-ai-edge/mediapipe.git mediapipe-src

# 2) Copy your plugin sources into the mediapipe workspace
WORKDIR /opt/mediapipe-src
COPY gstfacelandmarks/ gstfacelandmarks/
COPY gstmozzamp/       gstmozzamp/
COPY gstshared/        gstshared/
# If you keep a top-level imgwarp/ in your repo, copy it beneath gstmozzamp/
# If you already have gstmozzamp/imgwarp in your repo, comment this next line.
COPY imgwarp/          gstmozzamp/imgwarp/

# 3) Provide a tiny Bazel package with GLib/GStreamer headers so our BUILDs can depend on it
RUN bash -eux <<'BASH'
mkdir -p third_party/sysroot_gst/include \
         third_party/sysroot_gst/lib/glib-2.0/include
cp -a /opt/gstreamer/include/gstreamer-1.0 third_party/sysroot_gst/include/
cp -a /usr/include/glib-2.0               third_party/sysroot_gst/include/
cp -a /usr/lib/x86_64-linux-gnu/glib-2.0/include/* \
      third_party/sysroot_gst/lib/glib-2.0/include/
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
    linkopts = [
        "-L/opt/gstreamer/lib/x86_64-linux-gnu",
        "-lgstvideo-1.0", "-lgstbase-1.0", "-lgstreamer-1.0",
    ],
    visibility = ["//visibility:public"],
)
EOF
BASH

# 4) Bazel config (CPU and GPU + make TF repo rules happy)
RUN printf '%s\n' \
  'common --experimental_repo_remote_exec' \
  'common --repo_env=HERMETIC_PYTHON_VERSION=3.11' \
  'build --define xnn_enable_avxvnni=false' \
  'build --define xnn_enable_avxvnniint8=false' \
  'build --cxxopt=-std=gnu++17 --host_cxxopt=-std=gnu++17' \
  'build --cxxopt=-I/usr/include/opencv4' \

  > .bazelrc

# 5) Build both plugins inside the Mediapipe workspace
RUN set -eux; \
  bazel clean --expunge; \
  bazel build \
    //gstshared:libmp_runtime.so \
    //gstfacelandmarks:libgstfacelandmarks.so \
    //gstmozzamp:libgstmozzamp.so; \
  bbin="$(bazel info bazel-bin)"; \
  install -D -m0755 "$bbin/gstshared/libmp_runtime.so"               /out/lib/libmp_runtime.so; \
  install -D -m0755 "$bbin/gstfacelandmarks/libgstfacelandmarks.so" /out/plugins/libgstfacelandmarks.so; \
  install -D -m0755 "$bbin/gstmozzamp/libgstmozzamp.so"             /out/plugins/libgstmozzamp.so; \
  export GST_PLUGIN_PATH=/out/plugins:/opt/gstreamer/lib/x86_64-linux-gnu/gstreamer-1.0; \
  export GST_REGISTRY=/tmp/gst-registry.bin; export GST_REGISTRY_FORK=no; \
  GST_DEBUG=GST_PLUGIN_LOADING:3 gst-inspect-1.0 facelandmarks || true; \
  GST_DEBUG=GST_PLUGIN_LOADING:3 gst-inspect-1.0 mozza_mp || true

# 6) Export just the compiled artifacts
FROM scratch AS artifacts
COPY --from=builder /out /out