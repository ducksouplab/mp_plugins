# Build MediaPipe Face Landmarker GStreamer plugin using the same Ducksoup base.
# Produces /app/plugins/libgstfacelandmarks.so inside the image.

FROM ducksouplab/debian-gstreamer:deb12-cuda12.2-plugins-gst1.24.10 AS builder

ARG MEDIAPIPE_TAG=v0.10.26
ARG BAZEL_VERSION=6.5.0
ENV DEBIAN_FRONTEND=noninteractive

# Toolchain + minimal Python for Bazel's hermetic rules
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git pkg-config build-essential \
    python3 python3-numpy libglib2.0-dev \
 && rm -rf /var/lib/apt/lists/*

# Bazel
RUN curl -fsSL -o /usr/local/bin/bazel \
    https://releases.bazel.build/${BAZEL_VERSION}/release/bazel-${BAZEL_VERSION}-linux-x86_64 \
 && chmod +x /usr/local/bin/bazel

WORKDIR /opt/gst-facelandmarks

# MediaPipe sources
RUN git clone --depth=1 --branch ${MEDIAPIPE_TAG} \
    https://github.com/google-ai-edge/mediapipe.git mediapipe-src

# Your plugin source
COPY src/gstfacelandmarks.cpp mediapipe-src/gstfacelandmarks/gstfacelandmarks.cpp

# Vendor only the GStreamer/GLib headers into a safe Bazel package.
RUN bash -eux <<'BASH'
cd mediapipe-src

mkdir -p third_party/sysroot_gst/include \
         third_party/sysroot_gst/lib/glib-2.0/include

# NOTE: GStreamer headers live under /opt/gstreamer in your base image.
cp -a /opt/gstreamer/include/gstreamer-1.0 third_party/sysroot_gst/include/
cp -a /usr/include/glib-2.0               third_party/sysroot_gst/include/
cp -a /usr/lib/x86_64-linux-gnu/glib-2.0/include/* \
      third_party/sysroot_gst/lib/glib-2.0/include/

# Define cc_libraries for GLib and GStreamer.
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
    # Ensure linker finds libs under /opt/gstreamer.
    linkopts = ["-L/opt/gstreamer/lib/x86_64-linux-gnu",
                "-lgstvideo-1.0", "-lgstbase-1.0", "-lgstreamer-1.0"],
    visibility = ["//visibility:public"],
)
EOF

# Plugin BUILD rule: don't force OpenCV; let MediaPipe decide.
mkdir -p gstfacelandmarks
cat > gstfacelandmarks/BUILD <<'EOF'
load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "libgstfacelandmarks.so",
    srcs = ["gstfacelandmarks.cpp"],
    linkshared = 1,
    copts = [
        "-std=gnu++17",
        "-fPIC",
        "-O2",
        "-Wno-deprecated-declarations",
    ],
    deps = [
        "//third_party/sysroot_gst:gstreamer",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker",
        "@com_google_absl//absl/status:statusor",
    ],
    # Helpful RPATHs so the loader can find libs at runtime without extra env.
    linkopts = [
        "-Wl,-rpath,/opt/gstreamer/lib/x86_64-linux-gnu",
        "-Wl,-rpath,/usr/local/lib",
        "-Wl,-rpath,/usr/lib/x86_64-linux-gnu"
    ],
    visibility = ["//visibility:public"],
)
EOF

# Patch overlay loop for Tasks 0.10.x (NormalizedLandmarks has .landmarks)
sed -i -E 's/for\s*\(\s*size_t\s+i\s*=\s*0;\s*i\s*<\s*face\.size\(\)\s*;\s*\+\+i\s*\)/for (const auto\& lm : face.landmarks)/' \
  gstfacelandmarks/gstfacelandmarks.cpp
sed -i -E '/^\s*const\s+auto&\s+lm\s*=\s*face\[i\];\s*$/d' \
  gstfacelandmarks/gstfacelandmarks.cpp
BASH

WORKDIR /opt/gst-facelandmarks/mediapipe-src

# Stable CPU-only build. Keep OpenCV enabled (needed by ImageToTensor CPU path).
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
    bazel build //gstfacelandmarks:libgstfacelandmarks.so; \
    bbin="$(bazel info bazel-bin)"; \
    real="${bbin}/gstfacelandmarks/libgstfacelandmarks.so"; \
    install -D -m 0755 "$real" /app/plugins/libgstfacelandmarks.so; \
    # quick self-check (scanner runs in a separate process)
    GST_PLUGIN_PATH=/app/plugins \
    GST_DEBUG=GST_PLUGIN_LOADING:4 \
    gst-inspect-1.0 facelandmarks || true

# Optional: final stage identical to your base, just ships the .so
FROM ducksouplab/debian-gstreamer:deb12-cuda12.2-plugins-gst1.24.10 AS runtime
COPY --from=builder /app/plugins/libgstfacelandmarks.so /app/plugins/libgstfacelandmarks.so
ENV GST_PLUGIN_PATH=/app/plugins:$GST_PLUGIN_PATH
# (No CMD here — you’ll run your Ducksoup binary as usual)