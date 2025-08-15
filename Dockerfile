# Build both plugins + a small shared mp runtime using Bzlmod & a local mediapipe checkout.
# Artifacts are exported under /out (see final scratch stage).
FROM ducksouplab/debian-gstreamer:deb12-cuda12.2-plugins-gst1.24.10 AS builder

ARG MEDIAPIPE_TAG=v0.10.26
ARG BAZEL_VERSION=6.5.0
ENV DEBIAN_FRONTEND=noninteractive

# Toolchain + headers
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git pkg-config build-essential \
      python3 python3-numpy libglib2.0-dev \
      libopencv-core-dev libopencv-imgproc-dev \
    && rm -rf /var/lib/apt/lists/*

# Bazel
RUN curl -fsSL -o /usr/local/bin/bazel \
      https://releases.bazel.build/${BAZEL_VERSION}/release/bazel-${BAZEL_VERSION}-linux-x86_64 \
    && chmod +x /usr/local/bin/bazel

# 1) Clone MediaPipe (untouched)
WORKDIR /opt
RUN git clone --depth=1 --branch ${MEDIAPIPE_TAG} \
      https://github.com/google-ai-edge/mediapipe.git mediapipe-src

# 2) Copy YOUR repo into /opt/gst-plugins
WORKDIR /opt/gst-plugins
COPY . /opt/gst-plugins/

# 3) Bazel workspace + Bzlmod (use mediapipe via local_path_override)
#    We enable bzlmod so MediaPipe brings its own deps; no mediapipe_deps.bzl needed.
RUN bash -eux <<BASH
# WORKSPACE (minimal — just the name)
cat > WORKSPACE <<'WS'
workspace(name = "ducksoup_gst")
WS

# MODULE.bazel with a parametric version derived from MEDIAPIPE_TAG
VER="\${MEDIAPIPE_TAG#v}"
cat > MODULE.bazel <<EOF
module(name = "ducksoup_gst")
bazel_dep(name = "mediapipe", version = "\${VER}")
local_path_override(module_name = "mediapipe", path = "../mediapipe-src")
EOF

# .bazelrc — enable bzlmod + build flags
cat > .bazelrc <<'RC'
common --enable_bzlmod
build --define MEDIAPIPE_DISABLE_GPU=1
build --define xnn_enable_avxvnni=false
build --define xnn_enable_avxvnniint8=false
build --cxxopt=-std=gnu++17 --host_cxxopt=-std=gnu++17
RC
BASH

# 4) Vendor GStreamer/GLib headers into our workspace (NOT in mediapipe-src)
RUN bash -eux <<'BASH'
mkdir -p third_party/sysroot_gst/include \
         third_party/sysroot_gst/lib/glib-2.0/include
cp -a /opt/gstreamer/include/gstreamer-1.0 third_party/sysroot_gst/include/
cp -a /usr/include/glib-2.0               third_party/sysroot_gst/include/
cp -a /usr/lib/x86_64-linux-gnu/glib-2.0/include/* \
      third_party/sysroot_gst/lib/glib-2.0/include/
BASH

# 5) Create Bazel packages + copy your sources + write BUILD files
RUN bash -eux <<'BASH'
# Create packages
mkdir -p gstfacelandmarks gstmozzamp gstmozzamp/imgwarp gstshared third_party/sysroot_gst

# Copy your sources from repo /opt/gst-plugins/src/... into packages
# Face landmarker
if [ -f src/gstfacelandmarks.cpp ]; then
  cp -a src/gstfacelandmarks.cpp gstfacelandmarks/
fi

# Mozza MP (MLS)
for f in gstmozzamp.cpp dfm.cpp dfm.hpp deform_utils.cpp deform_utils.hpp; do
  if [ -f "src/$f" ]; then cp -a "src/$f" gstmozzamp/; fi
done
if [ -d src/imgwarp ]; then
  cp -a src/imgwarp/* gstmozzamp/imgwarp/
fi

# ---------------------------
# Shared runtime: header + cc
# ---------------------------
cat > gstshared/mp_runtime.h <<'EOF'
#pragma once
namespace mp_runtime {
  // Initialize MediaPipe/TFLite/Abseil registries exactly once in this process.
  void InitOnce();
}
EOF

cat > gstshared/mp_runtime.cc <<'EOF'
#include "gstshared/mp_runtime.h"
#include <mutex>
#include <typeinfo>

// Pull in a MediaPipe symbol so the linker keeps MP bits in this .so.
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker.h"

namespace mp_runtime {
namespace {
std::once_flag g_once;
void DoInit() {
  (void)typeid(mediapipe::tasks::vision::face_landmarker::FaceLandmarkerOptions);
}
}  // namespace
void InitOnce() { std::call_once(g_once, DoInit); }
}  // namespace mp_runtime
EOF

# Lightweight loader headers for each plugin
cat > gstfacelandmarks/mp_runtime_loader.h <<'EOF'
#pragma once
#include "gstshared/mp_runtime.h"
struct MpRuntimeInit { MpRuntimeInit(){ mp_runtime::InitOnce(); } };
static MpRuntimeInit g_mp_runtime_init;
EOF
cp gstfacelandmarks/mp_runtime_loader.h gstmozzamp/mp_runtime_loader.h

# -------------------------------------
# third_party sysroot BUILD (glib/gst)
# -------------------------------------
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

# --------------
# gstshared/BUILD
# --------------
cat > gstshared/BUILD <<'EOF'
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_import")

cc_binary(
    name = "libmp_runtime.so",
    srcs = ["mp_runtime.cc"],
    linkshared = 1,
    copts = [
        "-std=gnu++17", "-fPIC", "-O2",
        "-fvisibility=hidden", "-fvisibility-inlines-hidden",
        "-fno-gnu-unique",
    ],
    linkopts = [
        "-Wl,--as-needed",
        "-Wl,--exclude-libs,ALL",
        "-Wl,-Bsymbolic-functions",
        "-Wl,-soname,libmp_runtime.so",
    ],
    deps = [
        "@mediapipe//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker",
    ],
    visibility = ["//visibility:public"],
)

cc_import(
    name = "mp_runtime_shared",
    shared_library = ":libmp_runtime.so",
    hdrs = ["mp_runtime.h"],
    visibility = ["//visibility:public"],
)
EOF

# -----------------------
# gstfacelandmarks/BUILD
# -----------------------
cat > gstfacelandmarks/BUILD <<'EOF'
load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "libgstfacelandmarks.so",
    srcs = ["gstfacelandmarks.cpp"],
    linkshared = 1,
    copts = [
        "-std=gnu++17","-fPIC","-O2",
        "-Wno-deprecated-declarations",
        "-fvisibility=hidden","-fvisibility-inlines-hidden","-fno-gnu-unique",
    ],
    deps = [
        "//third_party/sysroot_gst:gstreamer",
        "//gstshared:mp_runtime_shared",
        # If your current code still includes MediaPipe headers directly,
        # keep these; remove once fully migrated to the mp_runtime API.
        "@mediapipe//mediapipe/framework/formats:image",
        "@mediapipe//mediapipe/framework/formats:image_frame",
        "@mediapipe//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker",
    ],
    linkopts = [
        "-Wl,--as-needed",
        "-Wl,--exclude-libs,ALL",
        "-Wl,-Bsymbolic-functions",
        "-Wl,-rpath,\\$ORIGIN/../lib",
        "-Wl,-rpath,/opt/gstreamer/lib/x86_64-linux-gnu",
    ],
    visibility = ["//visibility:public"],
)
EOF

# ----------------
# gstmozzamp/BUILD
# ----------------
cat > gstmozzamp/BUILD <<'EOF'
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary")

# imgwarp third-party bits
cc_library(
    name = "imgwarp",
    srcs = glob(["imgwarp/**/*.c","imgwarp/**/*.cc","imgwarp/**/*.cpp"]),
    hdrs = glob(["imgwarp/**/*.h","imgwarp/**/*.hpp"]),
    includes = ["imgwarp"],
    copts = [
        "-fPIC","-O2","-I/usr/include/opencv4",
        "-fvisibility=hidden","-fvisibility-inlines-hidden","-fno-gnu-unique",
    ],
    linkopts = ["-Wl,--as-needed","-Wl,--exclude-libs,ALL","-Wl,-Bsymbolic-functions"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mozzamp_core",
    srcs = ["dfm.cpp","deform_utils.cpp"],
    hdrs = ["dfm.hpp","deform_utils.hpp"],
    copts = [
        "-std=gnu++17","-fPIC","-O2","-I/usr/include/opencv4",
        "-fvisibility=hidden","-fvisibility-inlines-hidden","-fno-gnu-unique",
    ],
    deps = [":imgwarp"],
    linkopts = ["-Wl,--as-needed","-Wl,--exclude-libs,ALL","-Wl,-Bsymbolic-functions"],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "libgstmozzamp.so",
    srcs = ["gstmozzamp.cpp"],
    linkshared = 1,
    copts = [
        "-std=gnu++17","-fPIC","-O2","-I/usr/include/opencv4",
        "-fvisibility=hidden","-fvisibility-inlines-hidden","-fno-gnu-unique",
    ],
    deps = [
        ":mozzamp_core",
        "//third_party/sysroot_gst:gstreamer",
        "//gstshared:mp_runtime_shared",
    ],
    linkopts = [
        "-Wl,--as-needed",
        "-Wl,--exclude-libs,ALL",
        "-Wl,-Bsymbolic-functions",
        "-Wl,-rpath,\\$ORIGIN/../lib",
        "-Wl,-rpath,/opt/gstreamer/lib/x86_64-linux-gnu",
        "-lopencv_core","-lopencv_imgproc",
    ],
    visibility = ["//visibility:public"],
)
EOF
BASH

# 6) Build shared runtime + BOTH plugins
WORKDIR /opt/gst-plugins
RUN set -eux; \
  bazel clean --expunge; \
  bazel build \
    //gstshared:libmp_runtime.so \
    //gstfacelandmarks:libgstfacelandmarks.so \
    //gstmozzamp:libgstmozzamp.so; \
  bbin="$(bazel info bazel-bin)"; \
  install -D -m0755 "$bbin/gstshared/libmp_runtime.so"                  /out/lib/libmp_runtime.so; \
  install -D -m0755 "$bbin/gstfacelandmarks/libgstfacelandmarks.so"    /out/plugins/libgstfacelandmarks.so; \
  install -D -m0755 "$bbin/gstmozzamp/libgstmozzamp.so"                 /out/plugins/libgstmozzamp.so; \
  # quick sanity: inspect inside this process (non-fatal)
  export GST_PLUGIN_PATH=/out/plugins:/opt/gstreamer/lib/x86_64-linux-gnu/gstreamer-1.0; \
  export GST_REGISTRY=/tmp/gst-registry.bin; export GST_REGISTRY_FORK=no; \
  GST_DEBUG=GST_PLUGIN_LOADING:3 gst-inspect-1.0 facelandmarks || true; \
  GST_DEBUG=GST_PLUGIN_LOADING:3 gst-inspect-1.0 mozza_mp || true

# ---- Final stage that only ships the compiled artifacts
FROM scratch AS artifacts
COPY --from=builder /out /out