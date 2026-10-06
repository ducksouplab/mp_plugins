load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary")
package(default_visibility = ["//visibility:public"])

# Displacement-field face warping (basis JSON loading, face frame, mesh warp).
cc_library(
    name = "facewarp",
    srcs = ["facewarp.cpp"],
    hdrs = ["facewarp.hpp", "face_mesh_data.hpp"],
    includes = ["."],
    copts = ["-fPIC", "-I/usr/include/opencv4"],
    deps = ["//third_party/sysroot_gst:json_glib"],
)

# The mozza_mesh GStreamer plugin (MediaPipe via the shared mp_runtime loader).
cc_binary(
    name = "libgstmozzamesh.so",
    linkshared = 1,
    srcs = ["gstmozzamesh.cpp"],
    deps = [
        ":facewarp",
        "//gstshared:mp_runtime_hdrs",
        "//gstshared:mp_runtime_loader",
        "//third_party/sysroot_gst:gstreamer",
        "//third_party/sysroot_gst:json_glib",
    ],
    copts = ["-I/usr/include/opencv4"],
    linkopts = [
        "-Wl,-z,defs",
        "-Wl,--no-as-needed",
        "-Wl,--exclude-libs,ALL",
        "-ldl",
        "-lopencv_core",
        "-lopencv_imgproc",
    ],
)
