load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary")
package(default_visibility = ["//visibility:public"])

# Imgwarp helper lib (pulling sources under gstmozzamp/imgwarp)
cc_library(
    name = "imgwarp",
    srcs = glob(
        ["imgwarp/**/*.cc", "imgwarp/**/*.cpp"],
        exclude = [
            "imgwarp/**/CMakeLists.txt",
            "imgwarp/**/meson.build",
            "imgwarp/**/LICENSE",
            "imgwarp/OLD/**",
        ],
    ),
    hdrs = glob(["imgwarp/**/*.h", "imgwarp/**/*.hpp"]),
    includes = ["."],  # so #include "imgwarp/..." works
    copts = ["-I/usr/include/opencv4"],
    linkopts = ["-lopencv_core", "-lopencv_imgproc"],
)

# Your local sources + headers become a proper library so Bazel can expose the headers.
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
    deps = [
        ":imgwarp",
    ],
    copts = ["-I/usr/include/opencv4"],
    linkopts = ["-lopencv_core", "-lopencv_imgproc"],
)

# Shared object plugin
cc_binary(
    name = "libgstmozzamp.so",
    linkshared = 1,
    srcs = [
        "gstmozzamp.cpp",
    ],
    deps = [
        ":mozzamp_core",
        "//gstshared:mp_runtime_hdrs",
        "//gstshared:mp_runtime_loader_hdrs",
        "//third_party/sysroot_gst:gstreamer",
    ],
    copts = ["-I/usr/include/opencv4"],
    linkopts = ["-ldl", "-lopencv_core", "-lopencv_imgproc"],
)