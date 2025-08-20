load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary")
package(default_visibility = ["//visibility:public"])

# 1) Imgwarp helper lib (static archive, but force whole-archive semantics)
cc_library(
    name = "imgwarp",
    srcs = [
        "imgwarp/imgwarp_mls.cpp",
        "imgwarp/imgwarp_mls_rigid.cpp",
        "imgwarp/imgwarp_piecewiseaffine.cpp",
        "imgwarp/delaunay.cpp",  # if present, else remove
    ],
    hdrs = [
        "imgwarp/imgwarp_mls.h",
        "imgwarp/imgwarp_mls_rigid.h",
        "imgwarp/imgwarp_piecewiseaffine.h",
        "imgwarp/delaunay.h",
    ],
    copts = [
        "-fPIC",
        "-I/usr/include/opencv4",
        "-fvisibility=hidden",
    ],
    # Do NOT put linkopts here; keep them only on the final binary.
    alwayslink = True,   # <---- important
)

# 2) Local core util lib
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
    copts = [
        "-fPIC",
        "-I/usr/include/opencv4",
    ],
    deps = [
        ":imgwarp",
    ],
)

# 3) The plugin itself
cc_binary(
    name = "libgstmozzamp.so",
    linkshared = 1,
    srcs = [
        "gstmozzamp.cpp",
    ],
    deps = [
        ":mozzamp_core",
        ":imgwarp",
        "//gstshared:mp_runtime_hdrs",
        "//gstshared:mp_runtime_loader",
        "//third_party/sysroot_gst:gstreamer",
    ],
    copts = ["-I/usr/include/opencv4"],
    linkopts = [
        "-Wl,-z,defs",           # <---- fail at link time if anything is undefined
        "-Wl,--no-as-needed",    # <---- prevents ld from dropping needed libs
        "-Wl,--exclude-libs,ALL",
        "-ldl",
        "-lopencv_core",
        "-lopencv_imgproc",
    ],
)
