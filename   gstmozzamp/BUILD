load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary")

cc_library(
    name = "imgwarp",
    srcs = glob(["imgwarp/**/*.c","imgwarp/**/*.cc","imgwarp/**/*.cpp"]),
    hdrs = glob(["imgwarp/**/*.h","imgwarp/**/*.hpp"]),
    includes = ["imgwarp"],
    copts = ["-fPIC","-O2","-I/usr/include/opencv4",
             "-fvisibility=hidden","-fvisibility-inlines-hidden","-fno-gnu-unique"],
    linkopts = ["-Wl,--as-needed","-Wl,--exclude-libs,ALL","-Wl,-Bsymbolic-functions"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mozzamp_core",
    srcs = ["dfm.cpp","deform_utils.cpp"],
    hdrs = ["dfm.hpp","deform_utils.hpp"],
    includes = ["."],
    copts = ["-std=gnu++17","-fPIC","-O2","-I/usr/include/opencv4",
             "-fvisibility=hidden","-fvisibility-inlines-hidden","-fno-gnu-unique"],
    deps = [":imgwarp"],
    linkopts = ["-Wl,--as-needed","-Wl,--exclude-libs,ALL","-Wl,-Bsymbolic-functions"],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "libgstmozzamp.so",
    srcs = ["gstmozzamp.cpp", "mp_runtime_loader.h"],
    linkshared = 1,
    linkstatic = 0,
    copts = ["-std=gnu++17","-fPIC","-O2","-Wno-deprecated-declarations","-I/usr/include/opencv4",
             "-fvisibility=hidden","-fvisibility-inlines-hidden","-fno-gnu-unique"],
    deps = [
        ":mozzamp_core",
        "//third_party/sysroot_gst:gstreamer",
    ],
    linkopts = [
        "-Wl,--as-needed","-ldl","-lopencv_core","-lopencv_imgproc",
        "-Wl,-rpath,/app/lib",
        "-Wl,-rpath,/opt/gstreamer/lib/x86_64-linux-gnu",
    ],
    visibility = ["//visibility:public"],
)