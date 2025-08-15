load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "libgstfacelandmarks.so",
    srcs = ["gstfacelandmarks.cpp", "mp_runtime_loader.h"],
    linkshared = 1,
    linkstatic = 0,
    copts = [
        "-std=gnu++17","-fPIC","-O2",
        "-fvisibility=hidden","-fvisibility-inlines-hidden","-fno-gnu-unique",
    ],
    deps = ["//third_party/sysroot_gst:gstreamer"],
    linkopts = [
        "-Wl,--as-needed", "-ldl",
        "-Wl,-rpath,/app/lib",
        "-Wl,-rpath,/opt/gstreamer/lib/x86_64-linux-gnu",
    ],
    visibility = ["//visibility:public"],
)