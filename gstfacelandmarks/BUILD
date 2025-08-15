load("@rules_cc//cc:defs.bzl", "cc_binary")
package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "libgstfacelandmarks.so",
    linkshared = 1,
    srcs = ["gstfacelandmarks.cpp"],
    deps = [
        "//gstshared:mp_runtime_hdrs",
        "//gstshared:mp_runtime_loader_hdrs",
        "//third_party/sysroot_gst:gstreamer",
    ],
    linkopts = ["-ldl", "-lm"],   # for dlopen + lround
)