load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "gstreamer",
    copts = [
        "-I/usr/include/gstreamer-1.0",
        "-I/usr/include/glib-2.0",
        "-I/usr/lib/x86_64-linux-gnu/glib-2.0/include",
    ],
    linkopts = [
        "-lgstreamer-1.0",
        "-lgstvideo-1.0",
        "-lgobject-2.0",
        "-lglib-2.0",
    ],
    visibility = ["//visibility:public"],
)