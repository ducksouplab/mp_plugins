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
    linkopts = ["-L/opt/gstreamer/lib/x86_64-linux-gnu",
                "-lgstvideo-1.0", "-lgstbase-1.0", "-lgstreamer-1.0"],
    visibility = ["//visibility:public"],
)