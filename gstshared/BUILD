load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary")

package(default_visibility = ["//visibility:public"])

# Header-only exports
cc_library(
    name = "mp_runtime_hdrs",
    hdrs = ["mp_runtime.h"],
    includes = ["."],  # allow #include "gstshared/mp_runtime.h" or "mp_runtime.h" from this pkg
)

cc_library(
    name = "mp_runtime_loader_hdrs",
    hdrs = ["mp_runtime_loader.h"],
    includes = ["."],
    deps = [":mp_runtime_hdrs"],
)

# Optional empty runtime .so (kept so Dockerfile can copy it)
cc_binary(
    name = "libmp_runtime.so",
    linkshared = 1,
    srcs = ["mp_runtime.cc"],   # your file can be empty
    deps = [":mp_runtime_hdrs"],
)