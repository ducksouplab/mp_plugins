# gstshared/BUILD
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_library(
    name = "mp_runtime_hdrs",
    hdrs = ["mp_runtime.h"],
    includes = ["."],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mp_runtime_loader",
    srcs = ["mp_runtime_loader.cc"],
    hdrs = ["mp_runtime_loader.h"],
    deps = [":mp_runtime_hdrs"],
    includes = ["."],
    copts = ["-fPIC", "-std=c++14"],
    linkopts = ["-Wl,--no-as-needed", "-ldl"],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "libmp_runtime.so",
    srcs = ["mp_runtime.cc"],
    deps  = [":mp_runtime_hdrs"],
    linkshared = 1,
    copts = ["-O2", "-fPIC", "-std=c++14"],
    linkopts = ["-Wl,-z,defs"],
    visibility = ["//visibility:public"],
)