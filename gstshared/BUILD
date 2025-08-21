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
    copts = ["-fPIC", "-std=c++17"],
    linkopts = ["-Wl,--no-as-needed", "-ldl"],
    visibility = ["//visibility:public"],
)

# Keep only the GL inference calculators here; they bring required GL deps transitively.
cc_library(
    name = "mp_gpu_deps",
    srcs = [],
    deps = [
        "//mediapipe/calculators/tensor:inference_calculator_gl",
        "//mediapipe/calculators/tensor:inference_calculator_gl_advanced",
    ],
    alwayslink = 1,
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "libmp_runtime.so",
    srcs = ["mp_runtime.cc"],
    linkshared = 1,
    deps = [
        ":mp_runtime_hdrs",
        ":mp_gpu_deps",  # <-- use wrapper so registrations aren't GC’d
        "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker",
        "//mediapipe/framework/formats:image",
        "@com_google_absl//absl/status:statusor",
    ],
    copts = [
        "-O2",
        "-fPIC",
        "-std=c++17",
        "-I/usr/include/opencv4",
    ],
    linkopts = [
        "-Wl,-z,defs",
        "-lopencv_core",
        "-lopencv_imgproc",
        "-lEGL",
        "-lGLESv2",
    ],
    visibility = ["//visibility:public"],
)