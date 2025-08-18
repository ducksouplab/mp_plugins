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
    deps  = [
        ":mp_runtime_hdrs",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker",
        "//mediapipe/framework/formats:image",   # brings image_opencv.cc
        "@com_google_absl//absl/status:statusor",
    ],
    linkshared = 1,
    copts = [
        "-O2",
        "-fPIC",
        "-std=c++17",
        # (Optional if you added it to .bazelrc, harmless to keep)
        "-I/usr/include/opencv4",
    ],
    # Minimal OpenCV set needed by image_opencv.cc
    linkopts = [
        "-Wl,-z,defs",
        "-lopencv_core",
        "-lopencv_imgproc",
        # Add more if your runtime starts using them:
        # "-lopencv_imgcodecs", "-lopencv_video", "-lopencv_highgui",
    ],
    visibility = ["//visibility:public"],
)
