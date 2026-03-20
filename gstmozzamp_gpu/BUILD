load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary")
package(default_visibility = ["//visibility:public"])

# ── CUDA compilation via genrule ──
# We use nvcc via genrule.
# The @cuda repository provides the toolkit.

genrule(
    name = "cuda_preprocess_obj",
    srcs = ["cuda_preprocess.cu", "cuda_preprocess.h"],
    outs = ["cuda_preprocess.o"],
    cmd = "/usr/local/cuda/bin/nvcc -c -O3 -Xcompiler -fPIC " +
          "--gpu-architecture=compute_75 --gpu-code=sm_75,sm_80,sm_86,sm_89,sm_90 " +
          "-I$$(dirname $(location cuda_preprocess.cu)) " +
          "$(location cuda_preprocess.cu) -o $@",
)

genrule(
    name = "cuda_mls_warp_obj",
    srcs = ["cuda_mls_warp.cu", "cuda_mls_warp.h"],
    outs = ["cuda_mls_warp.o"],
    cmd = "/usr/local/cuda/bin/nvcc -c -O3 -Xcompiler -fPIC " +
          "--gpu-architecture=compute_75 --gpu-code=sm_75,sm_80,sm_86,sm_89,sm_90 " +
          "-I$$(dirname $(location cuda_mls_warp.cu)) " +
          "$(location cuda_mls_warp.cu) -o $@",
)

# Wrap CUDA objects as cc_library
cc_library(
    name = "cuda_kernels",
    srcs = [
        ":cuda_preprocess_obj",
        ":cuda_mls_warp_obj",
    ],
    hdrs = [
        "cuda_preprocess.h",
        "cuda_mls_warp.h",
    ],
    deps = ["@cuda//:cudart"],
)

# Task model extractor (ZIP/minizip)
cc_library(
    name = "task_model_extractor",
    srcs = ["task_model_extractor.cpp"],
    hdrs = ["task_model_extractor.h"],
    copts = ["-fPIC"],
    linkopts = ["-lminizip", "-lz"],
)

# TensorRT face landmarker
cc_library(
    name = "trt_face_landmarker",
    srcs = ["trt_face_landmarker.cpp"],
    hdrs = ["trt_face_landmarker.h"],
    copts = ["-fPIC"],
    deps = [
        ":task_model_extractor",
        ":cuda_kernels",
        "@cuda//:cuda_headers",
        "@cuda//:cudart",
    ],
    linkopts = [
        "-lnvinfer",
        "-lnvonnxparser",
    ],
)

# The GPU plugin
cc_binary(
    name = "libgstmozzampgpu.so",
    linkshared = 1,
    srcs = ["gstmozzamp_gpu.cpp"],
    deps = [
        ":trt_face_landmarker",
        ":cuda_kernels",
        "//gstmozzamp:mozzamp_core",
        "//third_party/sysroot_gst:gstreamer",
        "@cuda//:cuda_headers",
        "@cuda//:cudart",
    ],
    copts = ["-I/usr/include/opencv4"],
    linkopts = [
        "-Wl,-z,defs",
        "-Wl,--no-as-needed",
        "-Wl,--exclude-libs,ALL",
        "-lnvinfer",
        "-lnvonnxparser",
        "-lminizip",
        "-lz",
        "-lopencv_core",
    ],
)
