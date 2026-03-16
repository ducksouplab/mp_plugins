load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary")
package(default_visibility = ["//visibility:public"])

# ── CUDA compilation via genrule ──
# Bazel doesn't natively support .cu files. We use nvcc via genrule.
# The Docker image provides CUDA toolkit at /usr/local/cuda.

genrule(
    name = "cuda_preprocess_obj",
    srcs = ["cuda_preprocess.cu", "cuda_preprocess.h"],
    outs = ["cuda_preprocess.o"],
    cmd = "/usr/local/cuda/bin/nvcc -c -O3 -Xcompiler -fPIC " +
          "--gpu-architecture=sm_75 --gpu-code=sm_75,sm_80,sm_86,sm_89,sm_90 " +
          "-I$$(dirname $(location cuda_preprocess.cu)) " +
          "$(location cuda_preprocess.cu) -o $@",
)

genrule(
    name = "cuda_mls_warp_obj",
    srcs = ["cuda_mls_warp.cu", "cuda_mls_warp.h"],
    outs = ["cuda_mls_warp.o"],
    cmd = "/usr/local/cuda/bin/nvcc -c -O3 -Xcompiler -fPIC " +
          "--gpu-architecture=sm_75 --gpu-code=sm_75,sm_80,sm_86,sm_89,sm_90 " +
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
    linkopts = [
        "-L/usr/local/cuda/lib64",
        "-lcudart",
    ],
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
    copts = [
        "-fPIC",
        "-I/usr/local/cuda/include",
    ],
    deps = [
        ":task_model_extractor",
        ":cuda_kernels",
    ],
    linkopts = [
        "-L/usr/local/cuda/lib64",
        "-lcudart",
        "-lnvinfer",
        "-lnvonnxparser",
    ],
)

# The GPU plugin
cc_binary(
    name = "libgstmozzamp_gpu.so",
    linkshared = 1,
    srcs = ["gstmozzamp_gpu.cpp"],
    deps = [
        ":trt_face_landmarker",
        ":cuda_kernels",
        "//gstmozzamp:mozzamp_core",
        "//third_party/sysroot_gst:gstreamer",
    ],
    copts = [
        "-I/usr/include/opencv4",
        "-I/usr/local/cuda/include",
    ],
    linkopts = [
        "-Wl,-z,defs",
        "-Wl,--no-as-needed",
        "-Wl,--exclude-libs,ALL",
        "-L/usr/local/cuda/lib64",
        "-lcudart",
        "-lnvinfer",
        "-lnvonnxparser",
        "-lminizip",
        "-lz",
        "-lopencv_core",
    ],
)
