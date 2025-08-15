load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "libmp_runtime.so",
    srcs = ["mp_runtime.cc", "mp_runtime.h"],
    linkshared = 1,
    linkstatic = 1,
    copts = [
        "-std=gnu++17","-fPIC","-O2",
        "-fvisibility=hidden","-fvisibility-inlines-hidden","-fno-gnu-unique",
    ],
    deps = [
        "@mediapipe//mediapipe/framework/formats:image_frame",
        "@mediapipe//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker",
        "@com_google_absl//absl/status:statusor",
    ],
    linkopts = [
        "-Wl,--as-needed","-Wl,--exclude-libs,ALL","-Wl,-Bsymbolic-functions",
        "-Wl,-rpath,/app/lib","-Wl,-rpath,/usr/lib/x86_64-linux-gnu",
    ],
    visibility = ["//visibility:public"],
)