# OpenCV 4 from Debian/Ubuntu packages
load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "opencv",
    hdrs = glob([
        # If you need to pin to your multiarch triplet, uncomment the exact one:
        # "include/x86_64-linux-gnu/opencv4/opencv2/**/*.h*",
        # "include/aarch64-linux-gnu/opencv4/opencv2/**/*.h*",
        # Otherwise the generic include/opencv4 works across arch:
        "include/opencv4/opencv2/**/*.h*",
    ]),
    includes = [
        "include/opencv4",
        "include/x86_64-linux-gnu/opencv4",  # harmless if absent on non-multiarch
    ],
    linkopts = [
        "-l:libopencv_core.so",
        "-l:libopencv_imgproc.so",
        "-l:libopencv_imgcodecs.so",
        "-l:libopencv_highgui.so",
        "-l:libopencv_video.so",
        "-l:libopencv_features2d.so",
        "-l:libopencv_calib3d.so",
        "-l:libopencv_videoio.so",
    ],
    visibility = ["//visibility:public"],
)