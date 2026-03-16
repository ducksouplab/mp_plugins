// cuda_preprocess.h
// CUDA kernels for image preprocessing: RGBA->RGB, resize, crop, normalize.
// All ops run on device memory, no CPU copies.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// Resize RGBA GPU image to target size, convert RGBA->RGB float [0,1].
// Input:  d_src_rgba  (srcW x srcH, stride srcPitch bytes, RGBA uint8)
// Output: d_dst_rgb   (dstW x dstH x 3, float, packed CHW or HWC)
// Uses bilinear interpolation.
void cuda_rgba_to_rgb_resize_normalize(
    const uint8_t* d_src_rgba, int srcW, int srcH, int srcPitch,
    float* d_dst_rgb, int dstW, int dstH,
    bool chw_layout,  // true=CHW (TRT default), false=HWC
    cudaStream_t stream);

// Crop a ROI from RGBA GPU image, resize to target, convert to RGB float [0,1].
// roi_x, roi_y, roi_w, roi_h define the crop region.
void cuda_crop_rgba_to_rgb_resize_normalize(
    const uint8_t* d_src_rgba, int srcW, int srcH, int srcPitch,
    float* d_dst_rgb, int dstW, int dstH,
    int roi_x, int roi_y, int roi_w, int roi_h,
    bool chw_layout,
    cudaStream_t stream);
