// cuda_preprocess.cu
// CUDA preprocessing kernels for TensorRT inference.

#include "cuda_preprocess.h"
#include <algorithm>

// Bilinear sample RGBA from device memory
__device__ static inline float4 bilinear_sample_rgba(
    const uint8_t* __restrict__ src, int srcW, int srcH, int srcPitch,
    float sx, float sy) {
  // Clamp coordinates
  sx = fmaxf(0.0f, fminf(sx, (float)(srcW - 1)));
  sy = fmaxf(0.0f, fminf(sy, (float)(srcH - 1)));

  int x0 = (int)floorf(sx);
  int y0 = (int)floorf(sy);
  int x1 = min(x0 + 1, srcW - 1);
  int y1 = min(y0 + 1, srcH - 1);

  float fx = sx - x0;
  float fy = sy - y0;

  auto load = [&](int x, int y) -> float4 {
    const uint8_t* p = src + y * srcPitch + x * 4;
    return make_float4(p[0], p[1], p[2], p[3]);
  };

  float4 p00 = load(x0, y0);
  float4 p10 = load(x1, y0);
  float4 p01 = load(x0, y1);
  float4 p11 = load(x1, y1);

  float4 result;
  result.x = (p00.x * (1 - fx) + p10.x * fx) * (1 - fy) +
             (p01.x * (1 - fx) + p11.x * fx) * fy;
  result.y = (p00.y * (1 - fx) + p10.y * fx) * (1 - fy) +
             (p01.y * (1 - fx) + p11.y * fx) * fy;
  result.z = (p00.z * (1 - fx) + p10.z * fx) * (1 - fy) +
             (p01.z * (1 - fx) + p11.z * fx) * fy;
  result.w = (p00.w * (1 - fx) + p10.w * fx) * (1 - fy) +
             (p01.w * (1 - fx) + p11.w * fx) * fy;
  return result;
}

// Kernel: full-frame RGBA -> RGB float resize + normalize + letterboxing
__global__ void k_rgba_to_rgb_resize_norm(
    const uint8_t* __restrict__ d_src, int srcW, int srcH, int srcPitch,
    float* __restrict__ d_dst, int dstW, int dstH, bool chw) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= dstW || dy >= dstH) return;

  float scale = fminf((float)dstW / srcW, (float)dstH / srcH);
  int crop_w = (int)roundf(srcW * scale);
  int crop_h = (int)roundf(srcH * scale);
  int pad_x = (dstW - crop_w) / 2;
  int pad_y = (dstH - crop_h) / 2;

  float r = 0.0f, g = 0.0f, b = 0.0f;

  if (dx >= pad_x && dx < pad_x + crop_w && dy >= pad_y && dy < pad_y + crop_h) {
    float sx = ((float)(dx - pad_x) + 0.5f) / scale - 0.5f;
    float sy = ((float)(dy - pad_y) + 0.5f) / scale - 0.5f;
    float4 px = bilinear_sample_rgba(d_src, srcW, srcH, srcPitch, sx, sy);
    r = px.x / 255.0f;
    g = px.y / 255.0f;
    b = px.z / 255.0f;
  }

  if (chw) {
    // CHW layout: [C][H][W]
    int hw = dstH * dstW;
    d_dst[0 * hw + dy * dstW + dx] = r;
    d_dst[1 * hw + dy * dstW + dx] = g;
    d_dst[2 * hw + dy * dstW + dx] = b;
  } else {
    // HWC layout: [H][W][C]
    int idx = (dy * dstW + dx) * 3;
    d_dst[idx + 0] = r;
    d_dst[idx + 1] = g;
    d_dst[idx + 2] = b;
  }
}

// Kernel: crop ROI + resize + RGBA->RGB float
__global__ void k_warp_affine_rgba_to_rgb_norm(
    const uint8_t* __restrict__ d_src, int srcW, int srcH, int srcPitch,
    float* __restrict__ d_dst, int dstW, int dstH,
    const float* __restrict__ mat, bool chw) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= dstW || dy >= dstH) return;

  // Affine transform: sx = m0*dx + m1*dy + m2
  float sx = mat[0] * (float)dx + mat[1] * (float)dy + mat[2];
  float sy = mat[3] * (float)dx + mat[4] * (float)dy + mat[5];

  float4 px = bilinear_sample_rgba(d_src, srcW, srcH, srcPitch, sx, sy);

  float r = px.x / 255.0f;
  float g = px.y / 255.0f;
  float b = px.z / 255.0f;

  if (chw) {
    int hw = dstH * dstW;
    d_dst[0 * hw + dy * dstW + dx] = r;
    d_dst[1 * hw + dy * dstW + dx] = g;
    d_dst[2 * hw + dy * dstW + dx] = b;
  } else {
    int idx = (dy * dstW + dx) * 3;
    d_dst[idx + 0] = r;
    d_dst[idx + 1] = g;
    d_dst[idx + 2] = b;
  }
}

// ── Host wrappers ──

void cuda_rgba_to_rgb_resize_normalize(
    const uint8_t* d_src_rgba, int srcW, int srcH, int srcPitch,
    float* d_dst_rgb, int dstW, int dstH, bool chw_layout,
    cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y);
  k_rgba_to_rgb_resize_norm<<<grid, block, 0, stream>>>(
      d_src_rgba, srcW, srcH, srcPitch, d_dst_rgb, dstW, dstH, chw_layout);
}

void cuda_warp_affine_rgba_to_rgb_normalize(
    const uint8_t* d_src_rgba, int srcW, int srcH, int srcPitch,
    float* d_dst_rgb, int dstW, int dstH,
    const float* matrix, bool chw_layout,
    cudaStream_t stream) {
  // matrix is 2x3 on host or device? For simplicity, we assume caller provides device pointer
  // but if it's host, we need a small copy.
  float* d_mat;
  cudaMallocAsync(&d_mat, 6 * sizeof(float), stream);
  cudaMemcpyAsync(d_mat, matrix, 6 * sizeof(float), cudaMemcpyHostToDevice, stream);

  dim3 block(16, 16);
  dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y);
  k_warp_affine_rgba_to_rgb_norm<<<grid, block, 0, stream>>>(
      d_src_rgba, srcW, srcH, srcPitch, d_dst_rgb, dstW, dstH, d_mat, chw_layout);
  
  cudaFreeAsync(d_mat, stream);
}
