// cuda_mls_warp.cu
// GPU implementation of MLS Rigid warp.
// Direct port of imgwarp_mls_rigid.cpp::calcDelta() + imgwarp_mls.cpp::genNewImg().
//
// Two kernels:
//   1. mls_rigid_delta_kernel: computes displacement field (rDx, rDy) at grid points
//   2. mls_warp_kernel: bilinearly interpolates displacement + samples source image
//
// Memory layout:
//   - Displacement field: dense 2D float arrays (gridH x gridW)
//   - Source image: CUDA texture object for HW bilinear interpolation
//   - Control points: in shared memory (typically 20-100 points, fits easily)

#include "cuda_mls_warp.h"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cstring>

// Maximum control points that fit in shared memory (48KB / (2*8) = 3072)
// In practice we have ~80 points max, so this is very comfortable.
#define MAX_CTRL_POINTS 512

// ── Kernel 1: Compute MLS Rigid displacement field ──
// Each thread handles one grid node.
// Faithfully replicates the CPU algorithm in imgwarp_mls_rigid.cpp::calcDelta().
__global__ void mls_rigid_delta_kernel(
    float* __restrict__ d_rDx,
    float* __restrict__ d_rDy,
    const float* __restrict__ d_oldPts,  // dst points (interleaved x,y)
    const float* __restrict__ d_newPts,  // src points (interleaved x,y)
    int nPoint,
    int tarW, int tarH,
    int gridSize,
    float alpha_param,
    float ratio,
    int rx, int ry,
    int gridW, int gridH)       // pre-scale ratio (1.0 if disabled)
{
  // Load control points into shared memory
  __shared__ float s_oldX[MAX_CTRL_POINTS];
  __shared__ float s_oldY[MAX_CTRL_POINTS];
  __shared__ float s_newX[MAX_CTRL_POINTS];
  __shared__ float s_newY[MAX_CTRL_POINTS];

  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int blockSize = blockDim.x * blockDim.y;
  for (int k = tid; k < nPoint; k += blockSize) {
    s_oldX[k] = d_oldPts[k * 2];
    s_oldY[k] = d_oldPts[k * 2 + 1];
    // newPts already have preScale applied by host
    s_newX[k] = d_newPts[k * 2];
    s_newY[k] = d_newPts[k * 2 + 1];
  }
  __syncthreads();

  // Grid node index
  int gi = blockIdx.x * blockDim.x + threadIdx.x;  // x coord in grid space
  int gj = blockIdx.y * blockDim.y + threadIdx.y;  // y coord in grid space

  if (gi >= gridW || gj >= gridH) return;

  // Map grid index to pixel coordinate (matching CPU loop logic)
  // i and j are absolute coordinates in the image.
  int i = rx + gi * gridSize;
  int j = ry + gj * gridSize;

  // Clamp to image bounds (matches CPU edge handling)
  if (i >= tarW) i = tarW - 1;
  if (j >= tarH) j = tarH - 1;

  // --- MLS Rigid algorithm (exact port of CPU) ---
  float sw = 0.0f;
  float swpx = 0.0f, swpy = 0.0f;
  float swqx = 0.0f, swqy = 0.0f;

  int exact_k = -1;

  // 1. Loop 1: Compute sw, swpx, swpy, swqx, swqy
  for (int k = 0; k < nPoint; ++k) {
    float dx = (float)i - s_oldX[k];
    float dy = (float)j - s_oldY[k];
    float d2 = dx * dx + dy * dy;

    if (d2 < 1e-4f) { // Increased threshold for stability
      exact_k = k;
      break;
    }

    float wk = (alpha_param == 1.0f) ? 1.0f / d2 : powf(d2, -alpha_param);

    sw += wk;
    swpx += wk * s_oldX[k];
    swpy += wk * s_oldY[k];
    swqx += wk * s_newX[k];
    swqy += wk * s_newY[k];
  }

  float newPx = 0.0f, newPy = 0.0f;

  if (exact_k >= 0) {
    // Exactly on a control point
    newPx = s_newX[exact_k];
    newPy = s_newY[exact_k];
  } else {
    // 2. Compute pstar, qstar from those sums
    float inv_sw = 1.0f / sw;
    float pstarx = swpx * inv_sw;
    float pstary = swpy * inv_sw;
    float qstarx = swqx * inv_sw;
    float qstary = swqy * inv_sw;

    // 3. Loop 2: Compute s1, s2 (for miu_r) and sumTmpPx, sumTmpPy using pstar, qstar
    float s1 = 0.0f, s2 = 0.0f;
    float sumTmpPx = 0.0f, sumTmpPy = 0.0f;

    float curVx = (float)i - pstarx;
    float curVy = (float)j - pstary;
    float curVJx = -curVy;
    float curVJy = curVx;

    for (int k = 0; k < nPoint; ++k) {
      float dx = (float)i - s_oldX[k];
      float dy = (float)j - s_oldY[k];
      float d2 = dx * dx + dy * dy;
      
      float wk = (alpha_param == 1.0f) ? 1.0f / (d2 + 1e-6f) : powf(d2 + 1e-6f, -alpha_param);

      float Pix = s_oldX[k] - pstarx;
      float Piy = s_oldY[k] - pstary;
      float PiJx = -Piy;
      float PiJy = Pix;

      float Qix = s_newX[k] - qstarx;
      float Qiy = s_newY[k] - qstary;

      s1 += wk * (Qix * Pix + Qiy * Piy);
      s2 += wk * (Qix * PiJx + Qiy * PiJy);

      float PidotcurV = Pix * curVx + Piy * curVy;
      float PiJdotcurV = PiJx * curVx + PiJy * curVy;
      float PidotcurVJ = Pix * curVJx + Piy * curVJy;
      float PiJdotcurVJ = PiJx * curVJx + PiJy * curVJy;

      float tmpPx = PidotcurV * s_newX[k] - PiJdotcurV * s_newY[k];
      float tmpPy = -PidotcurVJ * s_newX[k] + PiJdotcurVJ * s_newY[k];

      sumTmpPx += wk * tmpPx;
      sumTmpPy += wk * tmpPy;
    }

    float miu_r = sqrtf(s1 * s1 + s2 * s2);

    if (miu_r < 1e-8f) { // Increased threshold
      newPx = qstarx;
      newPy = qstary;
    } else {
      newPx = sumTmpPx / miu_r + qstarx;
      newPy = sumTmpPy / miu_r + qstary;
    }
  }

  // Apply pre-scale ratio and compute displacement
  float dispX = newPx * ratio - (float)i;
  float dispY = newPy * ratio - (float)j;

  // Store in grid-indexed array
  int idx = gj * gridW + gi;
  d_rDx[idx] = dispX;
  d_rDy[idx] = dispY;
}

// ── Kernel 2: Apply warp using bilinear interpolation of displacement field ──
// Each thread handles one output pixel.
// Replicates genNewImg() from imgwarp_mls.cpp.
__global__ void mls_warp_kernel(
    const uint8_t* __restrict__ d_src,
    uint8_t* __restrict__ d_dst,
    const float* __restrict__ d_rDx,
    const float* __restrict__ d_rDy,
    int W, int H,
    int srcPitch, int dstPitch,
    int gridSize,
    int gridW, int gridH,
    float transRatio,
    int rx, int ry,
    int rw, int rh)
{
  int lx = blockIdx.x * blockDim.x + threadIdx.x;
  int ly = blockIdx.y * blockDim.y + threadIdx.y;
  if (lx >= rw || ly >= rh) return;

  int x = rx + lx;
  int y = ry + ly;

  // Find enclosing grid cell relative to ROI
  int gi = lx / gridSize;
  int gj = ly / gridSize;
  int gi1 = min(gi + 1, gridW - 1);
  int gj1 = min(gj + 1, gridH - 1);

  // Local position within cell [0, 1)
  float cellW = (gi1 > gi) ? (float)(min((gi + 1) * gridSize, rw - 1) - gi * gridSize) : 1.0f;
  float cellH = (gj1 > gj) ? (float)(min((gj + 1) * gridSize, rh - 1) - gj * gridSize) : 1.0f;
  float fx = (float)(lx - gi * gridSize) / fmaxf(cellW, 1.0f);
  float fy = (float)(ly - gj * gridSize) / fmaxf(cellH, 1.0f);

  // Bilinear interpolation of displacement field
  float dx00 = d_rDx[gj * gridW + gi];
  float dx10 = d_rDx[gj * gridW + gi1];
  float dx01 = d_rDx[gj1 * gridW + gi];
  float dx11 = d_rDx[gj1 * gridW + gi1];

  float dy00 = d_rDy[gj * gridW + gi];
  float dy10 = d_rDy[gj * gridW + gi1];
  float dy01 = d_rDy[gj1 * gridW + gi];
  float dy11 = d_rDy[gj1 * gridW + gi1];

  float deltaX = (dx00 * (1 - fy) + dx01 * fy) * (1 - fx) +
                 (dx10 * (1 - fy) + dx11 * fy) * fx;
  float deltaY = (dy00 * (1 - fy) + dy01 * fy) * (1 - fx) +
                 (dy10 * (1 - fy) + dy11 * fy) * fx;

  // Source coordinate
  float srcX = (float)x + deltaX * transRatio;
  float srcY = (float)y + deltaY * transRatio;

  // Clamp
  srcX = fmaxf(0.0f, fminf(srcX, (float)(W - 1)));
  srcY = fmaxf(0.0f, fminf(srcY, (float)(H - 1)));

  // Bilinear sampling of source image (RGBA)
  int sx0 = (int)floorf(srcX);
  int sy0 = (int)floorf(srcY);
  int sx1 = min(sx0 + 1, W - 1);
  int sy1 = min(sy0 + 1, H - 1);
  float sfx = srcX - sx0;
  float sfy = srcY - sy0;

  const uint8_t* p00 = d_src + sy0 * srcPitch + sx0 * 4;
  const uint8_t* p10 = d_src + sy0 * srcPitch + sx1 * 4;
  const uint8_t* p01 = d_src + sy1 * srcPitch + sx0 * 4;
  const uint8_t* p11 = d_src + sy1 * srcPitch + sx1 * 4;

  uint8_t* out = d_dst + y * dstPitch + x * 4;

  #pragma unroll
  for (int c = 0; c < 4; ++c) {
    float v = (p00[c] * (1 - sfx) + p10[c] * sfx) * (1 - sfy) +
              (p01[c] * (1 - sfx) + p11[c] * sfx) * sfy;
    out[c] = (uint8_t)fminf(fmaxf(v + 0.5f, 0.0f), 255.0f);
  }
}

// ── Host implementation ──

CudaMlsWarp::CudaMlsWarp(const CudaMlsWarpConfig& cfg) : cfg_(cfg) {}

CudaMlsWarp::~CudaMlsWarp() {
  if (d_rDx_) cudaFree(d_rDx_);
  if (d_rDy_) cudaFree(d_rDy_);
  if (d_oldPts_) cudaFree(d_oldPts_);
  if (d_newPts_) cudaFree(d_newPts_);
}

void CudaMlsWarp::setConfig(const CudaMlsWarpConfig& cfg) { cfg_ = cfg; }

void CudaMlsWarp::ensureBuffers(int gridW, int gridH, int nPoints) {
  if (gridW > alloc_grid_w_ || gridH > alloc_grid_h_) {
    if (d_rDx_) cudaFree(d_rDx_);
    if (d_rDy_) cudaFree(d_rDy_);
    size_t sz = (size_t)gridW * gridH * sizeof(float);
    cudaMalloc(&d_rDx_, sz);
    cudaMalloc(&d_rDy_, sz);
    alloc_grid_w_ = gridW;
    alloc_grid_h_ = gridH;
  }
  if (nPoints > max_points_) {
    if (d_oldPts_) cudaFree(d_oldPts_);
    if (d_newPts_) cudaFree(d_newPts_);
    cudaMalloc(&d_oldPts_, nPoints * 2 * sizeof(float));
    cudaMalloc(&d_newPts_, nPoints * 2 * sizeof(float));
    max_points_ = nPoints;
  }
}

void CudaMlsWarp::warp(const uint8_t* d_src_rgba, uint8_t* d_dst_rgba,
                        int width, int height, int srcPitch, int dstPitch,
                        const float* h_src_pts_xy, const float* h_dst_pts_xy,
                        int nPoints, int rx, int ry, int rw, int rh,
                        cudaStream_t stream) {
  if (nPoints < 2) {
    // No meaningful warp possible; copy src to dst
    cudaMemcpy2DAsync(d_dst_rgba, dstPitch, d_src_rgba, srcPitch, width * 4,
                      height, cudaMemcpyDeviceToDevice, stream);
    return;
  }

  int gridSize = cfg_.gridSize;
  int gridW = (rw + gridSize - 1) / gridSize + 1;
  int gridH = (rh + gridSize - 1) / gridSize + 1;

  ensureBuffers(gridW, gridH, nPoints);

  // Pre-scale: compute area ratio (matching CPU behavior)
  float ratio = 1.0f;
  // Temporary host copies for pre-scaling
  std::vector<float> h_old(nPoints * 2);  // dst points (targets)
  std::vector<float> h_new(nPoints * 2);  // src points (landmarks)

  // old = destination (where to warp TO), new = source (where the landmarks ARE)
  std::memcpy(h_old.data(), h_dst_pts_xy, nPoints * 2 * sizeof(float));
  std::memcpy(h_new.data(), h_src_pts_xy, nPoints * 2 * sizeof(float));

  if (cfg_.preScale) {
    // Compute bounding box area for old and new points
    auto calcArea = [](const float* pts, int n) -> float {
      float minx = 1e10f, miny = 1e10f, maxx = -1e10f, maxy = -1e10f;
      for (int i = 0; i < n; ++i) {
        float x = pts[i * 2], y = pts[i * 2 + 1];
        minx = std::min(minx, x);
        miny = std::min(miny, y);
        maxx = std::max(maxx, x);
        maxy = std::max(maxy, y);
      }
      return std::max(0.0f, (maxx - minx) * (maxy - miny));
    };

    float a_old = calcArea(h_old.data(), nPoints);
    float a_new = calcArea(h_new.data(), nPoints);

    if (a_old > 1e-12f && a_new > 1e-12f) {
      ratio = std::sqrt(a_new / a_old);
      if (std::isfinite(ratio) && ratio > 1e-12f) {
        float inv = 1.0f / ratio;
        for (int i = 0; i < nPoints * 2; ++i) {
          h_new[i] *= inv;
        }
      } else {
        ratio = 1.0f;
      }
    }
  }

  // Upload control points to GPU: d_oldPts_ gets targets, d_newPts_ gets landmarks
  cudaMemcpyAsync(d_oldPts_, h_old.data(), nPoints * 2 * sizeof(float),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_newPts_, h_new.data(), nPoints * 2 * sizeof(float),
                  cudaMemcpyHostToDevice, stream);

  // Kernel 1: Compute displacement field
  {
    dim3 block(16, 16);
    dim3 grid((gridW + block.x - 1) / block.x,
              (gridH + block.y - 1) / block.y);
    mls_rigid_delta_kernel<<<grid, block, 0, stream>>>(
        d_rDx_, d_rDy_, d_oldPts_, d_newPts_, nPoints, width, height,
        gridSize, cfg_.alpha, ratio, rx, ry, gridW, gridH);
  }

  // Kernel 2: Apply warp (bilinear interp of displacement + texture sampling)
  {
    dim3 block(16, 16);
    dim3 grid((rw + block.x - 1) / block.x,
              (rh + block.y - 1) / block.y);
    mls_warp_kernel<<<grid, block, 0, stream>>>(
        d_src_rgba, d_dst_rgba, d_rDx_, d_rDy_, width, height, srcPitch,
        dstPitch, gridSize, gridW, gridH, 1.0f, rx, ry, rw, rh);
  }
}
