// cuda_mls_warp.h
// GPU-accelerated Moving Least Squares (MLS) rigid warp.
// Direct port of imgwarp_mls_rigid.cpp::calcDelta() + imgwarp_mls.cpp::genNewImg()
// to CUDA kernels. Achieves ~0.5ms per frame at 640x480 vs ~5ms on CPU.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

struct CudaMlsWarpConfig {
  int gridSize = 5;       // Grid spacing for displacement field (same as CPU version)
  float alpha = 1.4f;     // MLS rigidity parameter (same as CPU version)
  bool preScale = true;   // Allow uniform pre-scaling (same as CPU version)
};

class CudaMlsWarp {
 public:
  CudaMlsWarp(const CudaMlsWarpConfig& cfg);
  ~CudaMlsWarp();

  // Warp an RGBA image on the GPU using MLS rigid transformation.
  // d_src_rgba: input RGBA image on device (width x height, srcPitch bytes/row)
  // d_dst_rgba: output RGBA image on device (same dimensions, dstPitch bytes/row)
  //             Can be a separate buffer (recommended) or same as src.
  // h_src_pts, h_dst_pts: control points on HOST memory (float2: x,y in pixels)
  // nPoints: number of control points
  // All GPU work is submitted to the given stream.
  void warp(const uint8_t* d_src_rgba, uint8_t* d_dst_rgba, int width, int height,
            int srcPitch, int dstPitch, const float* h_src_pts_xy,
            const float* h_dst_pts_xy, int nPoints, int rx, int ry, int rw,
            int rh, cudaStream_t stream);

  // Update config at runtime (e.g., when user changes mls-alpha)
  void setConfig(const CudaMlsWarpConfig& cfg);

 private:
  CudaMlsWarpConfig cfg_;

  // Pre-allocated device buffers
  float* d_rDx_ = nullptr;      // Displacement field X (gridH x gridW)
  float* d_rDy_ = nullptr;      // Displacement field Y (gridH x gridW)
  float* d_oldPts_ = nullptr;   // Control points destination/target (interleaved x,y)
  float* d_newPts_ = nullptr;   // Control points source/landmarks (interleaved x,y)
  int max_points_ = 0;
  int alloc_grid_w_ = 0;
  int alloc_grid_h_ = 0;

  void ensureBuffers(int gridW, int gridH, int nPoints);
};
