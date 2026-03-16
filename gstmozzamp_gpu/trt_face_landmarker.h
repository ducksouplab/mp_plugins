// trt_face_landmarker.h
// TensorRT-based two-stage face landmark detection.
// Stage 1: BlazeFace detector (128x128) -> face bounding box
// Stage 2: Face landmarks (192x192 crop) -> 478 (x,y,z) landmarks
//
// Replaces MediaPipe FaceLandmarker for NVIDIA GPU servers.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

struct TrtFaceLandmarkerConfig {
  std::string model_path;   // Path to .task bundle or directory with ONNX/engine
  int max_faces = 1;
  bool fp16 = true;         // Use FP16 precision (much faster on NVIDIA GPUs)
  int gpu_id = 0;
  float det_threshold = 0.5f;  // Face detection confidence threshold
};

struct GpuLandmarkResult {
  int face_count = 0;
  // Per-face data (only first max_faces are filled)
  struct Face {
    float bbox[4];  // x_center, y_center, width, height (normalized 0..1)
    float score;
    std::array<float, 478 * 3> landmarks;  // (x,y,z) normalized to full image
  };
  std::vector<Face> faces;
};

class TrtFaceLandmarker {
 public:
  ~TrtFaceLandmarker();

  // Factory: creates and initializes engines.
  // First call may take 20-60s to build TRT engines (cached on subsequent runs).
  static std::unique_ptr<TrtFaceLandmarker> Create(
      const TrtFaceLandmarkerConfig& cfg);

  // Run two-stage detection on a GPU RGBA buffer.
  // d_rgba: device pointer to RGBA frame (width x height, pitch bytes per row)
  // Returns landmarks in normalized [0,1] image coordinates.
  GpuLandmarkResult detect(const uint8_t* d_rgba, int width, int height,
                           int pitch, cudaStream_t stream);

  // Get the last error message.
  const char* last_error() const { return last_error_.c_str(); }

 private:
  TrtFaceLandmarker() = default;

  // Forward declaration of implementation details
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string last_error_;
};
