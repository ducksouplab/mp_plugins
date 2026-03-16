// trt_face_landmarker.cpp
// Two-stage TensorRT face landmark detection:
//   Stage 1: BlazeFace short-range detector (128x128 RGB) -> bounding box
//   Stage 2: Face landmarks regressor (192x192 RGB crop) -> 478 landmarks
//
// TFLite models are extracted from the .task bundle, converted to ONNX offline
// (via convert_models.py), then built into TRT engines at first run.
// Engine files are cached to disk for fast subsequent loads.

#include "trt_face_landmarker.h"
#include "cuda_preprocess.h"
#include "task_model_extractor.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <vector>

// ── TensorRT logger ──

class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::fprintf(stderr, "[TRT %d] %s\n", (int)severity, msg);
    }
  }
};
static TrtLogger g_trt_logger;

// ── Helpers ──

static std::string engine_cache_path(const std::string& onnx_path, bool fp16) {
  // Include GPU architecture in cache name for portability
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  char buf[512];
  std::snprintf(buf, sizeof(buf), "%s.engine_sm%d%d_%s", onnx_path.c_str(),
                prop.major, prop.minor, fp16 ? "fp16" : "fp32");
  return std::string(buf);
}

static bool file_exists(const std::string& path) {
  std::ifstream f(path);
  return f.good();
}

static std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return {};
  auto sz = f.tellg();
  f.seekg(0);
  std::vector<uint8_t> buf(sz);
  f.read(reinterpret_cast<char*>(buf.data()), sz);
  return buf;
}

static bool write_file(const std::string& path, const void* data, size_t sz) {
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  f.write(reinterpret_cast<const char*>(data), sz);
  return f.good();
}

// ── CUDA helpers ──

#define CUDA_CHECK(call)                                                \
  do {                                                                  \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,      \
                   __LINE__, cudaGetErrorString(err));                   \
      return nullptr;                                                   \
    }                                                                   \
  } while (0)

#define CUDA_CHECK_VOID(call)                                           \
  do {                                                                  \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,      \
                   __LINE__, cudaGetErrorString(err));                   \
    }                                                                   \
  } while (0)

// ── Implementation ──

struct TrtFaceLandmarker::Impl {
  TrtFaceLandmarkerConfig cfg;

  // TRT runtime (shared)
  nvinfer1::IRuntime* runtime = nullptr;

  // Stage 1: Face detector
  nvinfer1::ICudaEngine* det_engine = nullptr;
  nvinfer1::IExecutionContext* det_ctx = nullptr;
  static constexpr int DET_W = 128, DET_H = 128;

  // Stage 2: Face landmarks
  nvinfer1::ICudaEngine* lm_engine = nullptr;
  nvinfer1::IExecutionContext* lm_ctx = nullptr;
  static constexpr int LM_W = 192, LM_H = 192;

  // Pre-allocated GPU buffers
  float* d_det_input = nullptr;   // DET_W * DET_H * 3
  float* d_det_output = nullptr;  // detector outputs (boxes + scores)
  float* d_lm_input = nullptr;    // LM_W * LM_H * 3
  float* d_lm_output = nullptr;   // 478 * 3 (landmarks)
  float* d_lm_score = nullptr;    // face presence score

  // Host staging for small outputs
  std::vector<float> h_det_boxes;
  std::vector<float> h_det_scores;
  std::vector<float> h_lm_landmarks;
  float h_lm_score = 0.0f;

  ~Impl() {
    if (d_det_input) cudaFree(d_det_input);
    if (d_det_output) cudaFree(d_det_output);
    if (d_lm_input) cudaFree(d_lm_input);
    if (d_lm_output) cudaFree(d_lm_output);
    if (d_lm_score) cudaFree(d_lm_score);
    if (det_ctx) det_ctx->destroy();
    if (det_engine) det_engine->destroy();
    if (lm_ctx) lm_ctx->destroy();
    if (lm_engine) lm_engine->destroy();
    if (runtime) runtime->destroy();
  }
};

// Build or load a TRT engine from an ONNX file
static nvinfer1::ICudaEngine* build_or_load_engine(
    nvinfer1::IRuntime* runtime, const std::string& onnx_path, bool fp16) {
  std::string cache = engine_cache_path(onnx_path, fp16);

  // Try loading cached engine
  if (file_exists(cache)) {
    std::fprintf(stderr, "[TRT] Loading cached engine: %s\n", cache.c_str());
    auto data = read_file(cache);
    if (!data.empty()) {
      auto* engine = runtime->deserializeCudaEngine(data.data(), data.size());
      if (engine) return engine;
      std::fprintf(stderr, "[TRT] Cache invalid, rebuilding.\n");
    }
  }

  // Build from ONNX
  std::fprintf(stderr, "[TRT] Building engine from ONNX: %s (this may take 20-60s)...\n",
               onnx_path.c_str());

  auto* builder = nvinfer1::createInferBuilder(g_trt_logger);
  if (!builder) return nullptr;

  const auto explicitBatch =
      1U << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto* network = builder->createNetworkV2(explicitBatch);
  if (!network) {
    builder->destroy();
    return nullptr;
  }

  auto* parser = nvonnxparser::createParser(*network, g_trt_logger);
  if (!parser->parseFromFile(onnx_path.c_str(),
                             static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    std::fprintf(stderr, "[TRT] ONNX parse failed for %s\n", onnx_path.c_str());
    parser->destroy();
    network->destroy();
    builder->destroy();
    return nullptr;
  }

  auto* config = builder->createBuilderConfig();
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                             256ULL << 20);  // 256 MB workspace

  if (fp16 && builder->platformHasFastFp16()) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    std::fprintf(stderr, "[TRT] FP16 enabled.\n");
  }

  auto* serialized = builder->buildSerializedNetwork(*network, *config);
  if (!serialized) {
    std::fprintf(stderr, "[TRT] Engine build failed.\n");
    config->destroy();
    network->destroy();
    builder->destroy();
    return nullptr;
  }

  // Cache to disk
  write_file(cache, serialized->data(), serialized->size());
  std::fprintf(stderr, "[TRT] Engine cached to: %s\n", cache.c_str());

  auto* engine =
      runtime->deserializeCudaEngine(serialized->data(), serialized->size());

  serialized->destroy();
  config->destroy();
  parser->destroy();
  network->destroy();
  builder->destroy();

  return engine;
}

TrtFaceLandmarker::~TrtFaceLandmarker() = default;

std::unique_ptr<TrtFaceLandmarker> TrtFaceLandmarker::Create(
    const TrtFaceLandmarkerConfig& cfg) {
  auto self = std::unique_ptr<TrtFaceLandmarker>(new TrtFaceLandmarker());
  self->impl_ = std::make_unique<Impl>();
  self->impl_->cfg = cfg;

  cudaSetDevice(cfg.gpu_id);

  // Determine ONNX model paths.
  // Convention: models stored next to the .task file as:
  //   <dir>/face_detector.onnx
  //   <dir>/face_landmarks.onnx
  // If not found, try to extract from .task and point user to convert script.
  std::string dir;
  auto slash = cfg.model_path.rfind('/');
  if (slash != std::string::npos) {
    dir = cfg.model_path.substr(0, slash + 1);
  } else {
    dir = "./";
  }

  std::string det_onnx = dir + "face_detector.onnx";
  std::string lm_onnx = dir + "face_landmarks.onnx";

  if (!file_exists(det_onnx) || !file_exists(lm_onnx)) {
    self->last_error_ =
        "ONNX models not found. Run convert_models.py first:\n"
        "  python3 convert_models.py " +
        cfg.model_path +
        "\n"
        "Expected files:\n"
        "  " +
        det_onnx + "\n  " + lm_onnx;
    std::fprintf(stderr, "[TRT] %s\n", self->last_error_.c_str());
    return nullptr;
  }

  // Create TRT runtime
  self->impl_->runtime = nvinfer1::createInferRuntime(g_trt_logger);
  if (!self->impl_->runtime) {
    self->last_error_ = "Failed to create TensorRT runtime";
    return nullptr;
  }

  // Build/load detector engine
  self->impl_->det_engine =
      build_or_load_engine(self->impl_->runtime, det_onnx, cfg.fp16);
  if (!self->impl_->det_engine) {
    self->last_error_ = "Failed to build detector TRT engine from " + det_onnx;
    return nullptr;
  }
  self->impl_->det_ctx =
      self->impl_->det_engine->createExecutionContext();

  // Build/load landmarks engine
  self->impl_->lm_engine =
      build_or_load_engine(self->impl_->runtime, lm_onnx, cfg.fp16);
  if (!self->impl_->lm_engine) {
    self->last_error_ = "Failed to build landmarks TRT engine from " + lm_onnx;
    return nullptr;
  }
  self->impl_->lm_ctx =
      self->impl_->lm_engine->createExecutionContext();

  // Allocate GPU buffers
  auto& I = *self->impl_;
  CUDA_CHECK(cudaMalloc(&I.d_det_input,
                        I.DET_W * I.DET_H * 3 * sizeof(float)));
  // Detector output: typically 896 anchors * 17 values (bbox + keypoints) + 896 scores
  // Allocate generous buffer
  CUDA_CHECK(
      cudaMalloc(&I.d_det_output, 896 * 17 * sizeof(float) + 896 * sizeof(float)));

  CUDA_CHECK(
      cudaMalloc(&I.d_lm_input, I.LM_W * I.LM_H * 3 * sizeof(float)));
  // Landmarks output: 478 * 3 floats + 1 score
  CUDA_CHECK(cudaMalloc(&I.d_lm_output, 478 * 3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&I.d_lm_score, sizeof(float)));

  I.h_det_boxes.resize(896 * 17);
  I.h_det_scores.resize(896);
  I.h_lm_landmarks.resize(478 * 3);

  std::fprintf(stderr, "[TRT] Face landmarker initialized (det=%dx%d, lm=%dx%d, fp16=%d)\n",
               I.DET_W, I.DET_H, I.LM_W, I.LM_H, (int)cfg.fp16);
  return self;
}

// BlazeFace SSD anchor generation for 128x128
static std::vector<std::array<float, 2>> generate_anchors_128() {
  // BlazeFace uses a specific anchor scheme:
  // Feature maps at strides 8 and 16, 2 anchors per cell
  std::vector<std::array<float, 2>> anchors;
  const int strides[] = {8, 16};
  const int anchor_counts[] = {2, 6};

  for (int s = 0; s < 2; ++s) {
    int stride = strides[s];
    int grid = 128 / stride;
    for (int y = 0; y < grid; ++y) {
      for (int x = 0; x < grid; ++x) {
        float cx = (x + 0.5f) / (float)grid;
        float cy = (y + 0.5f) / (float)grid;
        for (int a = 0; a < anchor_counts[s]; ++a) {
          anchors.push_back({cx, cy});
        }
      }
    }
  }
  return anchors;
}

// Decode BlazeFace detector output
struct DetectedFace {
  float cx, cy, w, h;  // normalized center + size
  float score;
  // 6 keypoints (eye, ear, nose, mouth) - used for face crop alignment
  float kp[6][2];
};

static std::vector<DetectedFace> decode_detections(
    const float* raw_boxes, const float* raw_scores,
    const std::vector<std::array<float, 2>>& anchors, float threshold) {
  std::vector<DetectedFace> faces;
  int n = (int)anchors.size();

  for (int i = 0; i < n; ++i) {
    // Score: sigmoid(raw_score)
    float score = 1.0f / (1.0f + std::exp(-raw_scores[i]));
    if (score < threshold) continue;

    DetectedFace f;
    f.score = score;

    // Decode box: center offset + size, relative to anchor
    float ax = anchors[i][0];
    float ay = anchors[i][1];

    f.cx = raw_boxes[i * 17 + 0] / 128.0f + ax;
    f.cy = raw_boxes[i * 17 + 1] / 128.0f + ay;
    f.w = raw_boxes[i * 17 + 2] / 128.0f;
    f.h = raw_boxes[i * 17 + 3] / 128.0f;

    // 6 keypoints
    for (int k = 0; k < 6; ++k) {
      f.kp[k][0] = raw_boxes[i * 17 + 4 + k * 2 + 0] / 128.0f + ax;
      f.kp[k][1] = raw_boxes[i * 17 + 4 + k * 2 + 1] / 128.0f + ay;
    }

    faces.push_back(f);
  }

  // Simple NMS by score
  std::sort(faces.begin(), faces.end(),
            [](const DetectedFace& a, const DetectedFace& b) {
              return a.score > b.score;
            });

  return faces;
}

GpuLandmarkResult TrtFaceLandmarker::detect(const uint8_t* d_rgba, int width,
                                             int height, int pitch,
                                             cudaStream_t stream) {
  GpuLandmarkResult result;
  auto& I = *impl_;

  // ── Stage 1: Face Detection ──

  // Preprocess: resize full frame to 128x128 RGB float
  cuda_rgba_to_rgb_resize_normalize(d_rgba, width, height, pitch,
                                    I.d_det_input, I.DET_W, I.DET_H,
                                    /*chw=*/false, stream);

  // Run detector inference
  // Binding layout depends on the model; typical BlazeFace has:
  //   input[0]: 1x128x128x3 (HWC float)
  //   output[0]: 1x896x17 (boxes + keypoints)
  //   output[1]: 1x896x1 (scores)
  float* det_boxes_d = I.d_det_output;
  float* det_scores_d = I.d_det_output + 896 * 17;

  void* det_bindings[] = {I.d_det_input, det_boxes_d, det_scores_d};
  I.det_ctx->enqueueV2(det_bindings, stream, nullptr);

  // Copy small detection output to CPU for decoding
  cudaMemcpyAsync(I.h_det_boxes.data(), det_boxes_d,
                  896 * 17 * sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(I.h_det_scores.data(), det_scores_d,
                  896 * sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // Decode detections
  static auto anchors = generate_anchors_128();
  auto faces =
      decode_detections(I.h_det_boxes.data(), I.h_det_scores.data(), anchors,
                        I.cfg.det_threshold);

  if (faces.empty()) {
    result.face_count = 0;
    return result;
  }

  // Process up to max_faces
  int n_faces = std::min((int)faces.size(), I.cfg.max_faces);
  result.faces.resize(n_faces);

  for (int fi = 0; fi < n_faces; ++fi) {
    auto& det = faces[fi];

    // ── Stage 2: Face Landmarks ──

    // Compute crop ROI from detection bbox (expand slightly for landmark model)
    float roi_cx = det.cx * width;
    float roi_cy = det.cy * height;
    float roi_w = det.w * width * 1.5f;   // 1.5x expansion for landmark model
    float roi_h = det.h * height * 1.5f;

    int roi_x = std::max(0, (int)(roi_cx - roi_w / 2));
    int roi_y = std::max(0, (int)(roi_cy - roi_h / 2));
    int roi_right = std::min(width, (int)(roi_cx + roi_w / 2));
    int roi_bottom = std::min(height, (int)(roi_cy + roi_h / 2));
    int crop_w = roi_right - roi_x;
    int crop_h = roi_bottom - roi_y;

    if (crop_w < 10 || crop_h < 10) continue;

    // Preprocess: crop + resize to 192x192 RGB float
    cuda_crop_rgba_to_rgb_resize_normalize(
        d_rgba, width, height, pitch, I.d_lm_input, I.LM_W, I.LM_H, roi_x,
        roi_y, crop_w, crop_h,
        /*chw=*/false, stream);

    // Run landmark inference
    // Input: 1x192x192x3, Output: 1x1404 (478*3) + 1x1 (score)
    void* lm_bindings[] = {I.d_lm_input, I.d_lm_output, I.d_lm_score};
    I.lm_ctx->enqueueV2(lm_bindings, stream, nullptr);

    // Copy landmarks to CPU (tiny: 478*3*4 = ~6KB)
    cudaMemcpyAsync(I.h_lm_landmarks.data(), I.d_lm_output,
                    478 * 3 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&I.h_lm_score, I.d_lm_score, sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Fill result
    auto& face_out = result.faces[fi];
    face_out.bbox[0] = det.cx;
    face_out.bbox[1] = det.cy;
    face_out.bbox[2] = det.w;
    face_out.bbox[3] = det.h;
    face_out.score = det.score;

    // Transform landmarks from crop-local [0,1] to full-image [0,1]
    for (int li = 0; li < 478; ++li) {
      float lx = I.h_lm_landmarks[li * 3 + 0];  // x in [0, LM_W]
      float ly = I.h_lm_landmarks[li * 3 + 1];  // y in [0, LM_H]
      float lz = I.h_lm_landmarks[li * 3 + 2];  // z (depth)

      // Map from 192x192 crop back to full image coordinates
      // Landmark coords are in pixel space of the 192x192 input
      float img_x = roi_x + (lx / (float)I.LM_W) * crop_w;
      float img_y = roi_y + (ly / (float)I.LM_H) * crop_h;

      // Normalize to [0,1] in full image
      face_out.landmarks[li * 3 + 0] = img_x / (float)width;
      face_out.landmarks[li * 3 + 1] = img_y / (float)height;
      face_out.landmarks[li * 3 + 2] = lz;
    }
  }

  result.face_count = n_faces;
  return result;
}
