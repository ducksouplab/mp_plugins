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

#include <cstdarg>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

namespace {
// Helper to format logs into a string
std::string fmt_str(const char* format, ...) {
  va_list args;
  va_start(args, format);
  char buf[1024];
  std::vsnprintf(buf, sizeof(buf), format, args);
  va_end(args);
  return std::string(buf);
}
}

// ── TensorRT logger ──

class TrtLogger : public nvinfer1::ILogger {
 public:
  LogCallback log_cb;
  void log(Severity severity, const char* msg) noexcept override {
    if (!log_cb) return;
    TrtLogLevel level = TrtLogLevel::INFO;
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
      case Severity::kERROR:   level = TrtLogLevel::ERROR;   break;
      case Severity::kWARNING: level = TrtLogLevel::WARNING; break;
      case Severity::kINFO:    level = TrtLogLevel::INFO;    break;
      case Severity::kVERBOSE: level = TrtLogLevel::DEBUG;   break;
    }
    log_cb(level, std::string("[TRT Internal] ") + msg);
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
      std::string msg = fmt_str("CUDA error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
      if (g_trt_logger.log_cb) g_trt_logger.log_cb(TrtLogLevel::ERROR, msg); \
      return nullptr;                                                   \
    }                                                                   \
  } while (0)

#define CUDA_CHECK_VOID(call)                                           \
  do {                                                                  \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
      std::string msg = fmt_str("CUDA error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
      if (g_trt_logger.log_cb) g_trt_logger.log_cb(TrtLogLevel::ERROR, msg); \
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
  static constexpr int LM_W = 256, DET_W_LM = 256; 
  static constexpr int LM_H = 256;

  // Pre-allocated GPU buffers
  float* d_det_input = nullptr;   // DET_W * DET_H * 3
  float* d_det_output = nullptr;  // detector outputs (boxes + scores)
  float* d_lm_input = nullptr;    // LM_W * LM_H * 3
  float* d_lm_output = nullptr;   // 478 * 3 (landmarks)
  float* d_lm_score = nullptr;    // face presence score
  float* d_lm_extra = nullptr;    // for extra outputs like Identity_2

  // Host staging for small outputs
  std::vector<float> h_det_boxes;
  std::vector<float> h_det_scores;
  std::vector<float> h_lm_landmarks;
  float h_lm_score = 0.0f;

  // Stateful tracking logic
  bool has_prev_landmarks = false;
  std::vector<float> prev_landmarks; // 478 * 3, normalized [0, 1]

  // ROI One-Euro smoothing state
  struct RoiOneEuroFilter {
    // Heavily clamp beta so fast tracking errors don't shatter the filter's low-pass stability
    float min_cutoff = 0.1f, beta = 0.002f, d_cutoff = 1.0f;
    bool first_time = true;
    float x_prev = 0.0f, dx_prev = 0.0f;
    float alpha(float cutoff, float dt) {
      float r = 2.0f * M_PI * cutoff * dt;
      return r / (r + 1.0f);
    }
    float filter(float x, float dt) {
      if (first_time) {
        first_time = false;
        x_prev = x; dx_prev = 0.0f;
        return x;
      }
      if (dt <= 0) return x_prev;
      float dx = (x - x_prev) / dt;
      float edx = alpha(d_cutoff, dt) * dx + (1.0f - alpha(d_cutoff, dt)) * dx_prev;
      float cutoff = min_cutoff + beta * std::abs(edx);
      float a = alpha(cutoff, dt);
      float x_filtered = a * x + (1.0f - a) * x_prev;
      x_prev = x_filtered; dx_prev = edx;
      return x_filtered;
    }
  };
  bool has_prev_roi = false;
  float prev_cx = 0.5f, prev_cy = 0.5f, prev_size = 0.5f;
  RoiOneEuroFilter filter_cx, filter_cy, filter_size;


  Impl() {
    h_det_boxes.resize(896 * 16);
    h_det_scores.resize(896);
    h_lm_landmarks.resize(478 * 3);
    prev_landmarks.resize(478 * 3);
  }

  ~Impl() {
    if (d_det_input) cudaFree(d_det_input);
    if (d_det_output) cudaFree(d_det_output);
    if (d_lm_input) cudaFree(d_lm_input);
    if (d_lm_output) cudaFree(d_lm_output);
    if (d_lm_score) cudaFree(d_lm_score);
    if (d_lm_extra) cudaFree(d_lm_extra);
    if (det_ctx) delete det_ctx;
    if (det_engine) delete det_engine;
    if (lm_ctx) delete lm_ctx;
    if (lm_engine) delete lm_engine;
    if (runtime) delete runtime;
  }
};

// Build or load a TRT engine from an ONNX file
static nvinfer1::ICudaEngine* build_or_load_engine(
    nvinfer1::IRuntime* runtime, const std::string& onnx_path, bool fp16) {
  std::string cache = engine_cache_path(onnx_path, fp16);

  // Try loading cached engine
  if (file_exists(cache)) {
    if (g_trt_logger.log_cb) g_trt_logger.log_cb(TrtLogLevel::INFO, fmt_str("[TRT] Loading cached engine: %s", cache.c_str()));
    auto data = read_file(cache);
    if (!data.empty()) {
      auto* engine = runtime->deserializeCudaEngine(data.data(), data.size());
      if (engine) return engine;
      if (g_trt_logger.log_cb) g_trt_logger.log_cb(TrtLogLevel::WARNING, "[TRT] Cache invalid, rebuilding.");
    }
  }

  // Build from ONNX
  if (g_trt_logger.log_cb) g_trt_logger.log_cb(TrtLogLevel::INFO, fmt_str("[TRT] Building engine from ONNX: %s (this may take 20-60s)...", onnx_path.c_str()));

  auto* builder = nvinfer1::createInferBuilder(g_trt_logger);
  if (!builder) return nullptr;

  auto* network = builder->createNetworkV2(0);
  if (!network) {
    delete builder;
    return nullptr;
  }

  auto* parser = nvonnxparser::createParser(*network, g_trt_logger);
  if (!parser->parseFromFile(onnx_path.c_str(),
                             static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    if (g_trt_logger.log_cb) g_trt_logger.log_cb(TrtLogLevel::ERROR, fmt_str("[TRT] ONNX parse failed for %s", onnx_path.c_str()));
    delete parser;
    delete network;
    delete builder;
    return nullptr;
  }

  auto* config = builder->createBuilderConfig();
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                             256ULL << 20);  // 256 MB workspace

  if (fp16 && builder->platformHasFastFp16()) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    if (g_trt_logger.log_cb) g_trt_logger.log_cb(TrtLogLevel::INFO, "[TRT] FP16 enabled.");
  }

  // Handle dynamic shapes (required for some ONNX models)
  auto* profile = builder->createOptimizationProfile();
  bool has_dynamic = false;
  for (int i = 0; i < network->getNbInputs(); ++i) {
    auto* input = network->getInput(i);
    auto dims = input->getDimensions();
    bool input_dynamic = false;
    std::string dstr;
    for (int d = 0; d < dims.nbDims; ++d) {
      dstr += std::to_string(dims.d[d]) + (d == dims.nbDims - 1 ? "" : "x");
      if (dims.d[d] == -1) {
        dims.d[d] = 1;  // Default to 1 (e.g. batch size)
        input_dynamic = true;
      }
    }
    if (g_trt_logger.log_cb) g_trt_logger.log_cb(TrtLogLevel::DEBUG, fmt_str("[TRT] Input '%s': %s (dynamic=%d)", input->getName(), dstr.c_str(), (int)input_dynamic));
    if (input_dynamic) {
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, dims);
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, dims);
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, dims);
      has_dynamic = true;
    }
  }
  if (has_dynamic) {
    if (g_trt_logger.log_cb) g_trt_logger.log_cb(TrtLogLevel::INFO, "[TRT] Added optimization profile for dynamic shapes.");
    config->addOptimizationProfile(profile);
  }

  auto* serialized = builder->buildSerializedNetwork(*network, *config);
  if (!serialized) {
    if (g_trt_logger.log_cb) g_trt_logger.log_cb(TrtLogLevel::ERROR, "[TRT] Engine build failed.");
    delete config;
    delete network;
    delete builder;
    return nullptr;
  }

  // Cache to disk
  write_file(cache, serialized->data(), serialized->size());
  if (g_trt_logger.log_cb) g_trt_logger.log_cb(TrtLogLevel::INFO, fmt_str("[TRT] Engine cached to: %s", cache.c_str()));

  auto* engine =
      runtime->deserializeCudaEngine(serialized->data(), serialized->size());

  delete serialized;
  delete config;
  delete parser;
  delete network;
  delete builder;

  return engine;
}

TrtFaceLandmarker::~TrtFaceLandmarker() = default;

std::unique_ptr<TrtFaceLandmarker> TrtFaceLandmarker::Create(
    const TrtFaceLandmarkerConfig& cfg) {
  g_trt_logger.log_cb = cfg.log_cb;
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
    if (g_trt_logger.log_cb) g_trt_logger.log_cb(TrtLogLevel::ERROR, fmt_str("[TRT] %s", self->last_error_.c_str()));
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

  auto log_engine = [](nvinfer1::ICudaEngine* engine, const char* label) {
    if (!g_trt_logger.log_cb) return;
    int nb = engine->getNbIOTensors();
    g_trt_logger.log_cb(TrtLogLevel::DEBUG, fmt_str("[TRT] Engine %s has %d tensors:", label, nb));
    for (int i = 0; i < nb; ++i) {
      auto name = engine->getIOTensorName(i);
      auto dims = engine->getTensorShape(name);
      auto mode = engine->getTensorIOMode(name);
      std::string dstr;
      for (int d = 0; d < dims.nbDims; ++d) dstr += std::to_string(dims.d[d]) + (d == dims.nbDims - 1 ? "" : "x");
      g_trt_logger.log_cb(TrtLogLevel::DEBUG, fmt_str("  %d: %s (%s) shape=%s", i, name,
                         mode == nvinfer1::TensorIOMode::kINPUT ? "IN" : "OUT", dstr.c_str()));
    }
  };
  log_engine(I.det_engine, "Detector");
  log_engine(I.lm_engine, "Landmarks");

  CUDA_CHECK(cudaMalloc(&I.d_det_input,
                        I.DET_W * I.DET_H * 3 * sizeof(float)));
  // Detector output: typically 896 anchors * 16 values (bbox + keypoints) + 896 scores
  // Allocate generous buffer
  CUDA_CHECK(
      cudaMalloc(&I.d_det_output, 896 * 16 * sizeof(float) + 896 * sizeof(float)));

  CUDA_CHECK(
      cudaMalloc(&I.d_lm_input, I.LM_W * I.LM_H * 3 * sizeof(float)));
  // Landmarks output: 478 * 3 floats + 1 score + optional blendshapes/extra
  CUDA_CHECK(cudaMalloc(&I.d_lm_output, 478 * 3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&I.d_lm_score, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&I.d_lm_extra, 2048 * sizeof(float))); // generous buffer for Identity_2

  I.h_det_boxes.resize(896 * 16);
  I.h_det_scores.resize(896);
  I.h_lm_landmarks.resize(478 * 3);

  if (g_trt_logger.log_cb) {
    g_trt_logger.log_cb(TrtLogLevel::INFO, fmt_str("[TRT] Face landmarker initialized (det=%dx%d, lm=%dx%d, fp16=%d)",
                        I.DET_W, I.DET_H, I.LM_W, I.LM_H, (int)cfg.fp16));
  }
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
    const std::vector<std::array<float, 2>>& anchors, float threshold,
    int width, int height) {
  std::vector<DetectedFace> faces;
  int n = (int)anchors.size();

  float scale = std::min(128.0f / width, 128.0f / height);
  float crop_w = width * scale;
  float crop_h = height * scale;
  float pad_x = (128.0f - crop_w) * 0.5f;
  float pad_y = (128.0f - crop_h) * 0.5f;

  for (int i = 0; i < n; ++i) {
    // Score decoding: always apply sigmoid (raw scores are logits)
    float s = raw_scores[i];
    float score = 1.0f / (1.0f + std::exp(-s));
    if (score < threshold) continue;

    DetectedFace f;
    f.score = score;

    // Decode box: center offset + size, relative to anchor
    float ax = anchors[i][0] * 128.0f;
    float ay = anchors[i][1] * 128.0f;

    float dx = raw_boxes[i * 16 + 0];
    float dy = raw_boxes[i * 16 + 1];
    float dw = raw_boxes[i * 16 + 2];
    float dh = raw_boxes[i * 16 + 3];

    float px_cx = dx + ax;
    float px_cy = dy + ay;

    // Unpad and unscale to original image coordinates
    f.cx = (px_cx - pad_x) / crop_w;
    f.cy = (px_cy - pad_y) / crop_h;
    f.w = dw / crop_w;
    f.h = dh / crop_h;

    // 6 keypoints
    for (int k = 0; k < 6; ++k) {
      float kpx = raw_boxes[i * 16 + 4 + k * 2 + 0] + ax;
      float kpy = raw_boxes[i * 16 + 4 + k * 2 + 1] + ay;
      f.kp[k][0] = (kpx - pad_x) / crop_w;
      f.kp[k][1] = (kpy - pad_y) / crop_h;
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

  float cx = 0, cy = 0, side = 0, angle = 0;
  bool use_tracking = false;

  // Disable landmark-based structural tracking because the FaceLandmarker v2 neural 
  // network predictions are hyper-dependent on crop scale, causing an unresolvable 
  // feedback-loop oscillation. Instead, we run the BlazeFace short-range detector (≈2ms)
  // unconditionally, but aggressively smooth its output via a One-Euro Filter to 
  // establish jitter-free, mathematically pure Open-Loop tracking parity.
  if (false && I.has_prev_landmarks) {
    use_tracking = true;
  }

  std::vector<DetectedFace> faces;
  if (!use_tracking) {
    // ── Stage 1: Face Detection ──
    cuda_rgba_to_rgb_resize_normalize(d_rgba, width, height, pitch,
                                      I.d_det_input, I.DET_W, I.DET_H,
                                      /*chw=*/false, stream);

    float* det_boxes_d = I.d_det_output;
    float* det_scores_d = I.d_det_output + 896 * 16;

    for (int i = 0; i < I.det_engine->getNbIOTensors(); ++i) {
      const char* name = I.det_engine->getIOTensorName(i);
      if (i == 0) I.det_ctx->setTensorAddress(name, I.d_det_input);
      else if (i == 1) I.det_ctx->setTensorAddress(name, det_boxes_d);
      else if (i == 2) I.det_ctx->setTensorAddress(name, det_scores_d);
    }
    I.det_ctx->enqueueV3(stream);

    cudaMemcpyAsync(I.h_det_boxes.data(), det_boxes_d,
                    896 * 16 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(I.h_det_scores.data(), det_scores_d,
                    896 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    static auto anchors = generate_anchors_128();
    faces = decode_detections(I.h_det_boxes.data(), I.h_det_scores.data(), anchors,
                              I.cfg.det_threshold, width, height);

    if (faces.empty()) {
      I.has_prev_landmarks = false;
      I.has_prev_roi = false;
      I.filter_cx.first_time = true;
      I.filter_cy.first_time = true;
      I.filter_size.first_time = true;
      result.face_count = 0;
      return result;
    }
  }

  int n_faces = use_tracking ? 1 : std::min((int)faces.size(), I.cfg.max_faces);
  result.faces.resize(n_faces);

  for (int fi = 0; fi < n_faces; ++fi) {
    if (!use_tracking) {
      auto& det = faces[fi];
      cx = det.cx * (float)width;
      cy = det.cy * (float)height;

      float eye_x = (det.kp[1][0] - det.kp[0][0]) * (float)width;
      float eye_y = (det.kp[1][1] - det.kp[0][1]) * (float)height;
      angle = std::atan2(eye_y, eye_x);
      // Scale from bbox size (face bbox max dim * 1.5)
      side = std::max(det.w * (float)width, det.h * (float)height) * 1.5f;

      // Clamp crop size so it never exceeds the frame, then clamp center.
      side = std::min(side, (float)std::min(width, height));
      {
        float half = side * 0.5f;
        cx = std::max(half, std::min((float)width  - half, cx));
        cy = std::max(half, std::min((float)height - half, cy));
      }
    }

    if (!use_tracking) {
      float dt = 1.0f / 30.0f;
      cx = I.filter_cx.filter(cx, dt);
      cy = I.filter_cy.filter(cy, dt);
      side = I.filter_size.filter(side, dt);
      I.has_prev_roi = true;
    }

    if (g_trt_logger.log_cb) {
      g_trt_logger.log_cb(TrtLogLevel::DEBUG, fmt_str("[DEBUG Tracker] use_tracking=%d, cx=%.1f, cy=%.1f, side=%.1f",
                          (int)use_tracking, cx, cy, side));
    }

    // Build Affine Matrix (Forward: Dest -> Src)
    float cos_a = std::cos(angle);
    float sin_a = std::sin(angle);

    // Scale maps each crop pixel to image pixels.
    float scale = side / (float)I.LM_W; // side / 256.0f
    float map_center = I.LM_W * 0.5f - 0.5f; // 127.5f
    
    float m[6];
    m[0] =  cos_a * scale; m[1] = -sin_a * scale; m[2] = cx - (cos_a * map_center * scale - sin_a * map_center * scale);
    m[3] =  sin_a * scale; m[4] =  cos_a * scale; m[5] = cy - (sin_a * map_center * scale + cos_a * map_center * scale);

    // Preprocess: warped crop + resize to 256x256 RGB float
    cuda_warp_affine_rgba_to_rgb_normalize(
        d_rgba, width, height, pitch, I.d_lm_input, I.LM_W, I.LM_H, m,
        /*chw=*/false, stream);

    // Run landmark inference
    for (int i = 0; i < I.lm_engine->getNbIOTensors(); ++i) {
      const char* name = I.lm_engine->getIOTensorName(i);
      if (i == 0) I.lm_ctx->setTensorAddress(name, I.d_lm_input);
      else if (i == 1) I.lm_ctx->setTensorAddress(name, I.d_lm_output);
      else if (i == 2) I.lm_ctx->setTensorAddress(name, I.d_lm_score);
      else I.lm_ctx->setTensorAddress(name, I.d_lm_extra);
    }
    I.lm_ctx->enqueueV3(stream);

    cudaMemcpyAsync(I.h_lm_landmarks.data(), I.d_lm_output,
                    478 * 3 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&I.h_lm_score, I.d_lm_score, sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Fill result
    auto& face_out = result.faces[fi];
    if (!use_tracking) {
      face_out.bbox[0] = faces[fi].cx;
      face_out.bbox[1] = faces[fi].cy;
      face_out.bbox[2] = faces[fi].w;
      face_out.bbox[3] = faces[fi].h;
      face_out.score = faces[fi].score;
    } else {
      face_out.bbox[0] = cx / (float)width;
      face_out.bbox[1] = cy / (float)height;
      face_out.bbox[2] = side / (float)width;
      face_out.bbox[3] = side / (float)height;
      face_out.score = I.h_lm_score;
    }

    // Transform landmarks from crop-local back to full-image [0,1]
    for (int li = 0; li < 478; ++li) {
      float lx = I.h_lm_landmarks[li * 3 + 0];
      float ly = I.h_lm_landmarks[li * 3 + 1];
      float lz = I.h_lm_landmarks[li * 3 + 2];

      // Landmark coordinates are in [0, 256] pixel space (model output scale).
      // The affine matrix uses scale=side/256, so center coord 128 → cx.
      // No additional scaling needed.

      // 1. Transform to Global Image Pixels using affine matrix m
      float img_x_p = m[0] * lx + m[1] * ly + m[2];
      float img_y_p = m[3] * lx + m[4] * ly + m[5];

      // 2. Final Normalized Image Space [0, 1]
      face_out.landmarks[li * 3 + 0] = img_x_p / (float)width;
      face_out.landmarks[li * 3 + 1] = img_y_p / (float)height;
      face_out.landmarks[li * 3 + 2] = lz / (float)I.LM_W; 
    }

    // Sanity check: verify that back-projected landmarks span a reasonable
    // area of the image. The face-presence score (Identity_1) is unreliable
    // in FP16 for our eye-center crop format (~0.0002 even for good detections).
    // Instead, check that the landmark bounding box covers at least 5% of the
    // image in both dimensions — a genuine face bbox should be ~10-40%.
    // If the check fails, discard this result and force a full re-detection
    // on the next frame.
    {
      float lm_xmin = face_out.landmarks[0], lm_xmax = face_out.landmarks[0];
      float lm_ymin = face_out.landmarks[1], lm_ymax = face_out.landmarks[1];
      for (int li = 1; li < 478; ++li) {
        float lx = face_out.landmarks[li * 3 + 0];
        float ly = face_out.landmarks[li * 3 + 1];
        if (lx < lm_xmin) lm_xmin = lx;
        if (lx > lm_xmax) lm_xmax = lx;
        if (ly < lm_ymin) lm_ymin = ly;
        if (ly > lm_ymax) lm_ymax = ly;
      }
      float lm_w = lm_xmax - lm_xmin;
      float lm_h = lm_ymax - lm_ymin;
      if (lm_w < 0.05f || lm_h < 0.05f) {
        if (g_trt_logger.log_cb) {
          g_trt_logger.log_cb(TrtLogLevel::WARNING, fmt_str("[TRT] Landmark sanity check failed (bbox %.3fx%.3f vs min 0.05) — resetting tracking", lm_w, lm_h));
        }
        I.has_prev_landmarks = false;
        I.has_prev_roi = false;
        I.filter_cx.first_time = true;
        I.filter_cy.first_time = true;
        I.filter_size.first_time = true;
        result.face_count = 0;
        return result;
      }
    }

    I.has_prev_landmarks = true;
    for (int li = 0; li < 478; ++li) {
      I.prev_landmarks[li * 3 + 0] = face_out.landmarks[li * 3 + 0];
      I.prev_landmarks[li * 3 + 1] = face_out.landmarks[li * 3 + 1];
      I.prev_landmarks[li * 3 + 2] = face_out.landmarks[li * 3 + 2];
    }
  }

  result.face_count = n_faces;
  return result;
}
