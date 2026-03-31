// GStreamer GPU-accelerated facial deformation plugin.
// Drop-in replacement for mozza_mp that uses TensorRT + CUDA instead of
// MediaPipe + OpenCV for ~10x speedup on NVIDIA GPUs.
//
// Element: mozza_mp_gpu
// Props: same as mozza_mp (model, deform, alpha, mls-alpha, mls-grid, etc.)
//        + gpu-id (int, GPU device index, default 0)
//
// Caps: video/x-raw(memory:NVMM), format=RGBA  (zero-copy GPU path)
//       video/x-raw, format=RGBA                (CPU fallback with upload/download)

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <vector>

#include <cuda_runtime.h>

#include <opencv2/core.hpp>

// Reuse DFM parsing and group building from mozza_mp
#include "dfm.hpp"
#include "deform_utils.hpp"

// GPU components
#include "trt_face_landmarker.h"
#include "cuda_mls_warp.h"

#ifndef PACKAGE
#define PACKAGE "mozza_mp_gpu"
#endif

GST_DEBUG_CATEGORY_STATIC(gst_mozza_mp_gpu_debug_category);
#define GST_CAT_DEFAULT gst_mozza_mp_gpu_debug_category

G_BEGIN_DECLS

#define GST_TYPE_MOZZA_MP_GPU (gst_mozza_mp_gpu_get_type())
G_DECLARE_FINAL_TYPE(GstMozzaMpGpu, gst_mozza_mp_gpu, GST, MOZZA_MP_GPU,
                     GstVideoFilter)

enum WarpMode {
  WARP_GLOBAL = 0,
  WARP_PER_GROUP_ROI = 1,
};

// ── One-Euro Filter for Landmark Smoothing (Matching MediaPipe) ──
struct OneEuroFilter {
  float min_cutoff, beta, d_cutoff;
  bool first_time = true;
  float x_prev, dx_prev;
  // min_cutoff: lower = more smoothing at rest (less jitter), more lag
  // beta: higher = faster response to motion (less lag at speed)
  OneEuroFilter() : min_cutoff(0.5f), beta(0.007f), d_cutoff(1.0f) {}
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

struct _GstMozzaMpGpu {
  GstVideoFilter parent;

  // ── Properties (same as mozza_mp for drop-in compatibility) ──
  gchar* model_path;
  gchar* deform_path;
  gfloat alpha;
  gfloat mls_alpha;
  gint mls_grid;
  gboolean drop;
  gboolean strict_dfm;
  gboolean ignore_ts;
  guint log_every;
  gchar* user_id;
  gint max_faces;
  gint gpu_id;
  gboolean show_landmarks;
  gboolean no_warp;
  gfloat smooth;
  gfloat min_cutoff;
  gfloat beta;
  gboolean smooth_landmarks;
  gint warp_mode;
  gint roi_pad;

  // ── Smoothing state ──
  bool has_filters;
  std::vector<OneEuroFilter> filters_x;
  std::vector<OneEuroFilter> filters_y;
  GstClockTime prev_pts;

  // ── GPU runtime ──
  std::unique_ptr<TrtFaceLandmarker> trt_lm;
  std::unique_ptr<CudaMlsWarp> cuda_warp;
  std::optional<Deformations> dfm;

  // CUDA resources
  cudaStream_t cuda_stream;
  uint8_t* d_frame_in;     // GPU staging buffer (for CPU-path upload)
  uint8_t* d_frame_out;    // GPU warp output buffer
  int alloc_w, alloc_h;
  int alloc_pitch;

  // Stats
  guint64 frame_count;

  // Per-step timing accumulators (microseconds, faces-only frames)
  double sum_h2d_us;
  double sum_detect_us;
  double sum_smooth_us;
  double sum_warp_us;
  guint64 timing_count;
};

G_END_DECLS

// ── Properties ──
enum {
  PROP_0,
  PROP_MODEL_PATH,
  PROP_MODEL_ALIAS,
  PROP_DEFORM_PATH,
  PROP_DFM_ALIAS,
  PROP_ALPHA,
  PROP_MLS_ALPHA,
  PROP_MLS_GRID,
  PROP_DROP,
  PROP_STRICT_DFM,
  PROP_IGNORE_TS,
  PROP_LOG_EVERY,
  PROP_MAX_FACES,
  PROP_GPU_ID,
  PROP_SHOW_LANDMARKS,
  PROP_NO_WARP,
  PROP_SMOOTH,
  PROP_MIN_CUTOFF,
  PROP_BETA,
  PROP_SMOOTH_LANDMARKS,
  PROP_WARP_MODE,
  PROP_ROI_PAD,
  PROP_USER_ID,
};

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS(
        "video/x-raw, format=RGBA"
        // Future NVMM support:
        // "video/x-raw(memory:NVMM), format=RGBA; "
        // "video/x-raw, format=RGBA"
    ));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS(
        "video/x-raw, format=RGBA"
    ));

G_DEFINE_TYPE(GstMozzaMpGpu, gst_mozza_mp_gpu, GST_TYPE_VIDEO_FILTER)

// ── Property accessors ──

static void gst_mozza_mp_gpu_set_property(GObject* obj, guint prop_id,
                                           const GValue* value,
                                           GParamSpec* pspec) {
  auto* self = GST_MOZZA_MP_GPU(obj);
  switch (prop_id) {
    case PROP_MODEL_PATH:
    case PROP_MODEL_ALIAS:
      g_free(self->model_path);
      self->model_path = g_value_dup_string(value);
      break;
    case PROP_DEFORM_PATH:
    case PROP_DFM_ALIAS:
      g_free(self->deform_path);
      self->deform_path = g_value_dup_string(value);
      break;
    case PROP_ALPHA:
      self->alpha = g_value_get_float(value);
      break;
    case PROP_MLS_ALPHA:
      self->mls_alpha = g_value_get_float(value);
      if (self->cuda_warp) {
        CudaMlsWarpConfig c;
        c.alpha = self->mls_alpha;
        c.gridSize = self->mls_grid;
        c.preScale = true;
        self->cuda_warp->setConfig(c);
      }
      break;
    case PROP_MLS_GRID:
      self->mls_grid = g_value_get_int(value);
      if (self->cuda_warp) {
        CudaMlsWarpConfig c;
        c.alpha = self->mls_alpha;
        c.gridSize = self->mls_grid;
        c.preScale = true;
        self->cuda_warp->setConfig(c);
      }
      break;
    case PROP_DROP:
      self->drop = g_value_get_boolean(value);
      break;
    case PROP_STRICT_DFM:
      self->strict_dfm = g_value_get_boolean(value);
      break;
    case PROP_IGNORE_TS:
      self->ignore_ts = g_value_get_boolean(value);
      break;
    case PROP_LOG_EVERY:
      self->log_every = g_value_get_uint(value);
      break;
    case PROP_MAX_FACES:
      self->max_faces = g_value_get_int(value);
      break;
    case PROP_GPU_ID:
      self->gpu_id = g_value_get_int(value);
      break;
    case PROP_SHOW_LANDMARKS:
      self->show_landmarks = g_value_get_boolean(value);
      break;
    case PROP_NO_WARP:
      self->no_warp = g_value_get_boolean(value);
      break;
    case PROP_SMOOTH:
      self->smooth = g_value_get_float(value);
      if (self->has_filters) {
        float b = 0.001f + (1.0f - self->smooth) * 0.012f;
        for (int i = 0; i < 478; ++i) {
          self->filters_x[i].beta = b;
          self->filters_y[i].beta = b;
        }
      }
      break;
    case PROP_MIN_CUTOFF:
      self->min_cutoff = g_value_get_float(value);
      if (self->has_filters) {
        for (int i = 0; i < 478; ++i) {
          self->filters_x[i].min_cutoff = self->min_cutoff;
          self->filters_y[i].min_cutoff = self->min_cutoff;
        }
      }
      break;
    case PROP_BETA:
      self->beta = g_value_get_float(value);
      if (self->has_filters) {
        for (int i = 0; i < 478; ++i) {
          self->filters_x[i].beta = self->beta;
          self->filters_y[i].beta = self->beta;
        }
      }
      break;
    case PROP_SMOOTH_LANDMARKS:
      self->smooth_landmarks = g_value_get_boolean(value);
      break;
    case PROP_WARP_MODE:
      self->warp_mode = g_value_get_int(value);
      break;
    case PROP_ROI_PAD:
      self->roi_pad = g_value_get_int(value);
      break;
    case PROP_USER_ID:
      g_free(self->user_id);
      self->user_id = g_value_dup_string(value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(obj, prop_id, pspec);
  }
}

static void gst_mozza_mp_gpu_get_property(GObject* obj, guint prop_id,
                                           GValue* value, GParamSpec* pspec) {
  auto* self = GST_MOZZA_MP_GPU(obj);
  switch (prop_id) {
    case PROP_MODEL_PATH:
    case PROP_MODEL_ALIAS:
      g_value_set_string(value, self->model_path);
      break;
    case PROP_DEFORM_PATH:
    case PROP_DFM_ALIAS:
      g_value_set_string(value, self->deform_path);
      break;
    case PROP_ALPHA:
      g_value_set_float(value, self->alpha);
      break;
    case PROP_MLS_ALPHA:
      g_value_set_float(value, self->mls_alpha);
      break;
    case PROP_MLS_GRID:
      g_value_set_int(value, self->mls_grid);
      break;
    case PROP_DROP:
      g_value_set_boolean(value, self->drop);
      break;
    case PROP_STRICT_DFM:
      g_value_set_boolean(value, self->strict_dfm);
      break;
    case PROP_IGNORE_TS:
      g_value_set_boolean(value, self->ignore_ts);
      break;
    case PROP_LOG_EVERY:
      g_value_set_uint(value, self->log_every);
      break;
    case PROP_MAX_FACES:
      g_value_set_int(value, self->max_faces);
      break;
    case PROP_GPU_ID:
      g_value_set_int(value, self->gpu_id);
      break;
    case PROP_SHOW_LANDMARKS:
      g_value_set_boolean(value, self->show_landmarks);
      break;
    case PROP_NO_WARP:
      g_value_set_boolean(value, self->no_warp);
      break;
    case PROP_SMOOTH:
      g_value_set_float(value, self->smooth);
      break;
    case PROP_MIN_CUTOFF:
      g_value_set_float(value, self->min_cutoff);
      break;
    case PROP_BETA:
      g_value_set_float(value, self->beta);
      break;
    case PROP_SMOOTH_LANDMARKS:
      g_value_set_boolean(value, self->smooth_landmarks);
      break;
    case PROP_WARP_MODE:
      g_value_set_int(value, self->warp_mode);
      break;
    case PROP_ROI_PAD:
      g_value_set_int(value, self->roi_pad);
      break;
    case PROP_USER_ID:
      g_value_set_string(value, self->user_id);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(obj, prop_id, pspec);
  }
}

// ── Helper: ensure GPU staging buffers are allocated ──
static bool ensure_gpu_buffers(GstMozzaMpGpu* self, int W, int H) {
  if (self->alloc_w >= W && self->alloc_h >= H) return true;

  // Free existing
  if (self->d_frame_in) {
    cudaFree(self->d_frame_in);
    self->d_frame_in = nullptr;
  }
  if (self->d_frame_out) {
    cudaFree(self->d_frame_out);
    self->d_frame_out = nullptr;
  }

  int pitch = W * 4;  // RGBA, tightly packed
  size_t sz = (size_t)pitch * H;

  if (cudaMalloc(&self->d_frame_in, sz) != cudaSuccess) return false;
  if (cudaMalloc(&self->d_frame_out, sz) != cudaSuccess) return false;

  self->alloc_w = W;
  self->alloc_h = H;
  self->alloc_pitch = pitch;

  GST_INFO_OBJECT(self, "GPU buffers allocated: %dx%d (%zu bytes each)", W, H,
                  sz);
  return true;
}

// ── Lifecycle ──

static gboolean gst_mozza_mp_gpu_start(GstBaseTransform* base) {
  auto* self = GST_MOZZA_MP_GPU(base);

  GST_INFO_OBJECT(self, "start() [GPU plugin]");

  // Validate model path
  if (!self->model_path ||
      !g_file_test(self->model_path, G_FILE_TEST_EXISTS)) {
    GST_ERROR_OBJECT(
        self,
        "missing/invalid model: set model=/path/to/face_landmarker.task");
    return FALSE;
  }

  // Create CUDA stream
  cudaSetDevice(self->gpu_id);
  if (cudaStreamCreate(&self->cuda_stream) != cudaSuccess) {
    GST_ERROR_OBJECT(self, "Failed to create CUDA stream");
    return FALSE;
  }

  // Create TRT face landmarker
  TrtFaceLandmarkerConfig trt_cfg;
  trt_cfg.model_path = self->model_path;
  trt_cfg.max_faces = self->max_faces;
  trt_cfg.fp16 = true;
  trt_cfg.gpu_id = self->gpu_id;

  auto t0 = std::chrono::steady_clock::now();
  self->trt_lm = TrtFaceLandmarker::Create(trt_cfg);
  auto t1 = std::chrono::steady_clock::now();
  auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

  if (!self->trt_lm) {
    GST_ERROR_OBJECT(self,
                     "TRT face landmarker creation failed in %lld ms. "
                     "Error: %s",
                     (long long)ms,
                     self->trt_lm ? self->trt_lm->last_error() : "null");
    return FALSE;
  }

  GST_INFO_OBJECT(self, "TRT face landmarker created in %lld ms", (long long)ms);

  // Create CUDA MLS warp
  CudaMlsWarpConfig warp_cfg;
  warp_cfg.gridSize = self->mls_grid;
  warp_cfg.alpha = self->mls_alpha;
  warp_cfg.preScale = true;
  self->cuda_warp = std::make_unique<CudaMlsWarp>(warp_cfg);

  // Load DFM
  if (self->deform_path) {
    self->dfm = load_dfm(self->deform_path);
    if (!self->dfm) {
      GST_WARNING_OBJECT(self, "Failed to load DFM: %s", self->deform_path);
      if (self->strict_dfm) return FALSE;
    } else {
      GST_INFO_OBJECT(self, "Loaded DFM: %s (%zu entries)", self->deform_path,
                      self->dfm->entries.size());
    }
  }

  self->frame_count = 0;
  self->sum_h2d_us = 0;
  self->sum_detect_us = 0;
  self->sum_smooth_us = 0;
  self->sum_warp_us = 0;
  self->timing_count = 0;
  return TRUE;
}

static gboolean gst_mozza_mp_gpu_stop(GstBaseTransform* base) {
  auto* self = GST_MOZZA_MP_GPU(base);

  self->trt_lm.reset();
  self->cuda_warp.reset();
  self->dfm.reset();

  if (self->d_frame_in) {
    cudaFree(self->d_frame_in);
    self->d_frame_in = nullptr;
  }
  if (self->d_frame_out) {
    cudaFree(self->d_frame_out);
    self->d_frame_out = nullptr;
  }

  if (self->cuda_stream) {
    cudaStreamDestroy(self->cuda_stream);
    self->cuda_stream = nullptr;
  }

  self->alloc_w = 0;
  self->alloc_h = 0;

  return TRUE;
}

static void gst_mozza_mp_gpu_finalize(GObject* object) {
  auto* self = GST_MOZZA_MP_GPU(object);
  self->trt_lm.reset();
  self->cuda_warp.reset();
  g_clear_pointer(&self->model_path, g_free);
  g_clear_pointer(&self->deform_path, g_free);
  g_clear_pointer(&self->user_id, g_free);
  G_OBJECT_CLASS(gst_mozza_mp_gpu_parent_class)->finalize(object);
}

static gboolean gst_mozza_mp_gpu_set_info(GstVideoFilter*, GstCaps*,
                                            GstVideoInfo*, GstCaps*,
                                            GstVideoInfo*) {
  return TRUE;
}

// ── Frame processing ──

static inline void add_identity_anchors_roi(int rx, int ry, int rw, int rh, int gridSize,
                                             std::vector<cv::Point2f>& src,
                                             std::vector<cv::Point2f>& dst) {
  const int step = std::max(4, gridSize * 2);
  for (int x = 0; x < rw; x += step) {
    float fx = (float)rx + x;
    src.push_back({fx, (float)ry}); dst.push_back(src.back());
    src.push_back({fx, (float)ry + rh - 1}); dst.push_back(src.back());
  }
  for (int y = step; y < rh - step; y += step) {
    float fy = (float)ry + y;
    src.push_back({(float)rx, fy}); dst.push_back(src.back());
    src.push_back({(float)rx + rw - 1, fy}); dst.push_back(src.back());
  }
}

static inline void add_identity_anchors(int W, int H,
                                         std::vector<cv::Point2f>& src,
                                         std::vector<cv::Point2f>& dst) {
  const float inset = 2.0f;
  const float x0 = inset, y0 = inset;
  const float x1 = (float)W - 1.0f - inset, y1 = (float)H - 1.0f - inset;
  const cv::Point2f corners[4] = { {x0, y0}, {x1, y0}, {x1, y1}, {x0, y1} };
  for (int i = 0; i < 4; ++i) { src.push_back(corners[i]); dst.push_back(corners[i]); }
}

static inline void put_px(uint8_t* p, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
  const uint8_t dr = p[0], dg = p[1], db = p[2], da = p[3];
  const uint16_t ai = a;
  p[0] = static_cast<uint8_t>((dr * (255 - ai) + r * ai) / 255);
  p[1] = static_cast<uint8_t>((dg * (255 - ai) + g * ai) / 255);
  p[2] = static_cast<uint8_t>((db * (255 - ai) + b * ai) / 255);
  p[3] = std::max<uint8_t>(da, a);
}

static void draw_dot(uint8_t* base, int W, int H, int stride, int cx, int cy,
                     int radius, uint32_t rgba) {
  if (radius < 1) radius = 1;
  const uint8_t R = (rgba >> 24) & 0xFF;
  const uint8_t G = (rgba >> 16) & 0xFF;
  const uint8_t B = (rgba >>  8) & 0xFF;
  const uint8_t A = (rgba >>  0) & 0xFF;

  const int x0 = std::max(0, cx - radius), x1 = std::min(W - 1, cx + radius);
  const int y0 = std::max(0, cy - radius), y1 = std::min(H - 1, cy + radius);

  for (int y = y0; y <= y1; ++y) {
    for (int x = x0; x <= x1; ++x) {
      uint8_t* p = base + y * stride + x * 4;
      put_px(p, R, G, B, A);
    }
  }
}

static GstFlowReturn gst_mozza_mp_gpu_transform_frame_ip(
    GstVideoFilter* vf, GstVideoFrame* f) {
  auto* self = GST_MOZZA_MP_GPU(vf);
  if (!self->trt_lm) return GST_FLOW_OK;

  const int W = GST_VIDEO_FRAME_WIDTH(f);
  const int H = GST_VIDEO_FRAME_HEIGHT(f);
  const int stride = GST_VIDEO_FRAME_PLANE_STRIDE(f, 0);
  auto* data = static_cast<uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(f, 0));

  if (!data || W <= 0 || H <= 0) return GST_FLOW_OK;
  if (!ensure_gpu_buffers(self, W, H)) return GST_FLOW_ERROR;

  self->frame_count++;

  // Timing is only active when log-every > 0 AND GST INFO is enabled for this
  // category (i.e. GST_DEBUG=mozza_mp_gpu:4 or higher).  When inactive there
  // is zero overhead: no extra cudaStreamSynchronize, no clock reads.
  const bool do_timing =
      self->log_every > 0 &&
      gst_debug_category_get_threshold(GST_CAT_DEFAULT) >= GST_LEVEL_INFO;

  // ── Step 1: H2D upload ──
  std::chrono::steady_clock::time_point t_h2d_start, t_h2d_end,
      t_detect_end, t_smooth_end, t_warp_end;

  if (do_timing) t_h2d_start = std::chrono::steady_clock::now();

  cudaMemcpy2DAsync(self->d_frame_in, self->alloc_pitch, data, stride, W * 4,
                    H, cudaMemcpyHostToDevice, self->cuda_stream);

  if (do_timing) {
    // Sync only needed to isolate H2D from detect; skipped in normal operation.
    cudaStreamSynchronize(self->cuda_stream);
    t_h2d_end = std::chrono::steady_clock::now();
  }

  // ── Step 2: TRT face detection + landmark inference ──
  GpuLandmarkResult lm_result =
      self->trt_lm->detect(self->d_frame_in, W, H, self->alloc_pitch,
                           self->cuda_stream);

  if (do_timing) t_detect_end = std::chrono::steady_clock::now();

  if (lm_result.face_count == 0) {
    // Reset filter state so the filter re-initialises cleanly when the face
    // reappears, avoiding a snap from stale x_prev/dx_prev values.
    self->has_filters = false;
    return self->drop ? GST_BASE_TRANSFORM_FLOW_DROPPED : GST_FLOW_OK;
  }

  auto& face = lm_result.faces[0];
  std::vector<cv::Point2f> L;
  L.reserve(478);

  GstClockTime pts = GST_BUFFER_PTS(f->buffer);
  float dt = 1.0f / 30.0f;
  if (!self->ignore_ts && self->prev_pts != GST_CLOCK_TIME_NONE && pts != GST_CLOCK_TIME_NONE && pts > self->prev_pts) {
    dt = (float)(pts - self->prev_pts) / (float)GST_SECOND;
  }
  self->prev_pts = pts;

  if (!self->has_filters) {
    self->filters_x.assign(478, OneEuroFilter());
    self->filters_y.assign(478, OneEuroFilter());
    for(int i=0; i<478; ++i) { 
      self->filters_x[i].min_cutoff = self->min_cutoff; 
      self->filters_y[i].min_cutoff = self->min_cutoff;
      self->filters_x[i].beta = self->beta; 
      self->filters_y[i].beta = self->beta; 
    }
    self->has_filters = true;
  }

  for (int i = 0; i < 478; ++i) {
    // 1. Get mathematically perfect normalized coords
    float nx = face.landmarks[i * 3 + 0];
    float ny = face.landmarks[i * 3 + 1];

    // 2. Filter smoothly in normalized space
    if (self->smooth_landmarks) {
      nx = self->filters_x[i].filter(nx, dt);
      ny = self->filters_y[i].filter(ny, dt);
    }

    // 3. Map directly to screen pixels
    L.emplace_back(nx * (float)W, ny * (float)H);
  }

  // Export landmarks for comparison/validation
  if (const char* lm_out = std::getenv("LANDMARK_OUTPUT_FILE")) {
    if (FILE* lmf = std::fopen(lm_out, "a")) {
      std::fprintf(lmf, "Frame %llu Face 0:\n", (unsigned long long)self->frame_count);
      for (const auto& p : L)
        std::fprintf(lmf, "%.6f,%.6f,0.000000\n", p.x / (float)W, p.y / (float)H);
      std::fclose(lmf);
    }
  }

  if (do_timing) t_smooth_end = std::chrono::steady_clock::now();

  // ── Apply Deformation (MLS Warp) ──
  if (self->dfm && self->cuda_warp && !self->no_warp) {
    std::vector<std::vector<cv::Point2f>> srcGroups, dstGroups;
    build_groups_from_dfm(*self->dfm, L, self->alpha, srcGroups, dstGroups);

    if (!srcGroups.empty()) {
      if (self->warp_mode == WARP_PER_GROUP_ROI) {
        cudaMemcpy2DAsync(self->d_frame_out, self->alloc_pitch, self->d_frame_in,
                          self->alloc_pitch, W * 4, H, cudaMemcpyDeviceToDevice,
                          self->cuda_stream);

        for (size_t g = 0; g < srcGroups.size(); ++g) {
          auto& sg = srcGroups[g];
          auto& dg = dstGroups[g];
          float minx = (float)W, miny = (float)H, maxx = 0, maxy = 0;
          for (auto& p : sg) { minx = std::min(minx, p.x); maxx = std::max(maxx, p.x); miny = std::min(miny, p.y); maxy = std::max(maxy, p.y); }
          for (auto& p : dg) { minx = std::min(minx, p.x); maxx = std::max(maxx, p.x); miny = std::min(miny, p.y); maxy = std::max(maxy, p.y); }

          int rx = std::max(0, (int)minx - self->roi_pad);
          int ry = std::max(0, (int)miny - self->roi_pad);
          int rw = std::min(W - rx, (int)(maxx - minx) + 2 * self->roi_pad);
          int rh = std::min(H - ry, (int)(maxy - miny) + 2 * self->roi_pad);
          if (rw <= 0 || rh <= 0) continue;

          std::vector<cv::Point2f> src = sg, dst = dg;
          add_identity_anchors_roi(rx, ry, rw, rh, self->mls_grid, src, dst);

          // Anchor other landmarks inside ROI to prevent "bleeding" into eyes/nose
          for (const auto& p : L) {
            if (p.x >= rx && p.x < rx + rw && p.y >= ry && p.y < ry + rh) {
              // Check if this point is already being moved
              bool moving = false;
              for (const auto& sp : sg) {
                if (std::abs(p.x - sp.x) < 0.1f && std::abs(p.y - sp.y) < 0.1f) {
                  moving = true; break;
                }
              }
              if (!moving) {
                src.push_back(p); dst.push_back(p);
              }
            }
          }

          int nPts = (int)src.size();
          std::vector<float> h_src_xy(nPts * 2), h_dst_xy(nPts * 2);
          for (int i = 0; i < nPts; ++i) {
            h_src_xy[i * 2 + 0] = src[i].x; h_src_xy[i * 2 + 1] = src[i].y;
            h_dst_xy[i * 2 + 0] = dst[i].x; h_dst_xy[i * 2 + 1] = dst[i].y;
          }
          self->cuda_warp->warp(self->d_frame_in, self->d_frame_out, W, H, self->alloc_pitch, self->alloc_pitch, h_src_xy.data(), h_dst_xy.data(), nPts, rx, ry, rw, rh, self->cuda_stream);
        }
      } else {
        std::vector<cv::Point2f> src, dst;
        for (size_t g = 0; g < srcGroups.size(); ++g) {
          src.insert(src.end(), srcGroups[g].begin(), srcGroups[g].end());
          dst.insert(dst.end(), dstGroups[g].begin(), dstGroups[g].end());
        }
        add_identity_anchors(W, H, src, dst);

        // Anchor all other landmarks to prevent global bleeding
        for (const auto& p : L) {
          bool moving = false;
          for (size_t g = 0; g < srcGroups.size(); ++g) {
            for (const auto& sp : srcGroups[g]) {
              if (std::abs(p.x - sp.x) < 0.1f && std::abs(p.y - sp.y) < 0.1f) {
                moving = true; break;
              }
            }
            if (moving) break;
          }
          if (!moving) {
            src.push_back(p); dst.push_back(p);
          }
        }

        int nPts = (int)src.size();
        std::vector<float> h_src_xy(nPts * 2), h_dst_xy(nPts * 2);
        for (int i = 0; i < nPts; ++i) {
          h_src_xy[i * 2 + 0] = src[i].x; h_src_xy[i * 2 + 1] = src[i].y;
          h_dst_xy[i * 2 + 0] = dst[i].x; h_dst_xy[i * 2 + 1] = dst[i].y;
        }
        self->cuda_warp->warp(self->d_frame_in, self->d_frame_out, W, H, self->alloc_pitch, self->alloc_pitch, h_src_xy.data(), h_dst_xy.data(), nPts, 0, 0, W, H, self->cuda_stream);
      }

      cudaMemcpy2DAsync(data, stride, self->d_frame_out, self->alloc_pitch,
                        W * 4, H, cudaMemcpyDeviceToHost, self->cuda_stream);
      cudaStreamSynchronize(self->cuda_stream);
    }
  }
  if (do_timing) {
    t_warp_end = std::chrono::steady_clock::now();

    auto us_diff = [](std::chrono::steady_clock::time_point a,
                      std::chrono::steady_clock::time_point b) -> double {
      return (double)std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    };
    self->sum_h2d_us    += us_diff(t_h2d_start, t_h2d_end);
    self->sum_detect_us += us_diff(t_h2d_end,   t_detect_end);
    self->sum_smooth_us += us_diff(t_detect_end, t_smooth_end);
    self->sum_warp_us   += us_diff(t_smooth_end, t_warp_end);
    self->timing_count++;

    if (self->timing_count % self->log_every == 0) {
      double n = (double)self->log_every;
      double h2d_ms    = self->sum_h2d_us    / n / 1000.0;
      double detect_ms = self->sum_detect_us / n / 1000.0;
      double smooth_ms = self->sum_smooth_us / n / 1000.0;
      double warp_ms   = self->sum_warp_us   / n / 1000.0;
      double total_ms  = h2d_ms + detect_ms + smooth_ms + warp_ms;
      GST_INFO_OBJECT(self,
          "TIMING frame=%llu (window avg)  H2D=%.2fms  TRT-detect=%.2fms  "
          "smooth=%.2fms  warp+D2H=%.2fms  total=%.2fms  (%.0f fps)",
          (unsigned long long)self->timing_count,
          h2d_ms, detect_ms, smooth_ms, warp_ms, total_ms,
          total_ms > 0.0 ? 1000.0 / total_ms : 0.0);
      // Reset window accumulators
      self->sum_h2d_us = 0; self->sum_detect_us = 0;
      self->sum_smooth_us = 0; self->sum_warp_us = 0;
    }
  }

  // ── Draw landmarks ──
  if (self->show_landmarks) {
    for (int i = 0; i < 478; i++) {
      int cx = (int)L[i].x, cy = (int)L[i].y;
      int R = 5;
      for (int dy = -R; dy <= R; dy++) {
        for (int dx = -R; dx <= R; dx++) {
          int x = cx + dx, y = cy + dy;
          if (x >= 0 && x < W && y >= 0 && y < H) {
            uint8_t* pix = data + y * stride + x * 4;
            pix[0] = 0; pix[1] = 255; pix[2] = 0; pix[3] = 255; // PURE GREEN
          }
        }
      }
    }
  }

  return GST_FLOW_OK;
}

// ── Class init ──

static void gst_mozza_mp_gpu_class_init(GstMozzaMpGpuClass* klass) {
  GST_DEBUG_CATEGORY_INIT(gst_mozza_mp_gpu_debug_category, "mozza_mp_gpu", 0,
                          "Mozza MP GPU (TensorRT + CUDA)");

  auto* gobject_class = G_OBJECT_CLASS(klass);
  auto* basetr_class = GST_BASE_TRANSFORM_CLASS(klass);
  auto* vfilter_class = GST_VIDEO_FILTER_CLASS(klass);

  gobject_class->set_property = gst_mozza_mp_gpu_set_property;
  gobject_class->get_property = gst_mozza_mp_gpu_get_property;
  gobject_class->finalize = gst_mozza_mp_gpu_finalize;

  g_object_class_install_property(
      gobject_class, PROP_MODEL_PATH,
      g_param_spec_string("model_path", "Model path",
                          "Path to face_landmarker.task (ONNX models must be "
                          "in same directory)",
                          nullptr, G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_MODEL_ALIAS,
      g_param_spec_string("model", "Model path [alias]",
                          "Alias for 'model_path'", nullptr, G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_DEFORM_PATH,
      g_param_spec_string("deform", "Deformation file (.dfm)",
                          "Path to deformation file", nullptr,
                          G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_DFM_ALIAS,
      g_param_spec_string("dfm", "Deformation file (.dfm) [alias]",
                          "Alias for 'deform'", nullptr, G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_ALPHA,
      g_param_spec_float("alpha", "Deformation intensity",
                         "Scales deformation intensity", -10.f, 10.f, 1.f,
                         G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_MLS_ALPHA,
      g_param_spec_float("mls-alpha", "MLS alpha", "Rigidity parameter", 0.f,
                         10.f, 1.4f, G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_MLS_GRID,
      g_param_spec_int("mls-grid", "MLS grid size", "Grid size in pixels", 1,
                       100, 5, G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_DROP,
      g_param_spec_boolean("drop", "Drop on no face",
                           "Drop frames when no face detected", FALSE,
                           G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_STRICT_DFM,
      g_param_spec_boolean("strict-dfm", "Fail when DFM fails",
                           "If true and deform path fails, start() fails",
                           FALSE, G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_IGNORE_TS,
      g_param_spec_boolean("ignore-timestamps", "Ignore timestamps",
                           "Use synthetic timestamps", FALSE,
                           G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_LOG_EVERY,
      g_param_spec_uint("log-every", "Log interval",
                        "Log stats every N frames", 0, 1000000, 60,
                        G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_MAX_FACES,
      g_param_spec_int("max-faces", "Max faces", "Maximum faces to detect", 1,
                       16, 1, G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_GPU_ID,
      g_param_spec_int("gpu-id", "GPU device ID", "CUDA device index", 0, 8, 0,
                       G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_SHOW_LANDMARKS,
      g_param_spec_boolean("show-landmarks", "Show landmarks",
                           "Draw landmarks on frame", FALSE,
                           G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_NO_WARP,
      g_param_spec_boolean("no-warp", "No warp",
                           "Disable warping (landmark debug only)", FALSE,
                           G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_SMOOTH,
      g_param_spec_float("smooth", "Smoothing",
                         "Temporal smoothing factor (0=off, 0.9=max). Tunes beta of the OneEuroFilter.", 0.0f,
                         0.99f, 0.5f, G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_MIN_CUTOFF,
      g_param_spec_float("min-cutoff", "Min cutoff frequency",
                         "OneEuroFilter min_cutoff: lower = more smoothing at rest (less jitter, more lag). Default 1.5 Hz.",
                         0.001f, 100.0f, 1.5f, G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_BETA,
      g_param_spec_float("beta", "Beta (cutoff slope)",
                         "OneEuroFilter beta: higher = less lag at high speeds. Default 0.05.",
                         0.0f, 1.0f, 0.05f, G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_SMOOTH_LANDMARKS,
      g_param_spec_boolean("smooth-landmarks", "Smooth Landmarks",
                           "Apply OneEuroFilter to landmarks", TRUE,
                           G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_WARP_MODE,
      g_param_spec_int("warp-mode", "Warp mode", "0=global, 1=per-group-roi", 0, 1, 0, G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_ROI_PAD,
      g_param_spec_int("roi-pad", "ROI padding", "Padding for per-group-roi mode", 0, 128, 24, G_PARAM_READWRITE));

  g_object_class_install_property(

      gobject_class, PROP_USER_ID,
      g_param_spec_string("user-id", "User ID", "Opaque user identifier",
                          nullptr, G_PARAM_READWRITE));

  gst_element_class_set_static_metadata(
      GST_ELEMENT_CLASS(klass), "Mozza MP GPU", "Filter/Effect/Video",
      "GPU-accelerated DFM-driven MLS facial deformation (TensorRT + CUDA)",
      "DuckSoup Lab");

  gst_element_class_add_pad_template(
      GST_ELEMENT_CLASS(klass),
      gst_static_pad_template_get(&sink_template));
  gst_element_class_add_pad_template(
      GST_ELEMENT_CLASS(klass), gst_static_pad_template_get(&src_template));

  basetr_class->start = gst_mozza_mp_gpu_start;
  basetr_class->stop = gst_mozza_mp_gpu_stop;
  vfilter_class->set_info = gst_mozza_mp_gpu_set_info;
  vfilter_class->transform_frame_ip = gst_mozza_mp_gpu_transform_frame_ip;
}

static void gst_mozza_mp_gpu_init(GstMozzaMpGpu* self) {
  self->model_path = nullptr;
  self->deform_path = nullptr;
  self->alpha = 1.0f;
  self->mls_alpha = 1.4f;
  self->mls_grid = 5;
  self->drop = FALSE;
  self->strict_dfm = FALSE;
  self->ignore_ts = FALSE;
  self->log_every = 60;
  self->user_id = nullptr;
  self->max_faces = 1;
  self->gpu_id = 0;
  self->show_landmarks = FALSE;
  self->no_warp = FALSE;
  self->smooth = 0.5f;
  self->min_cutoff = 1.5f;
  self->beta = 0.05f;
  self->smooth_landmarks = TRUE;
  self->warp_mode = WARP_GLOBAL;
  self->roi_pad = 24;

  self->has_filters = false;
  self->prev_pts = GST_CLOCK_TIME_NONE;

  self->cuda_stream = nullptr;
  self->d_frame_in = nullptr;
  self->d_frame_out = nullptr;
  self->alloc_w = 0;
  self->alloc_h = 0;
  self->alloc_pitch = 0;
  self->frame_count = 0;
  self->sum_h2d_us = 0;
  self->sum_detect_us = 0;
  self->sum_smooth_us = 0;
  self->sum_warp_us = 0;
  self->timing_count = 0;
}

static gboolean plugin_init(GstPlugin* plugin) {
  return gst_element_register(plugin, "mozza_mp_gpu", GST_RANK_NONE,
                              GST_TYPE_MOZZA_MP_GPU);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, mozzamp_gpu,
                  "GPU-accelerated facial deformation via TensorRT + CUDA",
                  plugin_init, "1.0", "LGPL", "mozza_mp_gpu",
                  "https://ducksouplab.com")
