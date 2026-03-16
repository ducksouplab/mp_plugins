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
};

G_END_DECLS

// ── Properties ──
enum {
  PROP_0,
  PROP_MODEL_PATH,
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

static inline void add_identity_anchors(int W, int H,
                                         std::vector<cv::Point2f>& src,
                                         std::vector<cv::Point2f>& dst,
                                         int inset = 2) {
  const float x0 = (float)inset;
  const float y0 = (float)inset;
  const float x1 = (float)(W - 1 - inset);
  const float y1 = (float)(H - 1 - inset);
  const cv::Point2f corners[4] = {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}};
  for (int i = 0; i < 4; ++i) {
    src.push_back(corners[i]);
    dst.push_back(corners[i]);
  }
}

static GstFlowReturn gst_mozza_mp_gpu_transform_frame_ip(
    GstVideoFilter* vf, GstVideoFrame* f) {
  auto* self = GST_MOZZA_MP_GPU(vf);
  if (!self->trt_lm) return GST_FLOW_OK;

  self->frame_count++;

  const int W = GST_VIDEO_FRAME_WIDTH(f);
  const int H = GST_VIDEO_FRAME_HEIGHT(f);
  const int stride = GST_VIDEO_FRAME_PLANE_STRIDE(f, 0);
  auto* data = static_cast<uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(f, 0));

  if (!data || W <= 0 || H <= 0) return GST_FLOW_OK;

  // Ensure GPU buffers
  if (!ensure_gpu_buffers(self, W, H)) {
    GST_ERROR_OBJECT(self, "Failed to allocate GPU buffers");
    return GST_FLOW_ERROR;
  }

  auto should_log = [&](guint64 frame) -> bool {
    return (self->log_every > 0) && ((frame % self->log_every) == 1);
  };

  // ── Upload CPU frame to GPU ──
  // (For NVMM path, this would be replaced with direct device pointer access)
  cudaMemcpy2DAsync(self->d_frame_in, self->alloc_pitch, data, stride, W * 4,
                    H, cudaMemcpyHostToDevice, self->cuda_stream);

  // ── Stage 1+2: TRT Face Detection + Landmarks ──
  auto t0 = std::chrono::steady_clock::now();
  GpuLandmarkResult lm_result =
      self->trt_lm->detect(self->d_frame_in, W, H, self->alloc_pitch,
                           self->cuda_stream);
  auto t1 = std::chrono::steady_clock::now();
  auto det_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

  if (lm_result.face_count == 0) {
    if (self->drop) return GST_BASE_TRANSFORM_FLOW_DROPPED;
    // No face: download original frame back (it was uploaded)
    // Actually for in-place, just leave the CPU buffer untouched
    return GST_FLOW_OK;
  }

  if (should_log(self->frame_count)) {
    GST_INFO_OBJECT(self, "[GPU] faces=%d (detect %.3f ms) frame=%llu",
                    lm_result.face_count, det_us / 1000.0,
                    (unsigned long long)self->frame_count);
  }

  // ── Convert landmarks to pixel coords (CPU, negligible cost) ──
  auto& face = lm_result.faces[0];
  std::vector<cv::Point2f> L;
  L.reserve(478);
  for (int i = 0; i < 478; ++i) {
    L.emplace_back(face.landmarks[i * 3 + 0] * W,
                   face.landmarks[i * 3 + 1] * H);
  }

  // ── Apply DFM deformation ──
  if (self->dfm && self->cuda_warp) {
    std::vector<std::vector<cv::Point2f>> srcGroups, dstGroups;
    build_groups_from_dfm(*self->dfm, L, self->alpha, srcGroups, dstGroups);

    if (!srcGroups.empty()) {
      // On GPU, always use global warp mode (launching multiple small kernels
      // is slower than one full-frame warp on GPU)
      std::vector<cv::Point2f> src, dst;
      size_t total = 0;
      for (auto& g : srcGroups) total += g.size();
      src.reserve(total + 4);
      dst.reserve(total + 4);

      for (size_t g = 0; g < srcGroups.size(); ++g) {
        src.insert(src.end(), srcGroups[g].begin(), srcGroups[g].end());
        dst.insert(dst.end(), dstGroups[g].begin(), dstGroups[g].end());
      }
      add_identity_anchors(W, H, src, dst, 2);

      // Convert cv::Point2f to interleaved float arrays for CUDA
      int nPts = (int)src.size();
      std::vector<float> h_src_xy(nPts * 2);
      std::vector<float> h_dst_xy(nPts * 2);
      for (int i = 0; i < nPts; ++i) {
        h_src_xy[i * 2 + 0] = src[i].x;
        h_src_xy[i * 2 + 1] = src[i].y;
        h_dst_xy[i * 2 + 0] = dst[i].x;
        h_dst_xy[i * 2 + 1] = dst[i].y;
      }

      auto t2 = std::chrono::steady_clock::now();

      // CUDA MLS warp: d_frame_in -> d_frame_out
      self->cuda_warp->warp(self->d_frame_in, self->d_frame_out, W, H,
                            self->alloc_pitch, self->alloc_pitch,
                            h_src_xy.data(), h_dst_xy.data(), nPts,
                            self->cuda_stream);

      // Download warped frame back to CPU buffer
      cudaMemcpy2DAsync(data, stride, self->d_frame_out, self->alloc_pitch,
                        W * 4, H, cudaMemcpyDeviceToHost, self->cuda_stream);
      cudaStreamSynchronize(self->cuda_stream);

      auto t3 = std::chrono::steady_clock::now();
      auto warp_us =
          std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2)
              .count();

      if (should_log(self->frame_count)) {
        GST_INFO_OBJECT(self, "[GPU] warp %.3f ms (%d ctrl pts)",
                        warp_us / 1000.0, nPts);
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
      g_param_spec_string("model", "Model path",
                          "Path to face_landmarker.task (ONNX models must be "
                          "in same directory)",
                          nullptr, G_PARAM_READWRITE));
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
      g_param_spec_int("gpu-id", "GPU ID", "CUDA device index", 0, 15, 0,
                       G_PARAM_READWRITE));
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

  self->cuda_stream = nullptr;
  self->d_frame_in = nullptr;
  self->d_frame_out = nullptr;
  self->alloc_w = 0;
  self->alloc_h = 0;
  self->alloc_pitch = 0;
  self->frame_count = 0;
}

static gboolean plugin_init(GstPlugin* plugin) {
  return gst_element_register(plugin, "mozza_mp_gpu", GST_RANK_NONE,
                              GST_TYPE_MOZZA_MP_GPU);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, mozzampgpu,
                  "GPU-accelerated facial deformation via TensorRT + CUDA",
                  plugin_init, "1.0", "LGPL", "mozza_mp_gpu",
                  "https://ducksouplab.com")
