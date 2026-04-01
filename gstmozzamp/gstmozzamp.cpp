// MediaPipe-style smile/frown transformer using your flat C runtime + MLS.
// RGBA in-place.
//
// Element: mozza_mp
// Props:
//   model              : path to face_landmarker.task (string, required)
//   deform             : path to deformation .dfm (string, optional)
//   dfm                : alias of "deform" (string, optional; for legacy pipelines)
//   alpha              : float, [-10..10], default 1.0
//   mls-alpha          : float, default 1.4 (MLS rigidity parameter)
//   mls-grid           : int, default 5 (MLS grid size in pixels; smaller=denser)
//   warp-mode          : string, default "global" ("global" or "per-group-roi")
//   roi-pad            : int, default 24 (padding around per-group ROI in pixels)
//   overlay            : bool, default false (draw src/dst control points + vectors)
//   drop               : bool, default false (drop frame when no face)
//   show-landmarks     : bool, default false (draw all landmarks even without DFM)
//   strict-dfm         : bool, default false (fail start if deform given but load fails)
//   force-rgb          : bool, default false (no-op; pads require RGBA; kept for parity)
//   ignore-timestamps  : bool, default false (pass 0us into detector)
//   log-every          : uint, default 60 (periodic log interval; 0 disables)
//   user-id            : string, accepted but ignored (for Ducksoup uniform configs)
//
// Caps: video/x-raw, format=RGBA

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

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "mp_runtime.h"
#include "mp_runtime_loader.h"

#include "dfm.hpp"
#include "deform_utils.hpp"
#include "imgwarp/imgwarp_mls_rigid.h"
#include <limits>   // for std::numeric_limits

#ifndef PACKAGE
#define PACKAGE "mozza_mp"
#endif

GST_DEBUG_CATEGORY_STATIC(gst_mozza_mp_debug_category);
#define GST_CAT_DEFAULT gst_mozza_mp_debug_category

G_BEGIN_DECLS

#define GST_TYPE_MOZZA_MP (gst_mozza_mp_get_type())
G_DECLARE_FINAL_TYPE(GstMozzaMp, gst_mozza_mp, GST, MOZZA_MP, GstVideoFilter)

struct _GstMozzaMp {
  GstVideoFilter parent;

  // properties
  gchar* model_path;
  gchar* deform_path;     // also set by "dfm"
  gfloat   alpha;
  gfloat   mls_alpha;
  gint     mls_grid;
  gboolean overlay;
  gboolean drop;
  gboolean show_landmarks;
  gboolean no_warp;
  gboolean strict_dfm;
  gboolean force_rgb;       // no-op (pads are RGBA)
  gboolean ignore_ts;
  guint    log_every;
  gchar* user_id;         // accepted but not used
  gint     num_threads;
  gint     max_faces;
  gint     lm_radius;
  guint    lm_color;

  gint     warp_mode;       // WarpMode enum
  gint     roi_pad;         // padding for per-group ROI warps

  // runtime + helpers
  MpFaceCtx* mp_ctx;
  std::optional<Deformations> dfm;
  std::unique_ptr<mp_imgwarp::ImgWarp_MLS_Rigid> mls;

  // stats
  guint64 frame_count;
};

G_END_DECLS

enum WarpMode {
  WARP_GLOBAL = 0,
  WARP_PER_GROUP_ROI = 1,
};

// ── Properties ────────────────────────────────────────────────────────────────
enum {
  PROP_0,
  PROP_MODEL_PATH,
  PROP_DEFORM_PATH,
  PROP_ALPHA,
  PROP_OVERLAY,
  PROP_DROP,
  PROP_USER_ID,
  PROP_DFM_ALIAS,        // alias for "deform"
  PROP_SHOW_LANDMARKS,
  PROP_NO_WARP,
  PROP_STRICT_DFM,
  PROP_FORCE_RGB,        // NEW
  PROP_IGNORE_TS,        // NEW
  PROP_LOG_EVERY,        // NEW
  PROP_NUM_THREADS,
  PROP_MAX_FACES,
  PROP_LM_RADIUS,
  PROP_LM_COLOR,
  PROP_MLS_ALPHA,
  PROP_MLS_GRID,
  PROP_WARP_MODE,
  PROP_ROI_PAD,
};

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
  "sink", GST_PAD_SINK, GST_PAD_ALWAYS,
  GST_STATIC_CAPS("video/x-raw(memory:GLMemory), format=RGBA; video/x-raw, format=RGBA"));
static GstStaticPadTemplate src_template  = GST_STATIC_PAD_TEMPLATE(
  "src",  GST_PAD_SRC,  GST_PAD_ALWAYS,
  GST_STATIC_CAPS("video/x-raw(memory:GLMemory), format=RGBA; video/x-raw, format=RGBA"));

G_DEFINE_TYPE(GstMozzaMp, gst_mozza_mp, GST_TYPE_VIDEO_FILTER)

// ── Utils ────────────────────────────────────────────────────────────────────
static inline void put_px(uint8_t* p, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
  const uint8_t dr = p[0], dg = p[1], db = p[2], da = p[3];
  const uint16_t ai = a;
  p[0] = static_cast<uint8_t>((dr * (255 - ai) + r * ai) / 255);
  p[1] = static_cast<uint8_t>((dg * (255 - ai) + g * ai) / 255);
  p[2] = static_cast<uint8_t>((db * (255 - ai) + b * ai) / 255);
  p[3] = std::max<uint8_t>(da, a);
}
static inline int clampi(int v, int lo, int hi) { return (v < lo) ? lo : (v > hi ? hi : v); }

static void draw_dot(uint8_t* base, int W, int H, int stride, int cx, int cy,
                     int radius, uint32_t rgba) {
  if (radius < 1) radius = 1;
  const uint8_t R = (rgba >> 24) & 0xFF;
  const uint8_t G = (rgba >> 16) & 0xFF;
  const uint8_t B = (rgba >>  8) & 0xFF;
  const uint8_t A = (rgba >>  0) & 0xFF;

  const int x0 = std::max(0, cx - radius), x1 = std::min(W - 1, cx + radius);
  const int y0 = std::max(0, cy - radius), y1 = std::min(H - 1, cy + radius);
  const int r2 = radius * radius;

  for (int y = y0; y <= y1; ++y) {
    const int dy = y - cy;
    for (int x = x0; x <= x1; ++x) {
      const int dx = x - cx;
      if (dx*dx + dy*dy <= r2) {
        uint8_t* p = base + y * stride + x * 4;
        put_px(p, R, G, B, A);
      }
    }
  }
}

static void draw_line(uint8_t* base, int W, int H, int stride,
                      int x0, int y0, int x1, int y1, uint32_t rgba)
{
  const uint8_t R = (rgba >> 24) & 0xFF;
  const uint8_t G = (rgba >> 16) & 0xFF;
  const uint8_t B = (rgba >>  8) & 0xFF;
  const uint8_t A = (rgba >>  0) & 0xFF;

  int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
  int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
  int err = dx + dy, e2;

  for (;;) {
    if ((unsigned)x0 < (unsigned)W && (unsigned)y0 < (unsigned)H) {
      uint8_t* p = base + y0 * stride + x0 * 4;
      put_px(p, R, G, B, A);
    }
    if (x0 == x1 && y0 == y1) break;
    e2 = 2 * err;
    if (e2 >= dy) { err += dy; x0 += sx; }
    if (e2 <= dx) { err += dx; y0 += sy; }
  }
}

static void log_landmark_stats(GstMozzaMp* self, const MpFace& f0, int W, int H) {
  if (f0.landmarks_count <= 0) return;
  float minx=1e9f, miny=1e9f, maxx=-1e9f, maxy=-1e9f;
  for (int i=0;i<f0.landmarks_count;++i) {
    minx = std::min(minx, f0.landmarks[i].x);
    miny = std::min(miny, f0.landmarks[i].y);
    maxx = std::max(maxx, f0.landmarks[i].x);
    maxy = std::max(maxy, f0.landmarks[i].y);
  }
  GST_INFO_OBJECT(self, "landmarks: count=%d norm[min(%.3f,%.3f) max(%.3f,%.3f)] px[min(%d,%d) max(%d,%d)]",
                  f0.landmarks_count, minx, miny, maxx, maxy,
                  (int)std::floor(minx*W), (int)std::floor(miny*H),
                  (int)std::ceil (maxx*W), (int)std::ceil (maxy*H));
}

// ── GObject props ────────────────────────────────────────────────────────────
static void gst_mozza_mp_set_property(GObject* obj, guint prop_id,
                                      const GValue* value, GParamSpec* pspec) {
  auto* self = GST_MOZZA_MP(obj);
  switch (prop_id) {
    case PROP_MODEL_PATH:
      g_free(self->model_path);
      self->model_path = g_value_dup_string(value);
      GST_INFO_OBJECT(self, "prop:model = %s", self->model_path ? self->model_path : "(null)");
      break;
    case PROP_DEFORM_PATH:
    case PROP_DFM_ALIAS:  // alias: "dfm"
      g_free(self->deform_path);
      self->deform_path = g_value_dup_string(value);
      GST_INFO_OBJECT(self, "prop:deform/dfm = %s", self->deform_path ? self->deform_path : "(null)");
      break;
    case PROP_ALPHA:
      self->alpha = g_value_get_float(value);
      GST_INFO_OBJECT(self, "prop:alpha = %.3f", self->alpha);
      break;
    case PROP_MLS_ALPHA:
      self->mls_alpha = g_value_get_float(value);
      if (self->mls) self->mls->alpha = self->mls_alpha;
      GST_INFO_OBJECT(self, "prop:mls-alpha = %.3f", self->mls_alpha);
      break;
    case PROP_MLS_GRID:
      self->mls_grid = g_value_get_int(value);
      if (self->mls) self->mls->gridSize = self->mls_grid;
      GST_INFO_OBJECT(self, "prop:mls-grid = %d", self->mls_grid);
      break;
    case PROP_WARP_MODE: {
      const char* s = g_value_get_string(value);
      if (s && g_ascii_strcasecmp(s, "per-group-roi") == 0)
        self->warp_mode = WARP_PER_GROUP_ROI;
      else
        self->warp_mode = WARP_GLOBAL;
      GST_INFO_OBJECT(self, "prop:warp-mode = %s",
                      self->warp_mode == WARP_PER_GROUP_ROI ? "per-group-roi" : "global");
      break;
    }
    case PROP_ROI_PAD:
      self->roi_pad = g_value_get_int(value);
      GST_INFO_OBJECT(self, "prop:roi-pad = %d", self->roi_pad);
      break;
    case PROP_OVERLAY:
      self->overlay = g_value_get_boolean(value);
      GST_INFO_OBJECT(self, "prop:overlay = %s", self->overlay ? "true" : "false");
      break;
    case PROP_DROP:
      self->drop = g_value_get_boolean(value);
      GST_INFO_OBJECT(self, "prop:drop = %s", self->drop ? "true" : "false");
      break;
    case PROP_SHOW_LANDMARKS:
      self->show_landmarks = g_value_get_boolean(value);
      GST_INFO_OBJECT(self, "prop:show-landmarks = %s", self->show_landmarks ? "true" : "false");
      break;
    case PROP_NO_WARP:
      self->no_warp = g_value_get_boolean(value);
      GST_INFO_OBJECT(self, "prop:no-warp = %s", self->no_warp ? "true" : "false");
      break;
    case PROP_STRICT_DFM:
      self->strict_dfm = g_value_get_boolean(value);
      GST_INFO_OBJECT(self, "prop:strict-dfm = %s", self->strict_dfm ? "true" : "false");
      break;
    case PROP_FORCE_RGB:
      self->force_rgb = g_value_get_boolean(value);
      GST_WARNING_OBJECT(self, "prop:force-rgb = %s (no-op)", self->force_rgb ? "true" : "false");
      break;
    case PROP_IGNORE_TS:
      self->ignore_ts = g_value_get_boolean(value);
      GST_INFO_OBJECT(self, "prop:ignore-timestamps = %s", self->ignore_ts ? "true" : "false");
      break;
    case PROP_LOG_EVERY:
      self->log_every = g_value_get_uint(value);
      GST_INFO_OBJECT(self, "prop:log-every = %u", self->log_every);
      break;
    case PROP_NUM_THREADS:
      self->num_threads = g_value_get_int(value);
      GST_INFO_OBJECT(self, "prop:threads = %d", self->num_threads);
      break;
    case PROP_MAX_FACES:
      self->max_faces = g_value_get_int(value);
      GST_INFO_OBJECT(self, "prop:max-faces = %d", self->max_faces);
      break;
    case PROP_LM_RADIUS:
      self->lm_radius = g_value_get_int(value);
      GST_INFO_OBJECT(self, "prop:landmark-radius = %d", self->lm_radius);
      break;
    case PROP_LM_COLOR:
      self->lm_color = g_value_get_uint(value);
      GST_INFO_OBJECT(self, "prop:landmark-color = 0x%08X", self->lm_color);
      break;
    case PROP_USER_ID:
      g_free(self->user_id);
      self->user_id = g_value_dup_string(value);
      break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(obj, prop_id, pspec);
  }
}

static void gst_mozza_mp_get_property(GObject* obj, guint prop_id,
                                      GValue* value, GParamSpec* pspec) {
  auto* self = GST_MOZZA_MP(obj);
  switch (prop_id) {
    case PROP_MODEL_PATH:      g_value_set_string (value, self->model_path);  break;
    case PROP_DEFORM_PATH:     g_value_set_string (value, self->deform_path); break;
    case PROP_DFM_ALIAS:       g_value_set_string (value, self->deform_path); break; // alias
    case PROP_ALPHA:           g_value_set_float  (value, self->alpha);       break;
    case PROP_MLS_ALPHA:       g_value_set_float  (value, self->mls_alpha);   break;
    case PROP_MLS_GRID:        g_value_set_int    (value, self->mls_grid);    break;
    case PROP_WARP_MODE:
      g_value_set_string(value, self->warp_mode == WARP_PER_GROUP_ROI ? "per-group-roi" : "global");
      break;
    case PROP_ROI_PAD:         g_value_set_int    (value, self->roi_pad);    break;
    case PROP_OVERLAY:         g_value_set_boolean(value, self->overlay);     break;
    case PROP_DROP:            g_value_set_boolean(value, self->drop);        break;
    case PROP_STRICT_DFM:      g_value_set_boolean(value, self->strict_dfm); break;
    case PROP_SHOW_LANDMARKS:  g_value_set_boolean(value, self->show_landmarks); break;
    case PROP_NO_WARP:         g_value_set_boolean(value, self->no_warp);        break;
    case PROP_FORCE_RGB:       g_value_set_boolean(value, self->force_rgb);      break;
    case PROP_IGNORE_TS:       g_value_set_boolean(value, self->ignore_ts);      break;
    case PROP_LOG_EVERY:       g_value_set_uint   (value, self->log_every);      break;
    case PROP_NUM_THREADS:     g_value_set_int    (value, self->num_threads);     break;
    case PROP_MAX_FACES:       g_value_set_int    (value, self->max_faces);       break;
    case PROP_LM_RADIUS:       g_value_set_int    (value, self->lm_radius);       break;
    case PROP_LM_COLOR:        g_value_set_uint   (value, self->lm_color);        break;
    case PROP_USER_ID:         g_value_set_string (value, self->user_id);     break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(obj, prop_id, pspec);
  }
}

// ── Lifecycle ────────────────────────────────────────────────────────────────
static gboolean gst_mozza_mp_start(GstBaseTransform* base) {
  auto* self = GST_MOZZA_MP(base);

  GST_INFO_OBJECT(self, "start()");

  // --- DIAGNOSTICS: Check environment variables for Threading ---
  const char* omp = std::getenv("OMP_NUM_THREADS");
  const char* xnn = std::getenv("XNNPACK_NUM_THREADS");
  const char* tflite = std::getenv("TFLITE_NUM_THREADS");
  GST_INFO_OBJECT(self, "--- THREADING DIAGNOSTICS ---");
  GST_INFO_OBJECT(self, "Requested via Gst property: threads=%d", self->num_threads);
  GST_INFO_OBJECT(self, "Env OMP_NUM_THREADS:     %s", omp ? omp : "UNSET (Defaults to 1)");
  GST_INFO_OBJECT(self, "Env XNNPACK_NUM_THREADS: %s", xnn ? xnn : "UNSET (Defaults to 1)");
  GST_INFO_OBJECT(self, "Env TFLITE_NUM_THREADS:  %s", tflite ? tflite : "UNSET (Defaults to 1)");
  GST_INFO_OBJECT(self, "-----------------------------");

  if (!self->model_path || !g_file_test(self->model_path, G_FILE_TEST_EXISTS)) {
    GST_ERROR_OBJECT(self, "missing/invalid model: set model=/path/to/face_landmarker.task");
    return FALSE;
  }
  if (!MpApiOK()) {
    GST_ERROR_OBJECT(self, "mp_runtime loader not initialized");
    return FALSE;
  }

  MpFaceLandmarkerOptions opts{};
  opts.model_path       = self->model_path;
  opts.max_faces        = self->max_faces;
  opts.with_blendshapes = 0;
  opts.with_geometry    = 0;
  opts.num_threads      = self->num_threads;
  opts.delegate         = "cpu";

  self->mp_ctx = nullptr;
  auto t0 = std::chrono::steady_clock::now();
  int rc = MpApi().face_create(&opts, &self->mp_ctx);
  auto t1 = std::chrono::steady_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

  if (rc != 0 || !self->mp_ctx) {
    const char* loader_err = mp_runtime_loader::last_error();
    GST_FIXME_OBJECT(self, "mp_face_landmarker_create failed (rc=%d) in %lld ms. Error: %s", rc, (long long)ms, loader_err ? loader_err : "(none)");
    return FALSE;
  }
  
  self->mls = std::make_unique<mp_imgwarp::ImgWarp_MLS_Rigid>();
  self->mls->gridSize = self->mls_grid;
  self->mls->preScale = true;
  self->mls->alpha    = self->mls_alpha;

  if (self->deform_path) {
    errno = 0;
    self->dfm = load_dfm(self->deform_path);
    if (!self->dfm) {
      if (self->strict_dfm) return FALSE;
    }
  }

  self->frame_count = 0;
  return TRUE;
}

static gboolean gst_mozza_mp_stop(GstBaseTransform* base) {
  auto* self = GST_MOZZA_MP(base);
  if (self->mp_ctx) { MpApi().face_close(&self->mp_ctx); self->mp_ctx = nullptr; }
  self->mls.reset();
  self->dfm.reset();
  return TRUE;
}

static void gst_mozza_mp_finalize(GObject* object) {
  auto* self = GST_MOZZA_MP(object);
  if (self->mp_ctx) { MpApi().face_close(&self->mp_ctx); self->mp_ctx = nullptr; }
  g_clear_pointer(&self->model_path,  g_free);
  g_clear_pointer(&self->deform_path, g_free);
  g_clear_pointer(&self->user_id,     g_free);
  G_OBJECT_CLASS(gst_mozza_mp_parent_class)->finalize(object);
}

static gboolean gst_mozza_mp_set_info(GstVideoFilter*, GstCaps*, GstVideoInfo*, GstCaps*, GstVideoInfo*) { return TRUE; }

static inline void add_identity_anchors(const cv::Rect& roi, std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& dst, int inset = 2) {
  if (roi.width <= 0 || roi.height <= 0) return;
  const float x0 = (float)(roi.x + inset);
  const float y0 = (float)(roi.y + inset);
  const float x1 = (float)(roi.x + roi.width  - 1 - inset);
  const float y1 = (float)(roi.y + roi.height - 1 - inset);
  const cv::Point2f corners[4] = { {x0, y0}, {x1, y0}, {x1, y1}, {x0, y1} };
  for (int i = 0; i < 4; ++i) { src.push_back(corners[i]); dst.push_back(corners[i]); }
}

static inline uint64_t fnv1a64(const uint8_t* p, size_t n, uint64_t seed = 14695981039346656037ULL) {
  uint64_t h = seed; const uint64_t prime = 1099511628211ULL;
  for (size_t i = 0; i < n; ++i) { h ^= static_cast<uint64_t>(p[i]); h *= prime; }
  return h;
}

static inline uint64_t hash_frame_rgba(const uint8_t* base, int W, int H, int stride) {
  uint64_t h = 0ULL;
  for (int y = 0; y < H; ++y) h = fnv1a64(base + static_cast<size_t>(y) * static_cast<size_t>(stride), static_cast<size_t>(W) * 4u, h);
  return h;
}

static double mean_abs_rgb_diff(const cv::Mat& a, const cv::Mat& b) {
  cv::Mat diff; cv::absdiff(a, b, diff); cv::Scalar m = cv::mean(diff);
  double acc = 0.0; for (int c = 0; c < diff.channels(); ++c) acc += m[c];
  return acc / std::max(1, diff.channels());
}

static bool env_flag(const char* name) { const char* s = std::getenv(name); return s && *s && s[0] != '0'; }

static void overlay_landmarks(uint8_t* data, int W, int H, int stride,
                              const std::vector<cv::Point2f>& L,
                              int radius, uint32_t rgba) {
  const uint8_t R = (rgba >> 24) & 0xFF;
  const uint8_t G = (rgba >> 16) & 0xFF;
  const uint8_t B = (rgba >>  8) & 0xFF;
  const uint8_t A = (rgba >>  0) & 0xFF;

  for (const auto& p : L) {
    int cx = (int)p.x;
    int cy = (int)p.y;
    const int x0 = std::max(0, cx - radius), x1 = std::min(W - 1, cx + radius);
    const int y0 = std::max(0, cy - radius), y1 = std::min(H - 1, cy + radius);
    const int r2 = radius * radius;

    for (int y = y0; y <= y1; ++y) {
      uint8_t* row = data + y * stride;
      for (int x = x0; x <= x1; ++x) {
        if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r2) {
          uint8_t* pix = row + x * 4;
          const uint16_t ai = A;
          pix[0] = static_cast<uint8_t>((pix[0] * (255 - ai) + R * ai) / 255);
          pix[1] = static_cast<uint8_t>((pix[1] * (255 - ai) + G * ai) / 255);
          pix[2] = static_cast<uint8_t>((pix[2] * (255 - ai) + B * ai) / 255);
          pix[3] = std::max<uint8_t>(pix[3], A);
        }
      }
    }
  }
}

static GstFlowReturn gst_mozza_mp_transform_frame_ip(GstVideoFilter* vf,
                                                    GstVideoFrame* f) {

  auto* self = GST_MOZZA_MP(vf);
  if (!self->mp_ctx) return GST_FLOW_OK;

  self->frame_count++;

  const int W      = GST_VIDEO_FRAME_WIDTH(f);
  const int H      = GST_VIDEO_FRAME_HEIGHT(f);
  const int stride = GST_VIDEO_FRAME_PLANE_STRIDE(f, 0);
  auto* data       = static_cast<uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(f, 0));

  cv::Mat img_rgba(H, W, CV_8UC4, data, static_cast<size_t>(stride));
  if (img_rgba.empty()) return GST_FLOW_OK;

  MpImage img{};
  img.data   = static_cast<const uint8_t*>(data);
  img.width  = W;
  img.height = H;
  img.stride = stride;
  img.format = MP_IMAGE_RGBA8888;

  GstClockTime pts = GST_BUFFER_PTS(f->buffer);
  int64_t ts_us;
  if (self->ignore_ts) {
      ts_us = (int64_t)self->frame_count * 33333LL;
  } else {
      ts_us = (int64_t)GST_TIME_AS_USECONDS(pts);
  }

  if (self->frame_count <= 5) {
    GST_INFO_OBJECT(self, "FRAME %llu TRACKER DIAGNOSTIC | pts: %" GST_TIME_FORMAT " | synthetic ts_us: %lld | ts_ms sent to MP: %lld",
                    (unsigned long long)self->frame_count, GST_TIME_ARGS(pts), (long long)ts_us, (long long)(ts_us / 1000));
  }

  auto t0 = std::chrono::steady_clock::now();
  MpFaceResult out;
  int rc = MpApi().face_detect(self->mp_ctx, &img, ts_us, &out);
  auto t1 = std::chrono::steady_clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

  auto should_log = [&](guint64 frame) -> bool {
    if (self->log_every == 0) return false;
    if (self->log_every == 1) return true;
    return (frame % self->log_every) == 1;
  };

  if (rc != 0) { MpApi().face_free_result(&out); return GST_FLOW_OK; }

  if (out.faces_count == 0) {
    MpApi().face_free_result(&out);
    if (self->drop) return GST_BASE_TRANSFORM_FLOW_DROPPED;
    return GST_FLOW_OK;
  }

  const MpFace& f0 = out.faces[0];
  if (should_log(self->frame_count)) {
    GST_INFO_OBJECT(self, "faces=%d landmarks=%d (detect %.3f ms, ts_us=%lld)", out.faces_count, f0.landmarks_count, us / 1000.0, (long long)ts_us);
    log_landmark_stats(self, f0, W, H);
  }

  std::vector<cv::Point2f> L; L.reserve(f0.landmarks_count);
  for (int i = 0; i < f0.landmarks_count; ++i) L.emplace_back(f0.landmarks[i].x * W, f0.landmarks[i].y * H);

  // Export landmarks for comparison/validation
  if (const char* lm_out = std::getenv("LANDMARK_OUTPUT_FILE")) {
    if (FILE* lmf = std::fopen(lm_out, "a")) {
      GST_LOG_OBJECT(self, "Dumping landmarks to %s", lm_out);
      std::fprintf(lmf, "Frame %llu Face 0:\n", (unsigned long long)self->frame_count);
      for (const auto& p : L)
        std::fprintf(lmf, "%.6f,%.6f,0.000000\n", p.x / (float)W, p.y / (float)H);
      std::fclose(lmf);
    }
  }

  // 4) Warp
  if (self->dfm && !self->no_warp) {
    if (self->mls) {
      std::vector<std::vector<cv::Point2f>> srcGroups, dstGroups;
      build_groups_from_dfm(*self->dfm, L, self->alpha, srcGroups, dstGroups);
      if (!srcGroups.empty()) {
        if (self->warp_mode == WARP_PER_GROUP_ROI) {
          for (size_t g = 0; g < srcGroups.size(); ++g) compute_MLS_on_ROI(img_rgba, *self->mls, srcGroups[g], dstGroups[g], self->roi_pad);
        } else {
          std::vector<cv::Point2f> src, dst;
          for (size_t g = 0; g < srcGroups.size(); ++g) { src.insert(src.end(), srcGroups[g].begin(), srcGroups[g].end()); dst.insert(dst.end(), dstGroups[g].begin(), dstGroups[g].end()); }
          add_identity_anchors(cv::Rect(0, 0, W, H), src, dst, 2);
          cv::Mat warped = self->mls->setAllAndGenerate(img_rgba, src, dst, img_rgba.cols, img_rgba.rows);
          if (!warped.empty()) warped.copyTo(img_rgba);
        }
      }
    }
  }

  // Final draw pass: force bright green on raw buffer
  if (self->show_landmarks) {
    // DEBUG: Red square top-left
    for (int y=0; y<50; y++) {
      for (int x=0; x<50; x++) {
        uint8_t* p = data + y*stride + x*4;
        p[0]=255; p[1]=0; p[2]=0; p[3]=255;
      }
    }
    for (const auto& p : L) {
      int cx = (int)p.x, cy = (int)p.y;
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

  MpApi().face_free_result(&out);
  return GST_FLOW_OK;
}

static void gst_mozza_mp_class_init(GstMozzaMpClass* klass) {
  GST_DEBUG_CATEGORY_INIT(gst_mozza_mp_debug_category, "mozza_mp", 0, "Mozza MP (runtime loader)");
  auto* gobject_class = G_OBJECT_CLASS(klass);
  auto* basetr_class  = GST_BASE_TRANSFORM_CLASS(klass);
  auto* vfilter_class = GST_VIDEO_FILTER_CLASS(klass);

  gobject_class->set_property = gst_mozza_mp_set_property;
  gobject_class->get_property = gst_mozza_mp_get_property;
  gobject_class->finalize     = gst_mozza_mp_finalize;

  g_object_class_install_property(gobject_class, PROP_MODEL_PATH, g_param_spec_string("model", "Model path", "Path to face_landmarker.task", nullptr, G_PARAM_READWRITE));
  g_object_class_install_property(gobject_class, PROP_DEFORM_PATH, g_param_spec_string("deform", "Deformation file (.dfm)", "Path to deformation file with barycentric rules", nullptr, G_PARAM_READWRITE));
  g_object_class_install_property(gobject_class, PROP_DFM_ALIAS, g_param_spec_string("dfm", "Deformation file (.dfm) [alias]", "Alias for 'deform'", nullptr, G_PARAM_READWRITE));
  g_object_class_install_property(gobject_class, PROP_ALPHA, g_param_spec_float("alpha", "Smile intensity multiplicator", "Scales the intensity of the deformation", -10.f, 10.f, 1.f, G_PARAM_READWRITE));
  g_object_class_install_property(gobject_class, PROP_OVERLAY, g_param_spec_boolean("overlay", "Debug overlay", "Draw src/dst control points and vectors", FALSE, G_PARAM_READWRITE));
  g_object_class_install_property(gobject_class, PROP_DROP, g_param_spec_boolean("drop", "Drop on no face", "Drop frames when no face is detected", FALSE, G_PARAM_READWRITE));
  g_object_class_install_property(gobject_class, PROP_SHOW_LANDMARKS, g_param_spec_boolean("show-landmarks", "Draw landmarks", "Draw all detected landmarks", FALSE, G_PARAM_READWRITE));
  g_object_class_install_property(gobject_class, PROP_NO_WARP, g_param_spec_boolean("no-warp", "No warp", "Disable warping", FALSE, G_PARAM_READWRITE));
  g_object_class_install_property(gobject_class, PROP_STRICT_DFM, g_param_spec_boolean("strict-dfm", "Fail when DFM fails to load", "If true and deform path fails, start() fails", FALSE, G_PARAM_READWRITE));
  g_object_class_install_property(gobject_class, PROP_FORCE_RGB, g_param_spec_boolean("force-rgb", "Accept property for parity", "No-op", FALSE, G_PARAM_READWRITE));
  g_object_class_install_property(gobject_class, PROP_IGNORE_TS, g_param_spec_boolean("ignore-timestamps", "Force ts=0", "When true, pass 0us as timestamp into the detector", FALSE, G_PARAM_READWRITE));
  g_object_class_install_property(gobject_class, PROP_LOG_EVERY, g_param_spec_uint("log-every", "Periodic log interval", "Log every N frames", 0, 1000000, 60, G_PARAM_READWRITE));
  g_object_class_install_property(gobject_class, PROP_NUM_THREADS, g_param_spec_int("threads", "Number of threads", "CPU threads (0=default)", 0, 32, 4, G_PARAM_READWRITE));
  g_object_class_install_property(gobject_class, PROP_MAX_FACES, g_param_spec_int("max-faces", "Max faces", "Maximum number of faces", 1, 16, 1, G_PARAM_READWRITE));
  g_object_class_install_property(gobject_class, PROP_LM_RADIUS, g_param_spec_int("landmark-radius", "Landmark dot radius", "Radius", 1, 10, 2, G_PARAM_READWRITE));
  g_object_class_install_property(gobject_class, PROP_LM_COLOR, g_param_spec_uint("landmark-color", "Landmark dot color", "Packed RGBA color", 0, G_MAXUINT, 0x0066CCFFu, G_PARAM_READWRITE));
  g_object_class_install_property(gobject_class, PROP_MLS_ALPHA, g_param_spec_float("mls-alpha", "MLS alpha", "Rigidity parameter", 0.f, 10.f, 1.4f, G_PARAM_READWRITE));
  g_object_class_install_property(gobject_class, PROP_MLS_GRID, g_param_spec_int("mls-grid", "MLS grid size", "Grid size in pixels", 1, 100, 5, G_PARAM_READWRITE));
  g_object_class_install_property(gobject_class, PROP_WARP_MODE, g_param_spec_string("warp-mode", "Warp mode", "global or per-group-roi", "global", G_PARAM_READWRITE));
  g_object_class_install_property(gobject_class, PROP_ROI_PAD, g_param_spec_int("roi-pad", "ROI padding", "Padding around ROI", 0, 200, 24, G_PARAM_READWRITE));
  g_object_class_install_property(gobject_class, PROP_USER_ID, g_param_spec_string("user-id", "User ID", "Opaque user identifier", nullptr, G_PARAM_READWRITE));

  gst_element_class_set_static_metadata(GST_ELEMENT_CLASS(klass), "Mozza MP", "Filter/Effect/Video", "DFM-driven MLS", "DuckSoup Lab");
  gst_element_class_add_pad_template(GST_ELEMENT_CLASS(klass), gst_static_pad_template_get(&sink_template));
  gst_element_class_add_pad_template(GST_ELEMENT_CLASS(klass), gst_static_pad_template_get(&src_template));

  basetr_class->start = gst_mozza_mp_start;
  basetr_class->stop  = gst_mozza_mp_stop;
  vfilter_class->set_info           = gst_mozza_mp_set_info;
  vfilter_class->transform_frame_ip = gst_mozza_mp_transform_frame_ip;
}

static void gst_mozza_mp_init(GstMozzaMp* self) {
  self->model_path     = nullptr;
  self->deform_path    = nullptr;
  self->alpha          = 1.0f;
  self->mls_alpha      = 1.4f;
  self->mls_grid       = 5;
  self->overlay        = FALSE;
  self->drop           = FALSE;
  self->show_landmarks = FALSE;
  self->no_warp        = FALSE;
  self->strict_dfm     = FALSE;
  self->force_rgb      = FALSE;
  self->ignore_ts      = FALSE;
  self->log_every      = 60;
  self->user_id        = nullptr;
  self->num_threads     = 4;
  self->max_faces      = 1;
  self->lm_radius      = 3;
  self->lm_color       = 0x00FF00FFu; // green
  self->warp_mode      = WARP_GLOBAL;
  self->roi_pad        = 24;
  self->frame_count    = 0;
  self->mp_ctx         = nullptr;
}

static gboolean plugin_init(GstPlugin* plugin) { return gst_element_register(plugin, "mozza_mp", GST_RANK_NONE, GST_TYPE_MOZZA_MP); }
GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, mozzamp, "Facial deformation via mp_runtime", plugin_init, "1.02", "LGPL", "mozza_mp", "https://ducksouplab.com")