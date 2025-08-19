// MediaPipe-style smile/frown transformer using your flat C runtime + MLS.
// RGBA in-place.
//
// Element: mozza_mp
// Props:
//   model              : path to face_landmarker.task (string, required)
//   deform             : path to deformation .dfm (string, optional)
//   dfm                : alias of "deform" (string, optional; for legacy pipelines)
//   alpha              : float, [-10..10], default 1.0
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
  gchar*   model_path;
  gchar*   deform_path;     // also set by "dfm"
  gfloat   alpha;
  gboolean overlay;
  gboolean drop;
  gboolean show_landmarks;
  gboolean strict_dfm;
  gboolean force_rgb;       // no-op (pads are RGBA)
  gboolean ignore_ts;
  guint    log_every;
  gchar*   user_id;         // accepted but not used

  // runtime + helpers
  MpFaceCtx* mp_ctx;
  std::optional<Deformations> dfm;
  std::unique_ptr<ImgWarp_MLS_Rigid> mls;
  cv::Mat bgr_tmp;

  // debug flags (cached from environment)
  gboolean dbg_invert_if_zero;
  gboolean dbg_affine;
  gboolean dbg_hash;

  // stats
  guint64 frame_count;
};

G_END_DECLS

static bool env_flag(const char* name);

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
  PROP_STRICT_DFM,
  PROP_FORCE_RGB,        // NEW
  PROP_IGNORE_TS,        // NEW
  PROP_LOG_EVERY,        // NEW
};

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
  "sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw, format=RGBA"));
static GstStaticPadTemplate src_template  = GST_STATIC_PAD_TEMPLATE(
  "src",  GST_PAD_SRC,  GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw, format=RGBA"));

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
    case PROP_STRICT_DFM:
      self->strict_dfm = g_value_get_boolean(value);
      GST_INFO_OBJECT(self, "prop:strict-dfm = %s", self->strict_dfm ? "true" : "false");
      break;
    case PROP_FORCE_RGB:
      self->force_rgb = g_value_get_boolean(value);
      GST_WARNING_OBJECT(self, "prop:force-rgb = %s (no-op: pads are RGBA; upstream must convert)",
                         self->force_rgb ? "true" : "false");
      break;
    case PROP_IGNORE_TS:
      self->ignore_ts = g_value_get_boolean(value);
      GST_INFO_OBJECT(self, "prop:ignore-timestamps = %s", self->ignore_ts ? "true" : "false");
      break;
    case PROP_LOG_EVERY:
      // FIX: avoid std::max<int, guint> deduction clash; property already clamps to >= 0
      self->log_every = g_value_get_uint(value);
      GST_INFO_OBJECT(self, "prop:log-every = %u", self->log_every);
      break;
    case PROP_USER_ID:
      g_free(self->user_id);
      self->user_id = g_value_dup_string(value);
      GST_LOG_OBJECT(self, "prop:user-id = '%s' (ignored)", self->user_id ? self->user_id : "(null)");
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
    case PROP_OVERLAY:         g_value_set_boolean(value, self->overlay);     break;
    case PROP_DROP:            g_value_set_boolean(value, self->drop);        break;
    case PROP_SHOW_LANDMARKS:  g_value_set_boolean(value, self->show_landmarks); break;
    case PROP_STRICT_DFM:      g_value_set_boolean(value, self->strict_dfm);     break;
    case PROP_FORCE_RGB:       g_value_set_boolean(value, self->force_rgb);      break;
    case PROP_IGNORE_TS:       g_value_set_boolean(value, self->ignore_ts);      break;
    case PROP_LOG_EVERY:       g_value_set_uint   (value, self->log_every);      break;
    case PROP_USER_ID:         g_value_set_string (value, self->user_id);     break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(obj, prop_id, pspec);
  }
}

// ── Lifecycle ────────────────────────────────────────────────────────────────
static gboolean gst_mozza_mp_start(GstBaseTransform* base) {
  auto* self = GST_MOZZA_MP(base);

  GST_INFO_OBJECT(self, "start()");
  if (!self->model_path || !g_file_test(self->model_path, G_FILE_TEST_EXISTS)) {
    GST_ERROR_OBJECT(self, "missing/invalid model: set model=/path/to/face_landmarker.task (got: %s)",
                     self->model_path ? self->model_path : "(null)");
    return FALSE;
  }
  if (!MpApiOK()) {
    GST_ERROR_OBJECT(self, "mp_runtime loader not initialized");
    return FALSE;
  }

  // Cache MOZZA_DEBUG_* environment flags once
  self->dbg_invert_if_zero = env_flag("MOZZA_DEBUG_INVERT_IF_ZERO");
  self->dbg_affine         = env_flag("MOZZA_DEBUG_AFFINE");
#if !defined(NDEBUG)
  self->dbg_hash           = env_flag("MOZZA_DEBUG_HASH");
#else
  self->dbg_hash           = FALSE;
#endif

  MpFaceLandmarkerOptions opts{};
  opts.model_path       = self->model_path;
  opts.max_faces        = 1;
  opts.with_blendshapes = 0;
  opts.with_geometry    = 0;
  opts.num_threads      = 0;
  opts.delegate         = "xnnpack";

  self->mp_ctx = nullptr;
  auto t0 = std::chrono::steady_clock::now();
  int rc = MpApi().face_create(&opts, &self->mp_ctx);
  auto t1 = std::chrono::steady_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

  if (rc != 0 || !self->mp_ctx) {
    GST_ERROR_OBJECT(self, "mp_face_landmarker_create failed (rc=%d) in %lld ms", rc, (long long)ms);
    return FALSE;
  }
  GST_INFO_OBJECT(self, "mp_face_landmarker created in %lld ms (delegate=%s, threads=%d)",
                  (long long)ms, opts.delegate, opts.num_threads);

  self->mls = std::make_unique<ImgWarp_MLS_Rigid>();
  self->mls->gridSize = 5;
  self->mls->preScale = true;
  self->mls->alpha    = 1.4;

  if (self->deform_path) {
    errno = 0;
    self->dfm = load_dfm(self->deform_path);
    if (!self->dfm) {
      GST_ERROR_OBJECT(self, "DFM load FAILED: '%s'%s%s",
                       self->deform_path,
                       errno ? " (" : "",
                       errno ? g_strerror(errno) : "");
      if (self->strict_dfm) {
        GST_ERROR_OBJECT(self, "strict-dfm=true -> failing start()");
        return FALSE;
      } else {
        GST_WARNING_OBJECT(self, "continuing without deformation (strict-dfm=false)");
      }
    } else {
      GST_INFO_OBJECT(self, "DFM load OK: '%s' (using alpha=%.3f)", self->deform_path, self->alpha);
    }
  } else {
    GST_INFO_OBJECT(self, "No DFM path provided; will pass-through unless overlay/show-landmarks enabled");
  }

  if (self->force_rgb) {
    GST_WARNING_OBJECT(self, "force-rgb requested, but pads are RGBA. This is a no-op. "
                              "Keep a capsfilter/videoconvert upstream to ensure RGBA.");
  }

  self->frame_count = 0;
  return TRUE;
}

static gboolean gst_mozza_mp_stop(GstBaseTransform* base) {
  auto* self = GST_MOZZA_MP(base);
  GST_INFO_OBJECT(self, "stop()");
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

static gboolean gst_mozza_mp_set_info(GstVideoFilter*, GstCaps*, GstVideoInfo*,
                                      GstCaps*, GstVideoInfo*) { return TRUE; }



// Add 4 identity “pin” anchors (same src==dst) near rectangle corners, slightly
// inset so they are strictly inside the region. This helps keep the boundary
// stable during MLS warping.
static inline void add_identity_anchors(
    const cv::Rect& roi,
    std::vector<cv::Point2f>& src,
    std::vector<cv::Point2f>& dst,
    int inset = 2) {

  if (roi.width <= 0 || roi.height <= 0) return;

  const float x0 = (float)(roi.x + inset);
  const float y0 = (float)(roi.y + inset);
  const float x1 = (float)(roi.x + roi.width  - 1 - inset);
  const float y1 = (float)(roi.y + roi.height - 1 - inset);

  const cv::Point2f corners[4] = {
    {x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}
  };

  for (int i = 0; i < 4; ++i) {
    src.push_back(corners[i]);
    dst.push_back(corners[i]);  // identity anchor
  }
}


// ── Per-frame work ───────────────────────────────────────────────────────────

#if !defined(NDEBUG)
// Streamed/seeded FNV-1a (64-bit) — lets us continue hashing across rows.
// FNV-1a 64-bit, supports both 2-arg and 3-arg use (seed is optional)
static inline uint64_t fnv1a64(const uint8_t* p, size_t n, uint64_t seed = 14695981039346656037ULL) {
  uint64_t h = seed;                       // FNV offset basis by default
  const uint64_t prime = 1099511628211ULL; // FNV prime
  for (size_t i = 0; i < n; ++i) {
    h ^= static_cast<uint64_t>(p[i]);
    h *= prime;
  }
  return h;
}
#endif


static double mean_abs_rgb_diff(const cv::Mat& a, const cv::Mat& b) {
  cv::Mat diff; cv::absdiff(a, b, diff);
  cv::Scalar m = cv::mean(diff);
  // Average only over the channels present
  double acc = 0.0;
  for (int c = 0; c < diff.channels(); ++c) acc += m[c];
  return acc / std::max(1, diff.channels());
}

static bool env_flag(const char* name) {
  const char* s = std::getenv(name);
  return s && *s && s[0] != '0';
}



static inline uint32_t read_rgba_px(const uint8_t* data, int stride, int x, int y) {
  const uint8_t* p = data + (size_t)y * stride + (size_t)x * 4;
  // Pack as 0xAABBGGRR to make channel order obvious in logs.
  return (uint32_t)p[3] << 24 | (uint32_t)p[2] << 16 | (uint32_t)p[1] << 8 | (uint32_t)p[0];
}

// Cheap sample-based checksum over 4 image corners (BGR image)
static inline uint32_t sample_checksum_bgr(const cv::Mat& img) {
  if (img.empty() || img.channels() < 3) return 0;
  const int W = img.cols, H = img.rows; const int stride = img.step[0];
  const uint8_t* base = img.data;
  const uint8_t* p0 = base;
  const uint8_t* p1 = base + (size_t)(W - 1) * 3;
  const uint8_t* p2 = base + (size_t)(H - 1) * stride;
  const uint8_t* p3 = p2 + (size_t)(W - 1) * 3;
  auto pack = [](const uint8_t* p) -> uint32_t {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
  };
  uint32_t c = 0;
  c ^= pack(p0);
  c ^= pack(p1);
  c ^= pack(p2);
  c ^= pack(p3);
  return c;
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

  if (self->frame_count == 1) {
    GST_INFO_OBJECT(self,
      "first frame: %dx%d stride=%d overlay=%d show-landmarks=%d drop=%d dfm=%s (mls grid=%d alpha=%.2f)",
      W, H, stride, (int)self->overlay, (int)self->show_landmarks, (int)self->drop,
      self->dfm ? "yes" : "no",
      self->mls ? self->mls->gridSize : -1,
      self->mls ? self->mls->alpha    : -1.0);
  }

  // Wrap original RGBA buffer
  cv::Mat img_rgba(H, W, CV_8UC4, data, static_cast<size_t>(stride));
  if (img_rgba.empty()) return GST_FLOW_OK;

  // Build MpImage for detection
  MpImage img{};
  img.data   = static_cast<const uint8_t*>(data);
  img.width  = W;
  img.height = H;
  img.stride = stride;
  img.format = MP_IMAGE_RGBA8888;

  const GstClockTime pts = GST_BUFFER_PTS(f->buffer);
  const int64_t ts_us = self->ignore_ts ? 0 :
                        ((pts == GST_CLOCK_TIME_NONE) ? 0 : (int64_t)GST_TIME_AS_USECONDS(pts));

  MpFaceResult out{};
  const auto t0 = std::chrono::steady_clock::now();
  const int rc = MpApi().face_detect(self->mp_ctx, &img, ts_us, &out);
  const auto t1 = std::chrono::steady_clock::now();
  const auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

  auto should_log = [&](guint64 frame) -> bool {
    return (self->log_every > 0) && ((frame % self->log_every) == 1);
  };

  if (rc != 0) {
    GST_WARNING_OBJECT(self, "face_detect rc=%d (%.3f ms, ts_us=%lld) -> pass-through",
                       rc, us / 1000.0, (long long)ts_us);
    MpApi().face_free_result(&out);
    return GST_FLOW_OK;
  }

  if (out.faces_count == 0) {
    if (should_log(self->frame_count)) {
      GST_INFO_OBJECT(self, "no faces (detect %.3f ms, ts_us=%lld)",
                      us / 1000.0, (long long)ts_us);
    }
    MpApi().face_free_result(&out);
    if (self->drop) return GST_BASE_TRANSFORM_FLOW_DROPPED;
    return GST_FLOW_OK;
  }

  const MpFace& f0 = out.faces[0];
  if (should_log(self->frame_count)) {
    GST_INFO_OBJECT(self, "faces=%d landmarks=%d (detect %.3f ms, ts_us=%lld)",
                    out.faces_count, f0.landmarks_count, us / 1000.0, (long long)ts_us);
    log_landmark_stats(self, f0, W, H);
  }

  // Landmarks → pixel coords
  std::vector<cv::Point2f> L; L.reserve(f0.landmarks_count);
  for (int i = 0; i < f0.landmarks_count; ++i) {
    const MpLandmark& lm = f0.landmarks[i];
    L.emplace_back(lm.x * W, lm.y * H);
  }

  // Optional: draw all landmarks (blue), even without DFM
  if (self->show_landmarks) {
    const uint32_t blue = 0x0066CCFFu;
    for (const auto& p : L) {
      const int x = clampi((int)std::lround(p.x), 0, W - 1);
      const int y = clampi((int)std::lround(p.y), 0, H - 1);
      draw_dot(data, W, H, stride, x, y, 2, blue);
    }
  }

  // Apply deformation via DFM + MLS
  if (self->dfm) {
    if (!self->mls) {
      GST_WARNING_OBJECT(self, "DFM provided but MLS object is null — skipping warp");
      MpApi().face_free_result(&out);
      return GST_FLOW_OK;
    }

    std::vector<std::vector<cv::Point2f>> srcGroups, dstGroups;
    build_groups_from_dfm(*self->dfm, L, self->alpha, srcGroups, dstGroups);

    if (srcGroups.empty()) {
      GST_WARNING_OBJECT(self, "DFM produced 0 groups — topology mismatch? (detector count=%zu)", L.size());
    } else {
      const bool log_now = should_log(self->frame_count);

      // Displacement stats (from DFM)
      size_t ctrl_pts = 0;
      float  max_disp = 0.0f, mean_disp = 0.0f;
      size_t samples  = 0;
      for (size_t g = 0; g < srcGroups.size(); ++g) {
        ctrl_pts += srcGroups[g].size();
        if (log_now) {
          for (size_t i = 0; i < srcGroups[g].size(); ++i) {
            cv::Point2f d = dstGroups[g][i] - srcGroups[g][i];
            float v = std::hypot(d.x, d.y);
            max_disp = std::max(max_disp, v);
            mean_disp += v; ++samples;
          }
        }
      }
      if (log_now && samples) mean_disp /= samples;
      if (log_now) {
        GST_INFO_OBJECT(self, "DFM groups=%zu ctrl=%zu alpha=%.3f mean|Δ|=%.2f max|Δ|=%.2f",
                        srcGroups.size(), ctrl_pts, self->alpha, mean_disp, max_disp);
      }

      const auto t0w = std::chrono::steady_clock::now();

      // ImgWarp_MLS_Rigid only handles 3-channel images; cache a reusable
      // BGR buffer to avoid reallocating every frame.
      cv::Mat& img_bgr = self->bgr_tmp;
      img_bgr.create(H, W, CV_8UC3);
      cv::cvtColor(img_rgba, img_bgr, cv::COLOR_RGBA2BGR);

      // Combine all control points into a single warp to avoid repeatedly
      // warping the entire image for each group.
      std::vector<cv::Point2f> src;
      std::vector<cv::Point2f> dst;
      src.reserve(ctrl_pts + 4);
      dst.reserve(ctrl_pts + 4);
      for (size_t g = 0; g < srcGroups.size(); ++g) {
        src.insert(src.end(), srcGroups[g].begin(), srcGroups[g].end());
        dst.insert(dst.end(), dstGroups[g].begin(), dstGroups[g].end());
      }
      add_identity_anchors(cv::Rect(0, 0, W, H), src, dst, /*inset=*/2);

#if !defined(NDEBUG)
      uint64_t hash_before = 0;
      if (log_now && self->dbg_hash) {
        hash_before = fnv1a64(img_bgr.data,
                              (size_t)img_bgr.step[0] * img_bgr.rows);
      }
#endif
      uint32_t chk_before = 0;
      if (log_now && (!self->dbg_hash)) {
        chk_before = sample_checksum_bgr(img_bgr);
      }

      cv::Mat warped;
      if (self->dbg_affine) {
        cv::Mat M = (cv::Mat_<double>(2,3) << 1, 0, 10, 0, 1, 6);
        cv::warpAffine(img_bgr, warped, M, img_bgr.size(),
                       cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
      } else {
        warped = self->mls->setAllAndGenerate(
            img_bgr, src, dst, img_bgr.cols, img_bgr.rows);
      }

      double mean_delta = 0.0;
      const bool need_mean_delta = self->dbg_invert_if_zero || log_now;
      if (need_mean_delta && !warped.empty()) {
        mean_delta = mean_abs_rgb_diff(img_bgr, warped);
      }
      if (!warped.empty()) {
        warped.copyTo(img_bgr);
      }

#if !defined(NDEBUG)
      uint64_t hash_after = 0;
      if (log_now && self->dbg_hash) {
        hash_after = fnv1a64(img_bgr.data,
                             (size_t)img_bgr.step[0] * img_bgr.rows);
      }
#endif
      uint32_t chk_after = 0;
      if (log_now && (!self->dbg_hash)) {
        chk_after = sample_checksum_bgr(img_bgr);
      }

      if (self->dbg_invert_if_zero && mean_delta < 0.5) {
        cv::bitwise_not(img_bgr, img_bgr);
      }

      if (log_now) {
#if !defined(NDEBUG)
        if (self->dbg_hash) {
          GST_INFO_OBJECT(self,
            "MLS combined: groups=%zu ctrl=%zu (+4 anchors) | hash %016llx -> %016llx | meanΔ=%.2f%s",
            srcGroups.size(), (size_t)src.size(),
            (unsigned long long)hash_before, (unsigned long long)hash_after, mean_delta,
            (hash_before == hash_after ? "  (no byte change!)" : ""));
        } else
#endif
        {
          GST_INFO_OBJECT(self,
            "MLS combined: groups=%zu ctrl=%zu (+4 anchors) | chk %08x -> %08x | meanΔ=%.2f%s",
            srcGroups.size(), (size_t)src.size(),
            chk_before, chk_after, mean_delta,
            (chk_before == chk_after ? "  (no sample change!)" : ""));
        }
      }

      cv::cvtColor(img_bgr, img_rgba, cv::COLOR_BGR2RGBA);

      const auto t1w = std::chrono::steady_clock::now();
      if (log_now) {
        const auto msw = std::chrono::duration_cast<std::chrono::milliseconds>(t1w - t0w).count();
        GST_INFO_OBJECT(self, "MLS warp %lld ms", (long long)msw);
      }

      // Draw displacement vectors (top)
      if (self->overlay) {
        const uint32_t green = 0x00FF00FFu, red = 0xFF0000FFu;
        for (size_t g = 0; g < srcGroups.size(); ++g) {
          for (size_t i = 0; i < srcGroups[g].size(); ++i) {
            const auto& s = srcGroups[g][i];
            const auto& d = dstGroups[g][i];
            const int sx = clampi((int)std::lround(s.x), 0, W - 1);
            const int sy = clampi((int)std::lround(s.y), 0, H - 1);
            const int dx = clampi((int)std::lround(d.x), 0, W - 1);
            const int dy = clampi((int)std::lround(d.y), 0, H - 1);
            draw_dot (data, W, H, stride, sx, sy, 2, red);
            draw_dot (data, W, H, stride, dx, dy, 2, green);
            draw_line(data, W, H, stride, sx, sy, dx, dy, green);
          }
        }
      }
    }
  }

  MpApi().face_free_result(&out);
  return GST_FLOW_OK;
}



// ── Class / init / plugin boilerplate ────────────────────────────────────────
static void gst_mozza_mp_class_init(GstMozzaMpClass* klass) {
  GST_DEBUG_CATEGORY_INIT(gst_mozza_mp_debug_category, "mozza_mp", 0,
                          "Mozza MP (runtime loader)");

  auto* gobject_class = G_OBJECT_CLASS(klass);
  auto* basetr_class  = GST_BASE_TRANSFORM_CLASS(klass);
  auto* vfilter_class = GST_VIDEO_FILTER_CLASS(klass);

  gobject_class->set_property = gst_mozza_mp_set_property;
  gobject_class->get_property = gst_mozza_mp_get_property;
  gobject_class->finalize     = gst_mozza_mp_finalize;

  g_object_class_install_property(
      gobject_class, PROP_MODEL_PATH,
      g_param_spec_string("model", "Model path",
                          "Path to face_landmarker.task",
                          nullptr, G_PARAM_READWRITE));

  g_object_class_install_property(
      gobject_class, PROP_DEFORM_PATH,
      g_param_spec_string("deform", "Deformation file (.dfm)",
                          "Path to deformation file with barycentric rules",
                          nullptr, G_PARAM_READWRITE));

  // Alias: "dfm" → same storage as "deform"
  g_object_class_install_property(
      gobject_class, PROP_DFM_ALIAS,
      g_param_spec_string("dfm", "Deformation file (.dfm) [alias]",
                          "Alias for 'deform'",
                          nullptr, G_PARAM_READWRITE));

  g_object_class_install_property(
      gobject_class, PROP_ALPHA,
      g_param_spec_float("alpha", "Smile intensity multiplicator",
                         "Scales the intensity of the deformation (negative=frown)",
                         -10.f, 10.f, 1.f,
                         G_PARAM_READWRITE));

  g_object_class_install_property(
      gobject_class, PROP_OVERLAY,
      g_param_spec_boolean("overlay", "Debug overlay",
                           "Draw src/dst control points and vectors",
                           FALSE, G_PARAM_READWRITE));

  g_object_class_install_property(
      gobject_class, PROP_DROP,
      g_param_spec_boolean("drop", "Drop on no face",
                           "Drop frames when no face is detected",
                           FALSE, G_PARAM_READWRITE));

  g_object_class_install_property(
      gobject_class, PROP_SHOW_LANDMARKS,
      g_param_spec_boolean("show-landmarks", "Draw landmarks",
                           "Draw all detected landmarks (blue) even without DFM",
                           FALSE, G_PARAM_READWRITE));

  g_object_class_install_property(
      gobject_class, PROP_STRICT_DFM,
      g_param_spec_boolean("strict-dfm", "Fail when DFM fails to load",
                           "If true and deform path is given but cannot be loaded, start() fails",
                           FALSE, G_PARAM_READWRITE));

  g_object_class_install_property(
      gobject_class, PROP_FORCE_RGB,
      g_param_spec_boolean("force-rgb", "Accept property for parity (no-op)",
                           "No-op: pads are RGBA; keep videoconvert/caps upstream to ensure RGBA",
                           FALSE, G_PARAM_READWRITE));

  g_object_class_install_property(
      gobject_class, PROP_IGNORE_TS,
      g_param_spec_boolean("ignore-timestamps", "Force ts=0",
                           "When true, pass 0us as timestamp into the detector",
                           FALSE, G_PARAM_READWRITE));

  g_object_class_install_property(
      gobject_class, PROP_LOG_EVERY,
      g_param_spec_uint("log-every", "Periodic log interval",
                        "Log every N frames (0 disables periodic logs)",
                        0, 1000000, 60, G_PARAM_READWRITE));

  // Accept but ignore: user-id (for Ducksoup uniform configs)
  g_object_class_install_property(
      gobject_class, PROP_USER_ID,
      g_param_spec_string("user-id", "User ID",
                          "Opaque user identifier (accepted but not used)",
                          nullptr, G_PARAM_READWRITE));

  gst_element_class_set_static_metadata(GST_ELEMENT_CLASS(klass),
      "Mozza MP (runtime loader)", "Filter/Effect/Video",
      "Applies DFM-driven MLS deformation using mp_runtime landmarks",
      "DuckSoup Lab / CNRS");

  gst_element_class_add_pad_template(GST_ELEMENT_CLASS(klass),
      gst_static_pad_template_get(&sink_template));
  gst_element_class_add_pad_template(GST_ELEMENT_CLASS(klass),
      gst_static_pad_template_get(&src_template));

  basetr_class->start = gst_mozza_mp_start;
  basetr_class->stop  = gst_mozza_mp_stop;

  vfilter_class->set_info           = gst_mozza_mp_set_info;
  vfilter_class->transform_frame_ip = gst_mozza_mp_transform_frame_ip;
}

static void gst_mozza_mp_init(GstMozzaMp* self) {
  self->model_path     = nullptr;
  self->deform_path    = nullptr;
  self->alpha          = 1.0f;
  self->overlay        = FALSE;
  self->drop           = FALSE;
  self->show_landmarks = FALSE;
  self->strict_dfm     = FALSE;
  self->force_rgb      = FALSE;
  self->ignore_ts      = FALSE;
  self->log_every      = 60;
  self->user_id        = nullptr;
  self->frame_count    = 0;
  self->mp_ctx         = nullptr;
  self->dbg_invert_if_zero = FALSE;
  self->dbg_affine         = FALSE;
  self->dbg_hash           = FALSE;
}

static gboolean plugin_init(GstPlugin* plugin) {
  return gst_element_register(plugin, "mozza_mp", GST_RANK_NONE, GST_TYPE_MOZZA_MP);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    mozzamp,
    "Facial deformation (MLS) via mp_runtime",
    plugin_init,
    "1.02",
    "LGPL",
    "mozza_mp",
    "https://github.com/ducksouplab/mozza_mp")
