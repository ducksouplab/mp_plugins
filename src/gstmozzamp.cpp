// MediaPipe-based smile/frown transformer with MLS, RGBA in-place.
//
// Element: mozza_mp
// Props:
//   model   : path to face_landmarker.task (string, required)
//   deform  : path to deformation .dfm (string, required)
//   alpha   : float, [-10..10], default 1.0
//   overlay : bool, default false (draw src/dst control points + vectors)
//   drop    : bool, default false (drop frame when no face)
//
// Caps: video/x-raw, format=RGBA

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// ---- MediaPipe Tasks (same set as your gstfacelandmarks) ----
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker_result.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"

#include "dfm.hpp"              // DFM parser (keeps mozza format)
#include "deform_utils.hpp"     // build groups & MLS warp helpers
#include "imgwarp/imgwarp_mls_rigid.h"  // from your imgwarp.zip

using mediapipe::Image;
using mediapipe::ImageFrame;
using mediapipe::ImageFormat;

namespace mp_core   = ::mediapipe::tasks::core;
namespace mp_vision = ::mediapipe::tasks::vision;
namespace mp_face   = ::mediapipe::tasks::vision::face_landmarker;

#ifndef PACKAGE
#define PACKAGE "mozza_mp"
#endif

G_BEGIN_DECLS

#define GST_TYPE_MOZZA_MP (gst_mozza_mp_get_type())
G_DECLARE_FINAL_TYPE(GstMozzaMp, gst_mozza_mp, GST, MOZZA_MP, GstVideoFilter)

struct _GstMozzaMp {
  GstVideoFilter parent;

  // properties
  gchar*   model_path;
  gchar*   deform_path;
  gfloat   alpha;
  gboolean overlay;
  gboolean drop;

  // runtime
  std::unique_ptr<mp_face::FaceLandmarker> landmarker;
  std::optional<Deformations> dfm;
  std::unique_ptr<ImgWarp_MLS_Rigid> mls;

  // stats
  guint64 frame_count;
};

G_END_DECLS

// ── Properties ────────────────────────────────────────────────────────────────
enum {
  PROP_0,
  PROP_MODEL_PATH,
  PROP_DEFORM_PATH,
  PROP_ALPHA,
  PROP_OVERLAY,
  PROP_DROP
};

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
  "sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw, format=RGBA"));
static GstStaticPadTemplate src_template  = GST_STATIC_PAD_TEMPLATE(
  "src",  GST_PAD_SRC,  GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw, format=RGBA"));

G_DEFINE_TYPE(GstMozzaMp, gst_mozza_mp, GST_TYPE_VIDEO_FILTER)

// ── Utils: software overlay (same style as your gstfacelandmarks) ────────────
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

// ── GObject props ────────────────────────────────────────────────────────────
static void gst_mozza_mp_set_property(GObject* obj, guint prop_id,
                                      const GValue* value, GParamSpec* pspec) {
  auto* self = GST_MOZZA_MP(obj);
  switch (prop_id) {
    case PROP_MODEL_PATH:
      g_free(self->model_path);
      self->model_path = g_value_dup_string(value);
      break;
    case PROP_DEFORM_PATH:
      g_free(self->deform_path);
      self->deform_path = g_value_dup_string(value);
      // (lazy-load on start/first frame so we can log errors cleanly)
      break;
    case PROP_ALPHA:   self->alpha   = g_value_get_float(value); break;
    case PROP_OVERLAY: self->overlay = g_value_get_boolean(value); break;
    case PROP_DROP:    self->drop    = g_value_get_boolean(value); break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(obj, prop_id, pspec);
  }
}

static void gst_mozza_mp_get_property(GObject* obj, guint prop_id,
                                      GValue* value, GParamSpec* pspec) {
  auto* self = GST_MOZZA_MP(obj);
  switch (prop_id) {
    case PROP_MODEL_PATH: g_value_set_string (value, self->model_path); break;
    case PROP_DEFORM_PATH:g_value_set_string (value, self->deform_path); break;
    case PROP_ALPHA:      g_value_set_float  (value, self->alpha);      break;
    case PROP_OVERLAY:    g_value_set_boolean(value, self->overlay);    break;
    case PROP_DROP:       g_value_set_boolean(value, self->drop);       break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(obj, prop_id, pspec);
  }
}

// ── Lifecycle ────────────────────────────────────────────────────────────────
static gboolean gst_mozza_mp_start(GstBaseTransform* base) {
  auto* self = GST_MOZZA_MP(base);

  if (!self->model_path || !g_file_test(self->model_path, G_FILE_TEST_EXISTS)) {
    GST_ERROR_OBJECT(self, "Set a valid model path: model=/path/to/face_landmarker.task");
    return FALSE;
  }

  auto options = std::make_unique<mp_face::FaceLandmarkerOptions>();
  options->base_options = mp_core::BaseOptions();
  options->base_options.model_asset_path = self->model_path;
  options->running_mode = mp_vision::core::RunningMode::VIDEO;
  options->num_faces    = 1;
  options->output_face_blendshapes = false;
  options->output_facial_transformation_matrixes = false;

  absl::StatusOr<std::unique_ptr<mp_face::FaceLandmarker>> lm =
      mp_face::FaceLandmarker::Create(std::move(options));
  if (!lm.ok()) {
    GST_ERROR_OBJECT(self, "FaceLandmarker creation failed: %s", lm.status().ToString().c_str());
    return FALSE;
  }
  self->landmarker = std::move(lm.value());

  self->mls = std::make_unique<ImgWarp_MLS_Rigid>();
  self->mls->gridSize = 5;    // close to mozza’s setting
  self->mls->preScale = true;
  self->mls->alpha    = 1.4;  // same as mozza

  // Try loading DFM now if provided
  if (self->deform_path) {
    self->dfm = load_dfm(self->deform_path);
    if (!self->dfm) {
      GST_ERROR_OBJECT(self, "Failed to load DFM: %s", self->deform_path);
      // don't fail element start; we can still run pass-through
    }
  }

  self->frame_count = 0;
  return TRUE;
}

static gboolean gst_mozza_mp_stop(GstBaseTransform* base) {
  auto* self = GST_MOZZA_MP(base);
  self->landmarker.reset();
  self->mls.reset();
  self->dfm.reset();
  return TRUE;
}

static gboolean gst_mozza_mp_set_info(GstVideoFilter*, GstCaps*, GstVideoInfo*,
                                      GstCaps*, GstVideoInfo*) { return TRUE; }

// ── Per-frame work ───────────────────────────────────────────────────────────
static GstFlowReturn gst_mozza_mp_transform_frame_ip(GstVideoFilter* vf,
                                                     GstVideoFrame* f) {
  auto* self = GST_MOZZA_MP(vf);
  if (!self->landmarker) return GST_FLOW_OK;

  // Load DFM lazily if set after start
  if (!self->dfm && self->deform_path) {
    self->dfm = load_dfm(self->deform_path);
    if (!self->dfm) {
      GST_ERROR_OBJECT(self, "Failed to load DFM: %s", self->deform_path);
    }
  }

  const int W      = GST_VIDEO_FRAME_WIDTH(f);
  const int H      = GST_VIDEO_FRAME_HEIGHT(f);
  const int stride = GST_VIDEO_FRAME_PLANE_STRIDE(f, 0);
  auto* data       = static_cast<uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(f, 0));

  // Wrap as MediaPipe ImageFrame (SRGBA) without copy
  auto frame_ptr = std::make_shared<ImageFrame>(ImageFormat::SRGBA, W, H, /*alignment*/1);
  uint8_t* dst = frame_ptr->MutablePixelData();
  const int rowbytes = W * 4;
  for (int y = 0; y < H; ++y) {
    std::memcpy(dst + y * rowbytes, data + y * stride, rowbytes);
  }
  Image mp_image(frame_ptr);

  const GstClockTime pts = GST_BUFFER_PTS(f->buffer);
  const int64_t ts_ms = (pts == GST_CLOCK_TIME_NONE) ? 0 : static_cast<int64_t>(GST_TIME_AS_MSECONDS(pts));

  absl::StatusOr<mp_face::FaceLandmarkerResult> out =
      self->landmarker->DetectForVideo(mp_image, ts_ms);

  if (!out.ok()) {
    GST_WARNING_OBJECT(self, "DetectForVideo failed: %s", out.status().ToString().c_str());
    return GST_FLOW_OK; // pass-through on failure
  }

  const auto& faces = out->face_landmarks;
  if (faces.empty()) {
    if (self->drop) return GST_BASE_TRANSFORM_FLOW_DROPPED;
    return GST_FLOW_OK;
  }

  // Use first face; normalized coords → pixel coords
  const auto& lms = faces[0].landmarks;
  std::vector<cv::Point2f> L; L.reserve(lms.size());
  for (const auto& lm : lms) {
    L.emplace_back(lm.x * W, lm.y * H);
  }

  if (self->dfm) {
    // Build src/dst groups from DFM (same mozza semantics)
    std::vector<std::vector<cv::Point2f>> srcGroups, dstGroups;
    build_groups_from_dfm(*self->dfm, L, self->alpha, srcGroups, dstGroups);

    // Wrap frame as cv::Mat with stride (no copy)
    cv::Mat img_rgba(H, W, CV_8UC4, data, static_cast<size_t>(stride));

    // MLS per group on tight ROI
    for (size_t i = 0; i < srcGroups.size(); ++i) {
      compute_MLS_on_ROI(img_rgba, *self->mls, srcGroups[i], dstGroups[i]);
    }

    if (self->overlay) {
      // Software overlay: green = dst, red = src, line src→dst
      const uint32_t green = 0x00FF00FFu, red = 0xFF0000FFu;
      for (size_t g = 0; g < srcGroups.size(); ++g) {
        for (size_t i = 0; i < srcGroups[g].size(); ++i) {
          const auto& s = srcGroups[g][i];
          const auto& d = dstGroups[g][i];
          const int sx = std::clamp((int)std::lround(s.x), 0, W-1);
          const int sy = std::clamp((int)std::lround(s.y), 0, H-1);
          const int dx = std::clamp((int)std::lround(d.x), 0, W-1);
          const int dy = std::clamp((int)std::lround(d.y), 0, H-1);
          draw_dot(data, W, H, stride, sx, sy, 2, red);
          draw_dot(data, W, H, stride, dx, dy, 2, green);
          draw_line(data, W, H, stride, sx, sy, dx, dy, green);
        }
      }
    }
  }

  return GST_FLOW_OK;
}

// ── Class / init / plugin boilerplate ────────────────────────────────────────
static void gst_mozza_mp_class_init(GstMozzaMpClass* klass) {
  auto* gobject_class = G_OBJECT_CLASS(klass);
  auto* basetr_class  = GST_BASE_TRANSFORM_CLASS(klass);
  auto* vfilter_class = GST_VIDEO_FILTER_CLASS(klass);

  gobject_class->set_property = gst_mozza_mp_set_property;
  gobject_class->get_property = gst_mozza_mp_get_property;

  g_object_class_install_property(
      gobject_class, PROP_MODEL_PATH,
      g_param_spec_string("model", "Model path",
                          "Path to MediaPipe face_landmarker.task",
                          nullptr, G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_DEFORM_PATH,
      g_param_spec_string("deform", "Deformation file (.dfm)",
                          "Path to deformation file with barycentric rules",
                          nullptr, G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_ALPHA,
      g_param_spec_float("alpha", "Smile intensity multiplicator",
                         "Scales the intensity of the deformation (negative=frown)",
                         -10.f, 10.f, 1.f, G_PARAM_READWRITE));
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

  gst_element_class_set_static_metadata(GST_ELEMENT_CLASS(klass),
      "Mozza MP (MediaPipe smile transformer)", "Filter/Effect/Video",
      "Applies DFM-driven MLS deformation using MediaPipe landmarks",
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
  self->model_path = nullptr;
  self->deform_path = nullptr;
  self->alpha   = 1.0f;
  self->overlay = FALSE;
  self->drop    = FALSE;
  self->frame_count = 0;
}

static gboolean plugin_init(GstPlugin* plugin) {
  return gst_element_register(plugin, "mozza_mp", GST_RANK_NONE, GST_TYPE_MOZZA_MP);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    mozzamp,
    "MediaPipe-based facial deformation (MLS)",
    plugin_init,
    "1.0",
    "LGPL",
    "mozza_mp",
    "https://github.com/ducksouplab/mozza_mp")