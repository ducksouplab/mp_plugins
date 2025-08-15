// GStreamer video filter that runs MediaPipe Face Landmarker (Tasks) on CPU
// and overlays 2D landmarks directly in RGBA (no OpenCV).
//
// Element: facelandmarks
// Props:
//   model      : path to face_landmarker.task (string, required)
//   max-faces  : int  >=1 (default 1)
//   draw       : bool (default true)
//   radius     : int  >=1 (default 2)
//   color      : uint RGBA 0xRRGGBBAA (default 0x00FF00FF)
// Caps: video/x-raw, format=RGBA

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker_result.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h" 

using mediapipe::Image;
using mediapipe::ImageFrame;
using mediapipe::ImageFormat;

namespace mp_core   = ::mediapipe::tasks::core;
namespace mp_vision = ::mediapipe::tasks::vision;
namespace mp_face   = ::mediapipe::tasks::vision::face_landmarker;

#ifndef PACKAGE
#define PACKAGE "facelandmarks"
#endif

G_BEGIN_DECLS

#define GST_TYPE_FACE_LANDMARKS (gst_face_landmarks_get_type())
G_DECLARE_FINAL_TYPE(GstFaceLandmarks, gst_face_landmarks, GST, FACE_LANDMARKS, GstVideoFilter)

struct _GstFaceLandmarks {
  GstVideoFilter parent;

  gchar*   model_path;
  gint     max_faces;
  gboolean draw;
  gint     radius;
  guint    color_rgba; // 0xRRGGBBAA

  std::unique_ptr<mp_face::FaceLandmarker> landmarker;
};

G_END_DECLS

enum {
  PROP_0,
  PROP_MODEL_PATH,
  PROP_MAX_FACES,
  PROP_DRAW,
  PROP_RADIUS,
  PROP_COLOR
};

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
  "sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw, format=RGBA"));
static GstStaticPadTemplate src_template  = GST_STATIC_PAD_TEMPLATE(
  "src",  GST_PAD_SRC,  GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw, format=RGBA"));

G_DEFINE_TYPE(GstFaceLandmarks, gst_face_landmarks, GST_TYPE_VIDEO_FILTER)

// ── Properties ────────────────────────────────────────────────────────────────
static void gst_face_landmarks_set_property(GObject* obj, guint prop_id,
                                            const GValue* value, GParamSpec* pspec) {
  auto* self = GST_FACE_LANDMARKS(obj);
  switch (prop_id) {
    case PROP_MODEL_PATH:
      g_free(self->model_path);
      self->model_path = g_value_dup_string(value);
      break;
    case PROP_MAX_FACES: self->max_faces  = std::max(1, g_value_get_int(value)); break;
    case PROP_DRAW:      self->draw       = g_value_get_boolean(value);          break;
    case PROP_RADIUS:    self->radius     = std::max(1, g_value_get_int(value)); break;
    case PROP_COLOR:     self->color_rgba = g_value_get_uint(value);             break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(obj, prop_id, pspec);
  }
}

static void gst_face_landmarks_get_property(GObject* obj, guint prop_id,
                                            GValue* value, GParamSpec* pspec) {
  auto* self = GST_FACE_LANDMARKS(obj);
  switch (prop_id) {
    case PROP_MODEL_PATH: g_value_set_string (value, self->model_path);   break;
    case PROP_MAX_FACES:  g_value_set_int    (value, self->max_faces);    break;
    case PROP_DRAW:       g_value_set_boolean(value, self->draw);         break;
    case PROP_RADIUS:     g_value_set_int    (value, self->radius);       break;
    case PROP_COLOR:      g_value_set_uint   (value, self->color_rgba);   break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(obj, prop_id, pspec);
  }
}

// ── Lifecycle ────────────────────────────────────────────────────────────────
static gboolean gst_face_landmarks_start(GstBaseTransform* base) {
  auto* self = GST_FACE_LANDMARKS(base);

  if (!self->model_path || !g_file_test(self->model_path, G_FILE_TEST_EXISTS)) {
    GST_ERROR_OBJECT(self, "Set a valid model path: model=/path/to/face_landmarker.task");
    return FALSE;
  }

  // Tasks v0.10.x uses Create(std::unique_ptr<Options>) and Image input.
  auto options = std::make_unique<mp_face::FaceLandmarkerOptions>();
  options->base_options = mp_core::BaseOptions();
  options->base_options.model_asset_path = self->model_path;  // .task bundle
  options->running_mode = mp_vision::core::RunningMode::VIDEO; // timestamps in ms
  options->num_faces    = self->max_faces;
  options->output_face_blendshapes = false;
  options->output_facial_transformation_matrixes = false;

  absl::StatusOr<std::unique_ptr<mp_face::FaceLandmarker>> lm =
      mp_face::FaceLandmarker::Create(std::move(options));
  if (!lm.ok()) {
    GST_ERROR_OBJECT(self, "FaceLandmarker creation failed: %s",
                     lm.status().ToString().c_str());
    return FALSE;
  }
  self->landmarker = std::move(lm.value());
  return TRUE;
}

static gboolean gst_face_landmarks_stop(GstBaseTransform* base) {
  auto* self = GST_FACE_LANDMARKS(base);
  self->landmarker.reset();
  return TRUE;
}

// ── RGBA overlay helpers ─────────────────────────────────────────────────────
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

static void overlay_landmarks(uint8_t* data, int W, int H, int stride,
                              const mp_face::FaceLandmarkerResult& res,
                              int radius, uint32_t rgba) {
  for (const auto& face : res.face_landmarks) {
    const auto& lms = face.landmarks;        // <-- Tasks 0.10.x shape
    for (const auto& lm : lms) {
      const int x = std::clamp(static_cast<int>(lm.x * W + 0.5f), 0, W - 1);
      const int y = std::clamp(static_cast<int>(lm.y * H + 0.5f), 0, H - 1);
      draw_dot(data, W, H, stride, x, y, radius, rgba);
    }
  }
}

// ── Per-frame work ───────────────────────────────────────────────────────────
static GstFlowReturn gst_face_landmarks_transform_frame_ip(GstVideoFilter* vf,
                                                           GstVideoFrame* f) {
  auto* self = GST_FACE_LANDMARKS(vf);
  if (!self->landmarker) return GST_FLOW_OK;

  const int W      = GST_VIDEO_FRAME_WIDTH(f);
  const int H      = GST_VIDEO_FRAME_HEIGHT(f);
  const int stride = GST_VIDEO_FRAME_PLANE_STRIDE(f, 0);
  auto* data       = static_cast<uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(f, 0));

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
  if (out.ok() && self->draw) {
    overlay_landmarks(data, W, H, stride, out.value(), self->radius, self->color_rgba);
  } else if (!out.ok()) {
    GST_WARNING_OBJECT(self, "DetectForVideo failed: %s", out.status().ToString().c_str());
  }
  return GST_FLOW_OK;
}

static gboolean gst_face_landmarks_set_info(GstVideoFilter*, GstCaps*, GstVideoInfo*,
                                            GstCaps*, GstVideoInfo*) { return TRUE; }

static void gst_face_landmarks_class_init(GstFaceLandmarksClass* klass) {
  auto* gobject_class = G_OBJECT_CLASS(klass);
  auto* basetr_class  = GST_BASE_TRANSFORM_CLASS(klass);
  auto* vfilter_class = GST_VIDEO_FILTER_CLASS(klass);

  gobject_class->set_property = gst_face_landmarks_set_property;
  gobject_class->get_property = gst_face_landmarks_get_property;

  g_object_class_install_property(
      gobject_class, PROP_MODEL_PATH,
      g_param_spec_string("model", "Model path",
                          "Path to MediaPipe face_landmarker.task",
                          nullptr, G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_MAX_FACES,
      g_param_spec_int("max-faces", "Max faces",
                       "Maximum number of faces to detect",
                       1, 16, 1, G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_DRAW,
      g_param_spec_boolean("draw", "Draw landmarks",
                           "Overlay landmarks on the frame",
                           TRUE, G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_RADIUS,
      g_param_spec_int("radius", "Dot radius (px)",
                       "Radius of landmark dots in pixels",
                       1, 10, 2, G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_COLOR,
      g_param_spec_uint("color", "RGBA color 0xRRGGBBAA",
                        "Packed RGBA color for landmarks",
                        0, G_MAXUINT, 0x00FF00FFu, G_PARAM_READWRITE));

  gst_element_class_set_static_metadata(GST_ELEMENT_CLASS(klass),
      "MediaPipe Face Landmarks Overlay", "Filter/Effect/Video",
      "Detects face landmarks with MediaPipe Tasks and overlays them",
      "You <you@example.com>");

  gst_element_class_add_pad_template(GST_ELEMENT_CLASS(klass),
      gst_static_pad_template_get(&sink_template));
  gst_element_class_add_pad_template(GST_ELEMENT_CLASS(klass),
      gst_static_pad_template_get(&src_template));

  basetr_class->start = gst_face_landmarks_start;
  basetr_class->stop  = gst_face_landmarks_stop;

  vfilter_class->set_info           = gst_face_landmarks_set_info;
  vfilter_class->transform_frame_ip = gst_face_landmarks_transform_frame_ip;
}

static void gst_face_landmarks_init(GstFaceLandmarks* self) {
  self->model_path = nullptr;
  self->max_faces  = 1;
  self->draw       = TRUE;
  self->radius     = 2;
  self->color_rgba = 0x00FF00FFu; // green, opaque
}

static gboolean plugin_init(GstPlugin* plugin) {
  return gst_element_register(plugin, "facelandmarks", GST_RANK_NONE, GST_TYPE_FACE_LANDMARKS);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    facelandmarks,
    "MediaPipe face landmarks overlay",
    plugin_init,
    "1.0",
    "LGPL",
    "facelandmarks",
    "https://example.com")