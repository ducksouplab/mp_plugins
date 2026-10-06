// gstmozzamesh.cpp: mozza_mesh, face transformations from displacement-field
// bases with a mesh warp.
//
// Element: mozza_mesh (RGBA in, RGBA out, in place)
//
// Each frame: MediaPipe face landmarks (shared runtime, as mozza_mp), optional
// OneEuro smoothing, displacement = sum of amplitude * field over the loaded
// basis fields (face units, converted with the face's eye-corner frame), then
// a piecewise-affine warp on the face mesh (see facewarp.hpp).
//
// Properties
//   model              path to face_landmarker.task (required)
//   basis              comma-separated basis JSON files (e.g. au_basis_v1.json,trait_basis_v1.json)
//   amplitudes         "AU12=0.5,DOM_o=2": sets all amplitudes at once (unlisted ones -> 0)
//   au1 ... au43       float amplitude of one AU field (AU1 ... AU43)
//   trust, dom, threat float amplitude of TRUST_o, DOM_o, THREAT (in SD)
//   fold-guard         max folded area (px^2) inside the face before the deformation
//                      is scaled down; recovers gradually (default 5; negative = off)
//   smooth-landmarks, min-cutoff, beta   OneEuro landmark smoothing (as mozza_mp_gpu)
//   show-landmarks, landmark-radius, landmark-color, drop, no-warp,
//   ignore-timestamps, threads, max-faces, log-every, user-id   (as mozza_mp)
// Amplitudes can be changed while playing (e.g. from DuckSoup's controlFx).

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>

#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "facewarp.hpp"
#include "mp_runtime.h"
#include "mp_runtime_loader.h"

#ifndef PACKAGE
#define PACKAGE "mozza_mesh"
#endif

GST_DEBUG_CATEGORY_STATIC(gst_mozza_mesh_debug_category);
#define GST_CAT_DEFAULT gst_mozza_mesh_debug_category

// Named float properties and the basis fields they drive.
struct NamedAmp { const char* prop; const char* field; const char* blurb; };
static const NamedAmp kNamed[] = {
  {"au1", "AU1", "Inner brow raiser"},
  {"au2", "AU2", "Outer brow raiser"},
  {"au4", "AU4", "Brow lowerer"},
  {"au5", "AU5", "Upper lid raiser (eye widening)"},
  {"au7", "AU7", "Lid tightener"},
  {"au12", "AU12", "Lip corner puller (smile)"},
  {"au15", "AU15", "Lip corner depressor"},
  {"au20", "AU20", "Lip stretcher"},
  {"au43", "AU43", "Eye closure (negative: widening)"},
  {"trust", "TRUST_o", "Trustworthiness, in SD (orthogonal model)"},
  {"dom", "DOM_o", "Dominance, in SD (orthogonal model)"},
  {"threat", "THREAT", "Threat, in SD"},
};
static constexpr int kNumNamed = sizeof(kNamed) / sizeof(kNamed[0]);
// Fold guard: per-frame recovery of the scale after a fold (0.05 = ~0.7 s at 30 fps from 0 to 1)
static constexpr float kGuardRecovery = 0.05f;

// ── One-Euro filter (same as mozza_mp_gpu) ───────────────────────────────────
struct OneEuroFilter {
  float min_cutoff = 2.0f, beta = 0.05f, d_cutoff = 1.0f;
  bool first_time = true;
  float x_prev = 0.f, dx_prev = 0.f;
  static float alpha(float cutoff, float dt) {
    float r = 2.0f * (float)M_PI * cutoff * dt;
    return r / (r + 1.0f);
  }
  float filter(float x, float dt) {
    if (first_time) { first_time = false; x_prev = x; dx_prev = 0.f; return x; }
    if (dt <= 0) return x_prev;
    float dx = (x - x_prev) / dt;
    float edx = alpha(d_cutoff, dt) * dx + (1.0f - alpha(d_cutoff, dt)) * dx_prev;
    float a = alpha(min_cutoff + beta * std::abs(edx), dt);
    x_prev = a * x + (1.0f - a) * x_prev;
    dx_prev = edx;
    return x_prev;
  }
};

G_BEGIN_DECLS
#define GST_TYPE_MOZZA_MESH (gst_mozza_mesh_get_type())
G_DECLARE_FINAL_TYPE(GstMozzaMesh, gst_mozza_mesh, GST, MOZZA_MESH, GstVideoFilter)
G_END_DECLS

struct _GstMozzaMesh {
  GstVideoFilter parent;

  // properties
  gchar* model_path;
  gchar* basis_paths;
  gfloat fold_guard;
  gboolean smooth_landmarks;
  gfloat min_cutoff, beta;
  gboolean show_landmarks;
  gint lm_radius;
  guint lm_color;
  gboolean drop, no_warp, ignore_ts;
  gint num_threads, max_faces;
  guint log_every;
  gchar* user_id;

  // shared with property setters (guarded by lock)
  GMutex lock;
  std::map<std::string, float>* amps;
  gboolean basis_dirty;

  // runtime
  MpFaceCtx* mp_ctx;
  facewarp::Basis* basis;
  std::set<std::string>* warned;
  std::vector<OneEuroFilter>* fx;
  std::vector<OneEuroFilter>* fy;
  GstClockTime prev_pts;
  float guard_scale;  // fold-guard scale applied on the previous frame
  guint64 frame_count;
  double sum_detect_us, sum_warp_us;
  guint64 timing_count;
};

G_DEFINE_TYPE(GstMozzaMesh, gst_mozza_mesh, GST_TYPE_VIDEO_FILTER)

enum {
  PROP_0,
  PROP_MODEL,
  PROP_BASIS,
  PROP_AMPLITUDES,
  PROP_FOLD_GUARD,
  PROP_SMOOTH_LANDMARKS,
  PROP_MIN_CUTOFF,
  PROP_BETA,
  PROP_SHOW_LANDMARKS,
  PROP_LM_RADIUS,
  PROP_LM_COLOR,
  PROP_DROP,
  PROP_NO_WARP,
  PROP_IGNORE_TS,
  PROP_NUM_THREADS,
  PROP_MAX_FACES,
  PROP_LOG_EVERY,
  PROP_USER_ID,
  PROP_NAMED_BASE,  // + index into kNamed
};

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
  "sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw, format=RGBA"));
static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
  "src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw, format=RGBA"));

// ── Properties ───────────────────────────────────────────────────────────────

static void gst_mozza_mesh_set_property(GObject* obj, guint id, const GValue* value, GParamSpec* pspec) {
  auto* self = GST_MOZZA_MESH(obj);
  if (id >= PROP_NAMED_BASE && id < PROP_NAMED_BASE + kNumNamed) {
    const NamedAmp& n = kNamed[id - PROP_NAMED_BASE];
    float v = g_value_get_float(value);
    g_mutex_lock(&self->lock);
    if (v == 0.f) self->amps->erase(n.field); else (*self->amps)[n.field] = v;
    g_mutex_unlock(&self->lock);
    GST_DEBUG_OBJECT(self, "prop:%s (%s) = %.3f", n.prop, n.field, v);
    return;
  }
  switch (id) {
    case PROP_MODEL:
      g_free(self->model_path);
      self->model_path = g_value_dup_string(value);
      break;
    case PROP_BASIS:
      g_mutex_lock(&self->lock);
      g_free(self->basis_paths);
      self->basis_paths = g_value_dup_string(value);
      self->basis_dirty = TRUE;  // (re)loaded on the next frame
      g_mutex_unlock(&self->lock);
      break;
    case PROP_AMPLITUDES: {
      const gchar* s = g_value_get_string(value);
      std::map<std::string, float> parsed;
      std::string err;
      if (!facewarp::parse_amplitudes(s ? s : "", &parsed, &err)) {
        GST_WARNING_OBJECT(self, "amplitudes ignored: %s", err.c_str());
        break;
      }
      g_mutex_lock(&self->lock);
      *self->amps = parsed;
      g_mutex_unlock(&self->lock);
      GST_DEBUG_OBJECT(self, "prop:amplitudes = %s", s ? s : "");
      break;
    }
    case PROP_FOLD_GUARD: self->fold_guard = g_value_get_float(value); break;
    case PROP_SMOOTH_LANDMARKS: self->smooth_landmarks = g_value_get_boolean(value); break;
    case PROP_MIN_CUTOFF: self->min_cutoff = g_value_get_float(value); break;
    case PROP_BETA: self->beta = g_value_get_float(value); break;
    case PROP_SHOW_LANDMARKS: self->show_landmarks = g_value_get_boolean(value); break;
    case PROP_LM_RADIUS: self->lm_radius = g_value_get_int(value); break;
    case PROP_LM_COLOR: self->lm_color = g_value_get_uint(value); break;
    case PROP_DROP: self->drop = g_value_get_boolean(value); break;
    case PROP_NO_WARP: self->no_warp = g_value_get_boolean(value); break;
    case PROP_IGNORE_TS: self->ignore_ts = g_value_get_boolean(value); break;
    case PROP_NUM_THREADS: self->num_threads = g_value_get_int(value); break;
    case PROP_MAX_FACES: self->max_faces = g_value_get_int(value); break;
    case PROP_LOG_EVERY: self->log_every = g_value_get_uint(value); break;
    case PROP_USER_ID:
      g_free(self->user_id);
      self->user_id = g_value_dup_string(value);
      break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(obj, id, pspec);
  }
}

static void gst_mozza_mesh_get_property(GObject* obj, guint id, GValue* value, GParamSpec* pspec) {
  auto* self = GST_MOZZA_MESH(obj);
  if (id >= PROP_NAMED_BASE && id < PROP_NAMED_BASE + kNumNamed) {
    g_mutex_lock(&self->lock);
    auto it = self->amps->find(kNamed[id - PROP_NAMED_BASE].field);
    float v = it == self->amps->end() ? 0.f : it->second;
    g_mutex_unlock(&self->lock);
    g_value_set_float(value, v);
    return;
  }
  switch (id) {
    case PROP_MODEL: g_value_set_string(value, self->model_path); break;
    case PROP_BASIS:
      g_mutex_lock(&self->lock);
      g_value_set_string(value, self->basis_paths);
      g_mutex_unlock(&self->lock);
      break;
    case PROP_AMPLITUDES: {
      g_mutex_lock(&self->lock);
      std::string s = facewarp::format_amplitudes(*self->amps);
      g_mutex_unlock(&self->lock);
      g_value_set_string(value, s.c_str());
      break;
    }
    case PROP_FOLD_GUARD: g_value_set_float(value, self->fold_guard); break;
    case PROP_SMOOTH_LANDMARKS: g_value_set_boolean(value, self->smooth_landmarks); break;
    case PROP_MIN_CUTOFF: g_value_set_float(value, self->min_cutoff); break;
    case PROP_BETA: g_value_set_float(value, self->beta); break;
    case PROP_SHOW_LANDMARKS: g_value_set_boolean(value, self->show_landmarks); break;
    case PROP_LM_RADIUS: g_value_set_int(value, self->lm_radius); break;
    case PROP_LM_COLOR: g_value_set_uint(value, self->lm_color); break;
    case PROP_DROP: g_value_set_boolean(value, self->drop); break;
    case PROP_NO_WARP: g_value_set_boolean(value, self->no_warp); break;
    case PROP_IGNORE_TS: g_value_set_boolean(value, self->ignore_ts); break;
    case PROP_NUM_THREADS: g_value_set_int(value, self->num_threads); break;
    case PROP_MAX_FACES: g_value_set_int(value, self->max_faces); break;
    case PROP_LOG_EVERY: g_value_set_uint(value, self->log_every); break;
    case PROP_USER_ID: g_value_set_string(value, self->user_id); break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(obj, id, pspec);
  }
}

// ── Basis loading ────────────────────────────────────────────────────────────

// Loads the comma-separated basis files into self->basis. Called with the lock held.
static bool load_basis_locked(GstMozzaMesh* self) {
  self->basis->clear();
  self->warned->clear();
  self->basis_dirty = FALSE;
  if (!self->basis_paths || !*self->basis_paths) return true;
  std::stringstream ss(self->basis_paths);
  std::string path, err;
  bool ok = true;
  while (std::getline(ss, path, ',')) {
    size_t a = path.find_first_not_of(" \t"), b = path.find_last_not_of(" \t");
    if (a == std::string::npos) continue;
    path = path.substr(a, b - a + 1);
    if (!self->basis->load(path, &err)) {
      GST_ERROR_OBJECT(self, "basis: %s", err.c_str());
      ok = false;
    }
  }
  std::string names;
  for (const auto& [name, f] : self->basis->fields()) names += (names.empty() ? "" : ", ") + name;
  GST_INFO_OBJECT(self, "basis loaded from '%s': %s", self->basis_paths, names.c_str());
  return ok;
}

// ── Lifecycle ────────────────────────────────────────────────────────────────

static gboolean gst_mozza_mesh_start(GstBaseTransform* base) {
  auto* self = GST_MOZZA_MESH(base);
  if (!self->model_path || !g_file_test(self->model_path, G_FILE_TEST_EXISTS)) {
    GST_ERROR_OBJECT(self, "missing/invalid model: set model=/path/to/face_landmarker.task");
    return FALSE;
  }
  if (!MpApiOK()) {
    GST_ERROR_OBJECT(self, "mp_runtime loader not initialized: %s", mp_runtime_loader::last_error());
    return FALSE;
  }
  MpFaceLandmarkerOptions opts{};
  opts.model_path = self->model_path;
  opts.max_faces = self->max_faces;
  opts.with_blendshapes = 0;
  opts.with_geometry = 0;
  opts.num_threads = self->num_threads;
  opts.delegate = "cpu";
  self->mp_ctx = nullptr;
  int rc = MpApi().face_create(&opts, &self->mp_ctx);
  if (rc != 0 || !self->mp_ctx) {
    GST_ERROR_OBJECT(self, "face landmarker creation failed (rc=%d): %s", rc, mp_runtime_loader::last_error());
    return FALSE;
  }
  g_mutex_lock(&self->lock);
  bool ok = load_basis_locked(self);
  g_mutex_unlock(&self->lock);
  if (!ok) return FALSE;
  self->fx->clear();
  self->fy->clear();
  self->prev_pts = GST_CLOCK_TIME_NONE;
  self->guard_scale = 1.f;
  self->frame_count = 0;
  self->sum_detect_us = self->sum_warp_us = 0;
  self->timing_count = 0;
  return TRUE;
}

static gboolean gst_mozza_mesh_stop(GstBaseTransform* base) {
  auto* self = GST_MOZZA_MESH(base);
  if (self->mp_ctx) { MpApi().face_close(&self->mp_ctx); self->mp_ctx = nullptr; }
  return TRUE;
}

static void gst_mozza_mesh_finalize(GObject* object) {
  auto* self = GST_MOZZA_MESH(object);
  if (self->mp_ctx) { MpApi().face_close(&self->mp_ctx); self->mp_ctx = nullptr; }
  g_clear_pointer(&self->model_path, g_free);
  g_clear_pointer(&self->basis_paths, g_free);
  g_clear_pointer(&self->user_id, g_free);
  delete self->amps;
  delete self->basis;
  delete self->warned;
  delete self->fx;
  delete self->fy;
  g_mutex_clear(&self->lock);
  G_OBJECT_CLASS(gst_mozza_mesh_parent_class)->finalize(object);
}

static void draw_landmarks(uint8_t* data, int W, int H, int stride, const std::vector<cv::Point2f>& L,
                           int radius, guint rgba) {
  const uint8_t c[4] = {(uint8_t)(rgba >> 24), (uint8_t)(rgba >> 16), (uint8_t)(rgba >> 8), (uint8_t)rgba};
  for (const auto& p : L) {
    const int cx = (int)p.x, cy = (int)p.y;
    for (int y = std::max(0, cy - radius); y <= std::min(H - 1, cy + radius); ++y)
      for (int x = std::max(0, cx - radius); x <= std::min(W - 1, cx + radius); ++x)
        if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= radius * radius) {
          uint8_t* px = data + y * stride + x * 4;
          for (int k = 0; k < 3; ++k) px[k] = (uint8_t)((px[k] * (255 - c[3]) + c[k] * c[3]) / 255);
        }
  }
}

// ── Frame ────────────────────────────────────────────────────────────────────

static GstFlowReturn gst_mozza_mesh_transform_frame_ip(GstVideoFilter* vf, GstVideoFrame* f) {
  auto* self = GST_MOZZA_MESH(vf);
  if (!self->mp_ctx) return GST_FLOW_OK;
  self->frame_count++;

  const int W = GST_VIDEO_FRAME_WIDTH(f), H = GST_VIDEO_FRAME_HEIGHT(f);
  const int stride = GST_VIDEO_FRAME_PLANE_STRIDE(f, 0);
  auto* data = static_cast<uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(f, 0));

  const bool timing = self->log_every > 0 && gst_debug_category_get_threshold(GST_CAT_DEFAULT) >= GST_LEVEL_INFO;
  auto t0 = std::chrono::steady_clock::now();

  MpImage img{};
  img.data = data;
  img.width = W;
  img.height = H;
  img.stride = stride;
  img.format = MP_IMAGE_RGBA8888;
  GstClockTime pts = GST_BUFFER_PTS(f->buffer);
  const int64_t ts_us = self->ignore_ts || !GST_CLOCK_TIME_IS_VALID(pts)
                            ? (int64_t)self->frame_count * 33333LL : (int64_t)GST_TIME_AS_USECONDS(pts);
  MpFaceResult out{};
  int rc = MpApi().face_detect(self->mp_ctx, &img, ts_us, &out);
  auto t1 = std::chrono::steady_clock::now();
  if (rc != 0 || out.faces_count == 0) {
    if (rc == 0) MpApi().face_free_result(&out);
    return self->drop ? GST_BASE_TRANSFORM_FLOW_DROPPED : GST_FLOW_OK;
  }

  // Landmarks (pixels), smoothed in normalised coordinates like mozza_mp_gpu
  const MpFace& face = out.faces[0];
  float dt = 1.0f / 30.0f;
  if (!self->ignore_ts && GST_CLOCK_TIME_IS_VALID(pts) && GST_CLOCK_TIME_IS_VALID(self->prev_pts) && pts > self->prev_pts)
    dt = (float)(pts - self->prev_pts) / (float)GST_SECOND;
  self->prev_pts = pts;
  const int n = face.landmarks_count;
  if ((int)self->fx->size() != n) { self->fx->assign(n, OneEuroFilter()); self->fy->assign(n, OneEuroFilter()); }
  std::vector<cv::Point2f> L;
  L.reserve(n);
  for (int i = 0; i < n; ++i) {
    float x = face.landmarks[i].x, y = face.landmarks[i].y;
    if (self->smooth_landmarks) {
      auto& a = (*self->fx)[i];
      auto& b = (*self->fy)[i];
      a.min_cutoff = b.min_cutoff = self->min_cutoff;
      a.beta = b.beta = self->beta;
      x = a.filter(x, dt);
      y = b.filter(y, dt);
    }
    L.emplace_back(x * W, y * H);
  }
  MpApi().face_free_result(&out);

  if (const char* lm_out = std::getenv("LANDMARK_OUTPUT_FILE")) {
    if (FILE* fp = std::fopen(lm_out, "a")) {
      std::fprintf(fp, "Frame %llu Face 0:\n", (unsigned long long)self->frame_count);
      for (const auto& p : L) std::fprintf(fp, "%.6f,%.6f,0.000000\n", p.x / W, p.y / H);
      std::fclose(fp);
    }
  }

  // Current displacement field (face units)
  facewarp::Field field;
  bool any = false;
  g_mutex_lock(&self->lock);
  if (self->basis_dirty) load_basis_locked(self);
  for (const auto& [name, a] : *self->amps) any |= (a != 0.f);
  std::vector<std::string> missing;
  if (any) field = facewarp::combine(*self->basis, *self->amps, &missing);
  for (const auto& m : missing)
    if (self->warned->insert(m).second) GST_WARNING_OBJECT(self, "amplitude %s set but no such field in basis", m.c_str());
  g_mutex_unlock(&self->lock);

  float scale = 1.f;
  if (any && !self->no_warp && n >= 468 && facewarp::FaceFrame(L).unit > 0.f) {
    facewarp::FaceFrame frame(L);
    std::vector<cv::Point2f> disp(field.size());
    for (size_t i = 0; i < field.size(); ++i) disp[i] = frame.to_px(field[i]);
    facewarp::MeshWarper mw(L, W, H);
    if (self->fold_guard >= 0.f) {
      // Folds inside the face only (outline slivers flip invisibly when the
      // head turns). The scale drops at once but recovers by at most
      // kGuardRecovery per frame, so the expression never jumps back.
      const float target = mw.safe_scale(disp, self->fold_guard, true);
      scale = std::min(target, self->guard_scale + kGuardRecovery);
      self->guard_scale = scale;
      if (scale < 1.f) for (auto& d : disp) d *= scale;
    }
    cv::Mat rgba(H, W, CV_8UC4, data, (size_t)stride);
    mw.warp(rgba, disp);
  }
  auto t2 = std::chrono::steady_clock::now();

  if (self->show_landmarks) draw_landmarks(data, W, H, stride, L, self->lm_radius, self->lm_color);

  if (timing) {
    self->sum_detect_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
    self->sum_warp_us += std::chrono::duration<double, std::micro>(t2 - t1).count();
    if (++self->timing_count % self->log_every == 0) {
      const double k = (double)self->log_every * 1000.0;
      GST_INFO_OBJECT(self, "TIMING frame=%llu detect=%.2fms warp=%.2fms (last fold-guard scale %.2f)",
                      (unsigned long long)self->frame_count, self->sum_detect_us / k, self->sum_warp_us / k, scale);
      self->sum_detect_us = self->sum_warp_us = 0;
    }
  }
  return GST_FLOW_OK;
}

// ── Class ────────────────────────────────────────────────────────────────────

static void gst_mozza_mesh_class_init(GstMozzaMeshClass* klass) {
  GST_DEBUG_CATEGORY_INIT(gst_mozza_mesh_debug_category, "mozza_mesh", 0, "Mozza mesh (basis fields, mesh warp)");
  auto* gobject_class = G_OBJECT_CLASS(klass);
  gobject_class->set_property = gst_mozza_mesh_set_property;
  gobject_class->get_property = gst_mozza_mesh_get_property;
  gobject_class->finalize = gst_mozza_mesh_finalize;
  const auto RW = (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS);
  const auto LIVE = (GParamFlags)(RW | GST_PARAM_CONTROLLABLE | GST_PARAM_MUTABLE_PLAYING);

  g_object_class_install_property(gobject_class, PROP_MODEL,
      g_param_spec_string("model", "Model path", "Path to face_landmarker.task", nullptr, RW));
  g_object_class_install_property(gobject_class, PROP_BASIS,
      g_param_spec_string("basis", "Basis files", "Comma-separated basis JSON files (fields in face units)", nullptr, LIVE));
  g_object_class_install_property(gobject_class, PROP_AMPLITUDES,
      g_param_spec_string("amplitudes", "Amplitudes", "All amplitudes at once, e.g. \"AU12=0.5,DOM_o=2\" (unlisted -> 0)", "", LIVE));
  for (int i = 0; i < kNumNamed; ++i)
    g_object_class_install_property(gobject_class, PROP_NAMED_BASE + i,
        g_param_spec_float(kNamed[i].prop, kNamed[i].field, kNamed[i].blurb, -10.f, 10.f, 0.f, LIVE));
  g_object_class_install_property(gobject_class, PROP_FOLD_GUARD,
      g_param_spec_float("fold-guard", "Fold guard", "Max folded area (px^2) inside the face before the deformation is scaled down (recovers over ~0.7 s); negative = off",
                         -1.f, 1e6f, 5.f, LIVE));
  g_object_class_install_property(gobject_class, PROP_SMOOTH_LANDMARKS,
      g_param_spec_boolean("smooth-landmarks", "Smooth landmarks", "OneEuro filter on landmarks", TRUE, LIVE));
  g_object_class_install_property(gobject_class, PROP_MIN_CUTOFF,
      g_param_spec_float("min-cutoff", "Min cutoff", "OneEuro min cutoff (Hz): lower = less jitter, more lag", 0.001f, 100.f, 2.f, LIVE));
  g_object_class_install_property(gobject_class, PROP_BETA,
      g_param_spec_float("beta", "Beta", "OneEuro beta: higher = less lag on fast motion", 0.f, 1.f, 0.05f, LIVE));
  g_object_class_install_property(gobject_class, PROP_SHOW_LANDMARKS,
      g_param_spec_boolean("show-landmarks", "Show landmarks", "Draw the detected landmarks", FALSE, LIVE));
  g_object_class_install_property(gobject_class, PROP_LM_RADIUS,
      g_param_spec_int("landmark-radius", "Landmark radius", "Dot radius", 1, 10, 2, RW));
  g_object_class_install_property(gobject_class, PROP_LM_COLOR,
      g_param_spec_uint("landmark-color", "Landmark color", "Packed RGBA", 0, G_MAXUINT, 0x00FF00FFu, RW));
  g_object_class_install_property(gobject_class, PROP_DROP,
      g_param_spec_boolean("drop", "Drop on no face", "Drop frames without a face", FALSE, RW));
  g_object_class_install_property(gobject_class, PROP_NO_WARP,
      g_param_spec_boolean("no-warp", "No warp", "Detect only, don't deform", FALSE, LIVE));
  g_object_class_install_property(gobject_class, PROP_IGNORE_TS,
      g_param_spec_boolean("ignore-timestamps", "Ignore timestamps", "Use frame count for detector timestamps", FALSE, RW));
  g_object_class_install_property(gobject_class, PROP_NUM_THREADS,
      g_param_spec_int("threads", "Threads", "CPU threads for MediaPipe", 0, 32, 4, RW));
  g_object_class_install_property(gobject_class, PROP_MAX_FACES,
      g_param_spec_int("max-faces", "Max faces", "Faces to detect (only the first is transformed)", 1, 16, 1, RW));
  g_object_class_install_property(gobject_class, PROP_LOG_EVERY,
      g_param_spec_uint("log-every", "Log interval", "Timing log every N frames (GST_DEBUG=mozza_mesh:4)", 0, 1000000, 60, RW));
  g_object_class_install_property(gobject_class, PROP_USER_ID,
      g_param_spec_string("user-id", "User ID", "Accepted for DuckSoup configs; unused", nullptr, RW));

  gst_element_class_set_static_metadata(GST_ELEMENT_CLASS(klass), "Mozza mesh", "Filter/Effect/Video",
      "Face transformations from displacement-field bases (AUs, traits) with a mesh warp", "DuckSoup Lab");
  gst_element_class_add_pad_template(GST_ELEMENT_CLASS(klass), gst_static_pad_template_get(&sink_template));
  gst_element_class_add_pad_template(GST_ELEMENT_CLASS(klass), gst_static_pad_template_get(&src_template));
  GST_BASE_TRANSFORM_CLASS(klass)->start = gst_mozza_mesh_start;
  GST_BASE_TRANSFORM_CLASS(klass)->stop = gst_mozza_mesh_stop;
  GST_VIDEO_FILTER_CLASS(klass)->transform_frame_ip = gst_mozza_mesh_transform_frame_ip;
}

static void gst_mozza_mesh_init(GstMozzaMesh* self) {
  self->model_path = nullptr;
  self->basis_paths = nullptr;
  self->fold_guard = 5.f;
  self->smooth_landmarks = TRUE;
  self->min_cutoff = 2.f;
  self->beta = 0.05f;
  self->show_landmarks = FALSE;
  self->lm_radius = 2;
  self->lm_color = 0x00FF00FFu;
  self->drop = self->no_warp = self->ignore_ts = FALSE;
  self->num_threads = 4;
  self->max_faces = 1;
  self->log_every = 60;
  self->user_id = nullptr;
  g_mutex_init(&self->lock);
  self->amps = new std::map<std::string, float>();
  self->basis_dirty = FALSE;
  self->mp_ctx = nullptr;
  self->basis = new facewarp::Basis();
  self->warned = new std::set<std::string>();
  self->fx = new std::vector<OneEuroFilter>();
  self->fy = new std::vector<OneEuroFilter>();
  self->prev_pts = GST_CLOCK_TIME_NONE;
  self->frame_count = 0;
}

static gboolean plugin_init(GstPlugin* plugin) {
  return gst_element_register(plugin, "mozza_mesh", GST_RANK_NONE, GST_TYPE_MOZZA_MESH);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, mozzamesh,
                  "Face transformations from displacement-field bases with a mesh warp",
                  plugin_init, "1.0", "LGPL", PACKAGE, "https://ducksouplab.com")
