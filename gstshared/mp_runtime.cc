// gstshared/mp_runtime.cc
#include "mp_runtime.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>

#include <gst/gst.h>

// MediaPipe / Tasks
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker_result.h"

using mediapipe::Image;
using mediapipe::ImageFormat;
using mediapipe::ImageFrame;
namespace mp_core = ::mediapipe::tasks::core;
namespace mp_vision = ::mediapipe::tasks::vision;
namespace mp_face = ::mediapipe::tasks::vision::face_landmarker;
namespace mp_tasks = ::mediapipe::tasks;

GST_DEBUG_CATEGORY_STATIC(mp_runtime_debug);
#define GST_CAT_DEFAULT mp_runtime_debug

static std::mutex g_error_mutex;
static std::string g_last_runtime_error;

static void set_last_error(const std::string& err) {
    std::lock_guard<std::mutex> lock(g_error_mutex);
    g_last_runtime_error = err;
    fprintf(stderr, "[mp_runtime] ERROR: %s\n", err.c_str());
}

static const char* rt_get_last_error() {
    std::lock_guard<std::mutex> lock(g_error_mutex);
    return g_last_runtime_error.c_str();
}

// 1. The NEW, clean, lock-free Struct
struct MpFaceCtx {
  std::unique_ptr<mp_face::FaceLandmarker> landmarker;
  std::shared_ptr<mediapipe::GpuResources> gpu_resources;
  EGLDisplay egl_display = EGL_NO_DISPLAY;
  EGLContext egl_context = EGL_NO_CONTEXT;
  EGLSurface egl_surface = EGL_NO_SURFACE;
  int num_faces = 1;
};

// ---------- Version / build ----------
static int rt_version() { return 10002; } // arbitrary
static const char *rt_build() { return "mp_runtime (MediaPipe Tasks) 1.0"; }

// ---------- Helpers ----------
static void ensure_gst_debug() {
    static gsize initialized = 0;
    if (g_once_init_enter(&initialized)) {
        GST_DEBUG_CATEGORY_INIT(mp_runtime_debug, "mp_runtime", 0, "MediaPipe Runtime");
        g_once_init_leave(&initialized, 1);
    }
}
static std::shared_ptr<ImageFrame> make_imageframe_from_mp(const MpImage *img) {
  if (!img || !img->data || img->width <= 0 || img->height <= 0)
    return nullptr;

  const int W = img->width;
  const int H = img->height;
  const int stride = img->stride;

  if (img->format == MP_IMAGE_RGBA8888) {
    auto frame =
        std::make_shared<ImageFrame>(ImageFormat::SRGBA, W, H, /*alignment*/ 1);
    uint8_t *dst = frame->MutablePixelData();
    const int rowbytes = W * 4;
    for (int y = 0; y < H; ++y) {
      std::memcpy(dst + y * rowbytes, img->data + y * stride, rowbytes);
    }
    return frame;
  } else if (img->format == MP_IMAGE_RGB888) {
    auto frame =
        std::make_shared<ImageFrame>(ImageFormat::SRGB, W, H, /*alignment*/ 1);
    uint8_t *dst = frame->MutablePixelData();
    const int rowbytes = W * 3;
    for (int y = 0; y < H; ++y) {
      std::memcpy(dst + y * rowbytes, img->data + y * stride, rowbytes);
    }
    return frame;
  } else if (img->format == MP_IMAGE_GRAY8) {
    auto frame =
        std::make_shared<ImageFrame>(ImageFormat::SRGB, W, H, /*alignment*/ 1);
    uint8_t *dst = frame->MutablePixelData();
    for (int y = 0; y < H; ++y) {
      const uint8_t *src = img->data + y * stride;
      uint8_t *drow = dst + y * (W * 3);
      for (int x = 0; x < W; ++x) {
        uint8_t v = src[x];
        drow[3 * x + 0] = v;
        drow[3 * x + 1] = v;
        drow[3 * x + 2] = v;
      }
    }
    return frame;
  }

  return nullptr;
}

static void free_result_owned(MpFaceResult *out) {
  if (!out || !out->faces || out->faces_count <= 0)
    return;
  for (int i = 0; i < out->faces_count; ++i) {
    const MpFace *f = &out->faces[i];
    if (f->landmarks && f->landmarks_count > 0) {
      free((void *)f->landmarks);
    }
  }
  free((void *)out->faces);
  out->faces = nullptr;
  out->faces_count = 0;
}

// ---------- API impl ----------
static int rt_face_create(const MpFaceLandmarkerOptions *opts,
                          MpFaceCtx **out) {
  ensure_gst_debug();
  if (!out || !opts || !opts->model_path || !*opts->model_path)
    return -1;

  auto ctx = std::make_unique<MpFaceCtx>();

  auto options = std::make_unique<mp_face::FaceLandmarkerOptions>();
  options->base_options = mp_core::BaseOptions();
  options->base_options.model_asset_path = std::string(opts->model_path);
  using MpDelegate = mp_core::BaseOptions::Delegate;
  options->base_options.delegate = MpDelegate::CPU;

  if (opts->delegate && std::strcmp(opts->delegate, "gpu") == 0) {
    GST_INFO("GPU delegate requested, initializing EGL...");
    options->base_options.delegate = MpDelegate::GPU;

    PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT = (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
    PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");

    if (eglQueryDevicesEXT && eglGetPlatformDisplayEXT) {
        EGLDeviceEXT devices[16];
        EGLint num_devices;
        if (eglQueryDevicesEXT(16, devices, &num_devices) && num_devices > 0) {
            ctx->egl_display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, devices[0], nullptr);
        }
    }

    if (ctx->egl_display == EGL_NO_DISPLAY) {
        ctx->egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    }

    if (ctx->egl_display == EGL_NO_DISPLAY) {
        set_last_error("Failed to get EGL display");
        return -3;
    }

    EGLint major, minor;
    if (!eglInitialize(ctx->egl_display, &major, &minor)) return -3;
    if (!eglBindAPI(EGL_OPENGL_ES_API)) return -3;

    EGLint config_attribs[] = {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 0, EGL_STENCIL_SIZE, 0, EGL_NONE
    };
    EGLConfig config;
    EGLint num_configs;
    if (!eglChooseConfig(ctx->egl_display, config_attribs, &config, 1, &num_configs) || num_configs <= 0) {
        config_attribs[1] = EGL_OPENGL_ES2_BIT;
        if (!eglChooseConfig(ctx->egl_display, config_attribs, &config, 1, &num_configs) || num_configs <= 0) {
            return -3;
        }
    }

    EGLint pbuffer_attribs[] = { EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE };
    ctx->egl_surface = eglCreatePbufferSurface(ctx->egl_display, config, pbuffer_attribs);
    if (ctx->egl_surface == EGL_NO_SURFACE) return -3;

    EGLint context_attribs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
    ctx->egl_context = eglCreateContext(ctx->egl_display, config, EGL_NO_CONTEXT, context_attribs);
    if (ctx->egl_context == EGL_NO_CONTEXT) {
        context_attribs[1] = 2;
        ctx->egl_context = eglCreateContext(ctx->egl_display, config, EGL_NO_CONTEXT, context_attribs);
    }
    if (ctx->egl_context == EGL_NO_CONTEXT) return -3;

    if (!eglMakeCurrent(ctx->egl_display, ctx->egl_surface, ctx->egl_surface, ctx->egl_context)) return -3;

    auto gpu_res_status = mediapipe::GpuResources::Create(ctx->egl_context);
    if (!gpu_res_status.ok()) return -4;
    ctx->gpu_resources = gpu_res_status.value();
  }

  // 2. Use synchronous VIDEO mode. No background threads, no callbacks!
  options->running_mode = mp_vision::core::RunningMode::VIDEO;
  
  options->num_faces = std::max(1, opts->max_faces);
  options->output_face_blendshapes = (opts->with_blendshapes != 0);
  options->output_facial_transformation_matrixes = (opts->with_geometry != 0);

  absl::StatusOr<std::unique_ptr<mp_face::FaceLandmarker>> lm =
      mp_face::FaceLandmarker::Create(std::move(options));

  if (!lm.ok()) {
    std::string err = lm.status().ToString();
    set_last_error("FaceLandmarker::Create failed: " + err);
    GST_FIXME("FaceLandmarker::Create failed: %s", err.c_str());
    return -2;
  }

  ctx->landmarker = std::move(lm.value());
  ctx->num_faces = std::max(1, opts->max_faces);

  *out = ctx.release();
  GST_INFO("FaceLandmarker context created successfully");
  return 0;
}

// 3. The NEW Synchronous rt_face_detect
static int rt_face_detect(MpFaceCtx *ctx, const MpImage *img, int64_t ts_us,
                          MpFaceResult *out) {
  ensure_gst_debug();
  if (!ctx || !ctx->landmarker || !out)
    return -1;

  out->faces = nullptr;
  out->faces_count = 0;
  out->timestamp_us = (ts_us < 0) ? 0 : ts_us;

  std::shared_ptr<ImageFrame> frame_ptr = make_imageframe_from_mp(img);
  if (!frame_ptr) return -3;

  mediapipe::Image mp_image(frame_ptr);
  const int64_t ts_ms = (ts_us <= 0) ? 0 : (ts_us / 1000);

  // Synchronous processing! Fast and lock-free.
  auto res_or = ctx->landmarker->DetectForVideo(mp_image, ts_ms);

  if (!res_or.ok()) {
    GST_ERROR("FaceLandmarker DetectForVideo error: %s", res_or.status().ToString().c_str());
    out->faces = nullptr;
    out->faces_count = 0;
    out->timestamp_us = ts_us;
    return 0;
  }

  const mp_face::FaceLandmarkerResult &res = res_or.value();
  const int F = static_cast<int>(res.face_landmarks.size());
  MpFace *faces = nullptr;
  if (F > 0) {
    faces = static_cast<MpFace *>(malloc(sizeof(MpFace) * F));
    if (!faces) return -2;
    std::memset(faces, 0, sizeof(MpFace) * F);
  }

  for (int fi = 0; fi < F; ++fi) {
    const auto &lmks_struct = res.face_landmarks[fi];
    const std::vector<mp_tasks::components::containers::NormalizedLandmark>
        &pts_vec = lmks_struct.landmarks;

    const int N = static_cast<int>(pts_vec.size());
    MpLandmark *pts = nullptr;

    if (N > 0) {
      pts = static_cast<MpLandmark *>(malloc(sizeof(MpLandmark) * N));
      if (!pts) {
        for (int j = 0; j < fi; ++j) free(const_cast<MpLandmark *>(faces[j].landmarks));
        free(faces);
        return -2;
      }
      for (int i = 0; i < N; ++i) {
        pts[i].x = pts_vec[i].x;
        pts[i].y = pts_vec[i].y;
        pts[i].z = pts_vec[i].z;
      }
    }
    faces[fi].landmarks = pts;
    faces[fi].landmarks_count = N;
  }

  out->faces = faces;
  out->faces_count = F;
  out->timestamp_us = ts_us;
  return 0;
}

static void rt_face_free_result(MpFaceResult *out) { free_result_owned(out); }

// 4. RESTORED rt_face_close
static void rt_face_close(MpFaceCtx **pctx) {
  if (pctx && *pctx) {
    auto ctx = *pctx;
    ctx->landmarker.reset();
    ctx->gpu_resources.reset();
    if (ctx->egl_display != EGL_NO_DISPLAY) {
      eglMakeCurrent(ctx->egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
      if (ctx->egl_surface != EGL_NO_SURFACE) {
        eglDestroySurface(ctx->egl_display, ctx->egl_surface);
      }
      if (ctx->egl_context != EGL_NO_CONTEXT) {
        eglDestroyContext(ctx->egl_display, ctx->egl_context);
      }
      eglTerminate(ctx->egl_display);
    }
    delete ctx;
    *pctx = nullptr;
  }
}

// ---------- API table ----------
static const MpRuntimeApi g_api = {
    /*api_version=*/MP_RUNTIME_API_VERSION,
    /*runtime_version=*/rt_version,
    /*runtime_build=*/rt_build,
    /*face_create=*/rt_face_create,
    /*face_detect=*/rt_face_detect,
    /*face_free_result=*/rt_face_free_result,
    /*face_close=*/rt_face_close,
    /*get_last_error=*/rt_get_last_error,
};

extern "C" const MpRuntimeApi *mp_runtime_get_api(void) { return &g_api; }

// Optional flat C exports (aliases)
extern "C" int mp_runtime_version(void) { return rt_version(); }
extern "C" const char *mp_runtime_build(void) { return rt_build(); }
extern "C" int mp_face_landmarker_create(const MpFaceLandmarkerOptions *o,
                                         MpFaceCtx **c) {
  return rt_face_create(o, c);
}
extern "C" int mp_face_landmarker_detect(MpFaceCtx *c, const MpImage *i,
                                         int64_t t, MpFaceResult *r) {
  return rt_face_detect(c, i, t, r);
}
extern "C" void mp_face_landmarker_free_result(MpFaceResult *r) {
  rt_face_free_result(r);
}
extern "C" void mp_face_landmarker_close(MpFaceCtx **c) { rt_face_close(c); }
extern "C" int face_create(const MpFaceLandmarkerOptions *o, MpFaceCtx **c) {
  return rt_face_create(o, c);
}
extern "C" int face_detect(MpFaceCtx *c, const MpImage *i, int64_t t,
                           MpFaceResult *r) {
  return rt_face_detect(c, i, t, r);
}
extern "C" void face_free_result(MpFaceResult *r) { rt_face_free_result(r); }
extern "C" void face_close(MpFaceCtx **c) { rt_face_close(c); }