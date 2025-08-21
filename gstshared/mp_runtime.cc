// gstshared/mp_runtime.cc
#include "mp_runtime.h"

#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// MediaPipe / Tasks
#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
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

struct MpFaceCtx {
  std::unique_ptr<mp_face::FaceLandmarker> landmarker;
  int num_faces = 1;
  std::mutex mutex;
  std::condition_variable cv;
  absl::StatusOr<mp_face::FaceLandmarkerResult> pending_result;
  bool result_ready = false;
};

// ---------- Version / build ----------
static int rt_version() { return 10002; } // arbitrary
static const char *rt_build() { return "mp_runtime (MediaPipe Tasks) 1.0"; }

// ---------- Helpers ----------
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
    // Expand gray → SRGB
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
    // no blendshapes allocated in this minimal impl
  }
  free((void *)out->faces);
  out->faces = nullptr;
  out->faces_count = 0;
}

// ---------- API impl ----------
static int rt_face_create(const MpFaceLandmarkerOptions *opts,
                          MpFaceCtx **out) {
  if (!out || !opts || !opts->model_path || !*opts->model_path)
    return -1;

  auto ctx = std::make_unique<MpFaceCtx>();

  auto options = std::make_unique<mp_face::FaceLandmarkerOptions>();
  options->base_options = mp_core::BaseOptions();
  options->base_options.model_asset_path = std::string(opts->model_path);
  // Choose execution delegate (CPU/XNNPACK or GPU).
  options->base_options.delegate = mp_core::Delegate::CPU;
  if (opts->delegate) {
    if (std::strcmp(opts->delegate, "gpu") == 0) {
      options->base_options.delegate = mp_core::Delegate::GPU;
    }
  }
  options->running_mode =
      mp_vision::core::RunningMode::LIVE_STREAM; // expects timestamps in ms
  options->result_callback =
      [ctx_ptr = ctx.get()](absl::StatusOr<mp_face::FaceLandmarkerResult> res,
                            const Image &, int64_t) {
        std::lock_guard<std::mutex> lock(ctx_ptr->mutex);
        ctx_ptr->pending_result = std::move(res);
        ctx_ptr->result_ready = true;
        ctx_ptr->cv.notify_one();
      };
  options->num_faces = std::max(1, opts->max_faces);
  options->output_face_blendshapes = (opts->with_blendshapes != 0);
  options->output_facial_transformation_matrixes = (opts->with_geometry != 0);

  absl::StatusOr<std::unique_ptr<mp_face::FaceLandmarker>> lm =
      mp_face::FaceLandmarker::Create(std::move(options));
  if (!lm.ok()) {
    return -2;
  }

  ctx->landmarker = std::move(lm.value());
  ctx->num_faces = std::max(1, opts->max_faces);

  *out = ctx.release();
  return 0;
}

static int rt_face_detect(MpFaceCtx *ctx, const MpImage *img, int64_t ts_us,
                          MpFaceResult *out) {
  if (!ctx || !ctx->landmarker || !out)
    return -1;

  // Initialize output
  out->faces = nullptr;
  out->faces_count = 0;
  out->timestamp_us = (ts_us < 0) ? 0 : ts_us;

  // Convert input to MediaPipe Image
  std::shared_ptr<ImageFrame> frame_ptr = make_imageframe_from_mp(img);
  if (!frame_ptr)
    return -3;

  mediapipe::Image mp_image(frame_ptr);

  // MediaPipe LIVE_STREAM mode expects timestamps in milliseconds.
  const int64_t ts_ms = (ts_us <= 0) ? 0 : (ts_us / 1000);

  {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    ctx->result_ready = false;
  }

  absl::Status status = ctx->landmarker->DetectAsync(mp_image, ts_ms);
  if (!status.ok()) {
    out->faces = nullptr;
    out->faces_count = 0;
    out->timestamp_us = ts_us;
    return 0;
  }

  std::unique_lock<std::mutex> lock(ctx->mutex);
  ctx->cv.wait(lock, [&ctx]() { return ctx->result_ready; });
  auto res_or = std::move(ctx->pending_result);
  ctx->result_ready = false;
  lock.unlock();

  if (!res_or.ok()) {
    out->faces = nullptr;
    out->faces_count = 0;
    out->timestamp_us = ts_us;
    return 0;
  }

  const mp_face::FaceLandmarkerResult &res = res_or.value();

  // Allocate faces array
  const int F = static_cast<int>(res.face_landmarks.size());
  MpFace *faces = nullptr;
  if (F > 0) {
    faces = static_cast<MpFace *>(malloc(sizeof(MpFace) * F));
    if (!faces)
      return -2;
    std::memset(faces, 0, sizeof(MpFace) * F);
  }

  // Copy landmarks for each face
  for (int fi = 0; fi < F; ++fi) {
    // NormalizedLandmarks is a struct with a .landmarks vector
    const auto &lmks_struct = res.face_landmarks[fi];
    const std::vector<mp_tasks::components::containers::NormalizedLandmark>
        &pts_vec = lmks_struct.landmarks;

    const int N = static_cast<int>(pts_vec.size());
    MpLandmark *pts = nullptr;

    if (N > 0) {
      pts = static_cast<MpLandmark *>(malloc(sizeof(MpLandmark) * N));
      if (!pts) {
        // Clean up what we already allocated for earlier faces
        for (int j = 0; j < fi; ++j) {
          free(const_cast<MpLandmark *>(faces[j].landmarks));
        }
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

    // blendshapes/pose remain zero unless you enabled them in options elsewhere
  }

  out->faces = faces;
  out->faces_count = F;
  out->timestamp_us = ts_us;
  return 0;
}

static void rt_face_free_result(MpFaceResult *out) { free_result_owned(out); }

static void rt_face_close(MpFaceCtx **pctx) {
  if (pctx && *pctx) {
    delete *pctx;
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
