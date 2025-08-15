// gstshared/mp_runtime.cc
#include "mp_runtime.h"
#include <string.h>
#include <stdlib.h>

struct MpFaceCtx { int dummy; }; // opaque

static int rt_version() { return 10000; }
static const char* rt_build() { return "mp_runtime stub 1.0"; }

static int rt_face_create(const MpFaceLandmarkerOptions* /*opts*/, MpFaceCtx** out) {
  if (!out) return -1;
  *out = (MpFaceCtx*)malloc(sizeof(MpFaceCtx));
  return *out ? 0 : -2;
}

static int rt_face_detect(MpFaceCtx* /*ctx*/, const MpImage* /*img*/, int64_t ts, MpFaceResult* out) {
  if (!out) return -1;
  out->faces = nullptr;
  out->faces_count = 0;
  out->timestamp_us = ts;
  return 0;
}

static void rt_face_free_result(MpFaceResult* out) {
  if (!out) return;
  // stub allocates nothing; real runtime would free owned buffers here.
}

static void rt_face_close(MpFaceCtx** ctx) {
  if (ctx && *ctx) { free(*ctx); *ctx = nullptr; }
}

static const MpRuntimeApi g_api = {
  /*api_version=*/MP_RUNTIME_API_VERSION,
  /*runtime_version=*/rt_version,
  /*runtime_build=*/rt_build,
  /*face_create=*/rt_face_create,
  /*face_detect=*/rt_face_detect,
  /*face_free_result=*/rt_face_free_result,
  /*face_close=*/rt_face_close,
};

extern "C" const MpRuntimeApi* mp_runtime_get_api(void) {
  return &g_api;
}

// Flat C wrappers (optional)
extern "C" int mp_runtime_version(void) { return rt_version(); }
extern "C" const char* mp_runtime_build(void) { return rt_build(); }
extern "C" int mp_face_landmarker_create(const MpFaceLandmarkerOptions* o, MpFaceCtx** c) { return rt_face_create(o, c); }
extern "C" int mp_face_landmarker_detect(MpFaceCtx* c, const MpImage* i, int64_t t, MpFaceResult* r) { return rt_face_detect(c, i, t, r); }
extern "C" void mp_face_landmarker_free_result(MpFaceResult* r) { rt_face_free_result(r); }
extern "C" void mp_face_landmarker_close(MpFaceCtx** c) { rt_face_close(c); }
extern "C" int face_create(const MpFaceLandmarkerOptions* o, MpFaceCtx** c) { return rt_face_create(o, c); }
extern "C" int face_detect(MpFaceCtx* c, const MpImage* i, int64_t t, MpFaceResult* r) { return rt_face_detect(c, i, t, r); }
extern "C" void face_free_result(MpFaceResult* r) { rt_face_free_result(r); }
extern "C" void face_close(MpFaceCtx** c) { rt_face_close(c); }