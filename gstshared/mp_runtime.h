// gstshared/mp_runtime.h
#ifndef MP_RUNTIME_H_
#define MP_RUNTIME_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------- Version ----------
#define MP_RUNTIME_API_VERSION     1
#define MP_RUNTIME_API_MIN_VERSION 1
#define MP_RUNTIME_API_MAX_VERSION 1

// ---------- Image ----------
typedef enum MpImageFormat {
  MP_IMAGE_UNKNOWN  = 0,
  MP_IMAGE_RGBA8888 = 1,
  MP_IMAGE_RGB888   = 2,
  MP_IMAGE_GRAY8    = 3,
} MpImageFormat;

typedef struct MpImage {
  const uint8_t* data;
  int32_t width;
  int32_t height;
  int32_t stride;
  MpImageFormat format;
} MpImage;

// ---------- Face outputs ----------
typedef struct MpLandmark {
  float x, y, z;
} MpLandmark;

typedef struct MpBlendshape {
  const char* category;
  float score;
} MpBlendshape;

typedef struct MpFace {
  const MpLandmark*  landmarks;
  int32_t            landmarks_count;

  const MpBlendshape* blendshapes;
  int32_t             blendshapes_count;

  float   pose_quaternion_wxyz[4];
  int32_t pose_valid;
} MpFace;

typedef struct MpFaceResult {
  const MpFace* faces;
  int32_t       faces_count;
  int64_t       timestamp_us;
} MpFaceResult;

// ---------- Options / ctx ----------
typedef struct MpFaceCtx MpFaceCtx;

typedef struct MpFaceLandmarkerOptions {
  const char* model_path;
  int32_t     max_faces;
  int32_t     with_blendshapes;
  int32_t     with_geometry;
  int32_t     num_threads;
  const char* delegate;       // e.g. "xnnpack", "cpu"
} MpFaceLandmarkerOptions;

// ---------- Flat C API (optional exports) ----------
int   mp_runtime_version(void);                       // e.g. 10026
const char* mp_runtime_build(void);                   // string lifetime static

int   mp_face_landmarker_create(const MpFaceLandmarkerOptions*, MpFaceCtx**);
int   mp_face_landmarker_detect(MpFaceCtx*, const MpImage*, int64_t, MpFaceResult*);
void  mp_face_landmarker_free_result(MpFaceResult*);
void  mp_face_landmarker_close(MpFaceCtx**);

// Also export short names (some loaders expect these)
int   face_create(const MpFaceLandmarkerOptions*, MpFaceCtx**);
int   face_detect(MpFaceCtx*, const MpImage*, int64_t, MpFaceResult*);
void  face_free_result(MpFaceResult*);
void  face_close(MpFaceCtx**);

// ---------- Preferred: API table ----------
typedef struct MpRuntimeApi {
  int api_version;
  int         (*runtime_version)(void);
  const char* (*runtime_build)(void);

  int   (*face_create)(const MpFaceLandmarkerOptions*, MpFaceCtx**);
  int   (*face_detect)(MpFaceCtx*, const MpImage*, int64_t, MpFaceResult*);
  void  (*face_free_result)(MpFaceResult*);
  void  (*face_close)(MpFaceCtx**);
} MpRuntimeApi;

// The runtime shared object should export this symbol.
const MpRuntimeApi* mp_runtime_get_api(void);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // MP_RUNTIME_H_