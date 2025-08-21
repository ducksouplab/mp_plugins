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
  int32_t stride;      // bytes per row in 'data'
  MpImageFormat format;
} MpImage;

// ---------- Face outputs ----------
typedef struct MpLandmark {
  float x, y, z;       // normalized [0..1] in image space, z in canonical units
} MpLandmark;

typedef struct MpBlendshape {
  const char* category;  // e.g. "mouthSmileLeft"
  float score;           // [0..1]
} MpBlendshape;

typedef struct MpFace {
  const MpLandmark*  landmarks;
  int32_t            landmarks_count;

  const MpBlendshape* blendshapes;
  int32_t             blendshapes_count;

  float   pose_quaternion_wxyz[4]; // optional
  int32_t pose_valid;              // 0/1
} MpFace;

typedef struct MpFaceResult {
  const MpFace* faces;     // owned by runtime; freed by face_free_result()
  int32_t       faces_count;
  int64_t       timestamp_us; // echoed timestamp in microseconds
} MpFaceResult;

// ---------- Options / ctx ----------
typedef struct MpFaceCtx MpFaceCtx;

typedef struct MpFaceLandmarkerOptions {
  const char* model_path;      // .task bundle
  int32_t     max_faces;       // >=1
  int32_t     with_blendshapes;
  int32_t     with_geometry;   // pose matrices, etc.
  int32_t     num_threads;     // hint; may be ignored by backend
  const char* delegate;        // e.g. "gpu", "cpu" (informational)
} MpFaceLandmarkerOptions;

// ---------- Flat C API ----------
// NOTE: mp_face_landmarker_detect() expects ts in MICROSECONDS (μs).
int         mp_runtime_version(void);
const char* mp_runtime_build(void);

int   mp_face_landmarker_create(const MpFaceLandmarkerOptions*, MpFaceCtx**);
int   mp_face_landmarker_detect(MpFaceCtx*, const MpImage*, int64_t timestamp_us, MpFaceResult*);
void  mp_face_landmarker_free_result(MpFaceResult*);
void  mp_face_landmarker_close(MpFaceCtx**);

// Short aliases (some loaders look for these names)
int   face_create(const MpFaceLandmarkerOptions*, MpFaceCtx**);
int   face_detect(MpFaceCtx*, const MpImage*, int64_t timestamp_us, MpFaceResult*);
void  face_free_result(MpFaceResult*);
void  face_close(MpFaceCtx**);

// ---------- Preferred: API table ----------
typedef struct MpRuntimeApi {
  int api_version;
  int         (*runtime_version)(void);
  const char* (*runtime_build)(void);

  int   (*face_create)(const MpFaceLandmarkerOptions*, MpFaceCtx**);
  int   (*face_detect)(MpFaceCtx*, const MpImage*, int64_t timestamp_us, MpFaceResult*);
  void  (*face_free_result)(MpFaceResult*);
  void  (*face_close)(MpFaceCtx**);
} MpRuntimeApi;

// Exported by the runtime shared object:
const MpRuntimeApi* mp_runtime_get_api(void);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // MP_RUNTIME_H_
