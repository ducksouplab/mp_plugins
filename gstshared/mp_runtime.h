// gstshared/mp_runtime.h
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// --------------------
// Versioning / feature
// --------------------
#define MP_RUNTIME_API_VERSION       1
#define MP_RUNTIME_API_MIN_VERSION   1
#define MP_RUNTIME_API_MAX_VERSION   1

// --------------------
// Basic image wrapper
// --------------------
typedef enum MpImageFormat {
  MP_IMAGE_UNKNOWN    = 0,
  MP_IMAGE_RGBA8888   = 1,   // RGBA, 8-bit per channel
  MP_IMAGE_RGB888     = 2,   // RGB,  8-bit per channel
  MP_IMAGE_GRAY8      = 3,   // single channel
} MpImageFormat;

typedef struct MpImage {
  const uint8_t* data;   // pointer to top-left pixel
  int32_t        width;  // pixels
  int32_t        height; // pixels
  int32_t        stride; // bytes per row
  MpImageFormat  format; // see enum
} MpImage;

// -------------
// Face structs
// -------------
typedef struct MpLandmark {
  float x;  // normalized [0,1] or pixel — runtime will document what it returns
  float y;  // normalized [0,1] or pixel
  float z;  // model-dependent depth; may be 0 if not provided
} MpLandmark;

typedef struct MpBlendshape {
  const char* category;  // e.g. "mouthSmileLeft"
  float       score;     // 0..1
} MpBlendshape;

typedef struct MpFace {
  // Required: landmarks (e.g. 468 for the canonical model)
  const MpLandmark* landmarks;
  int32_t           landmarks_count;

  // Optional: blendshapes (0 if not requested)
  const MpBlendshape* blendshapes;
  int32_t             blendshapes_count;

  // Optional: head pose / geometry (filled only if enabled)
  float pose_quaternion_wxyz[4];  // {w,x,y,z}; undefined if not computed
  int32_t pose_valid;             // 1 if pose fields are valid, else 0
} MpFace;

typedef struct MpFaceResult {
  const MpFace* faces;   // owned by runtime; freed via mp_face_landmarker_free_result()
  int32_t       faces_count;
  int64_t       timestamp_us; // echo of inference timestamp
} MpFaceResult;

// ---------------------
// Creation / run options
// ---------------------
typedef struct MpFaceLandmarkerOptions {
  const char* model_path;     // nullptr = use runtime default
  int32_t     max_faces;      // e.g. 1..4
  int32_t     with_blendshapes; // boolean 0/1
  int32_t     with_geometry;    // boolean 0/1 (pose/mesh)
  int32_t     num_threads;      // 0 = runtime default
  const char* delegate;         // "xnnpack" (default), "gpu", etc., if supported
} MpFaceLandmarkerOptions;

// Opaque context owned by the runtime
typedef struct MpFaceCtx MpFaceCtx;

// -------------------------------------
// Flat C symbols (fallback or direct use)
// -------------------------------------
int           mp_runtime_version(void);            // e.g. 10026 for v0.10.26
const char*   mp_runtime_build(void);              // human string, never free()

int           mp_face_landmarker_create(
                const MpFaceLandmarkerOptions* opts,
                MpFaceCtx** out_ctx);

int           mp_face_landmarker_detect(
                MpFaceCtx* ctx,
                const MpImage* frame,
                int64_t timestamp_us,
                MpFaceResult* out);

void          mp_face_landmarker_free_result(
                MpFaceResult* out); // frees any memory referenced by 'out'

void          mp_face_landmarker_close(
                MpFaceCtx** ctx);   // sets *ctx = NULL

// -------------------------------------
// Preferred: single API table export
// -------------------------------------
typedef struct MpRuntimeApi {
  int api_version; // must be within [MP_RUNTIME_API_MIN_VERSION, MP_RUNTIME_API_MAX_VERSION]

  // General
  int           (*runtime_version)(void);
  const char*   (*runtime_build)(void);

  // Face landmarker
  int  (*face_create)(const MpFaceLandmarkerOptions*, MpFaceCtx**);
  int  (*face_detect)(MpFaceCtx*, const MpImage*, int64_t, MpFaceResult*);
  void (*face_free_result)(MpFaceResult*);
  void (*face_close)(MpFaceCtx**);
} MpRuntimeApi;

// Implemented by the runtime shared library
const MpRuntimeApi* mp_runtime_get_api(void);

#ifdef __cplusplus
} // extern "C"
#endif