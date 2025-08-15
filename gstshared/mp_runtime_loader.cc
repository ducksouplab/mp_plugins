// gstshared/mp_runtime_loader.cc
#include "mp_runtime_loader.h"

#include <dlfcn.h>
#include <mutex>
#include <string>
#include <cstdlib>

namespace {
std::once_flag        g_once;
void*                 g_handle     = nullptr;
const MpRuntimeApi*   g_api_ptr    = nullptr;
MpRuntimeApi          g_api_fallback{};
std::string           g_last_error;

template <typename T>
static T sym(void* h, const char* name) {
  return reinterpret_cast<T>(dlsym(h, name));
}

static bool init_impl(const char* path) {
  const char* env = std::getenv("MP_RUNTIME_PATH");
  const char* so_path = path && *path ? path : (env && *env ? env : "libmp_runtime.so");

  g_handle = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
  if (!g_handle) {
    g_last_error = std::string("dlopen failed: ") + dlerror();
    return false;
  }

  // Preferred: API table
  using get_api_fn = const MpRuntimeApi* (*)();
  if (auto get_api = sym<get_api_fn>(g_handle, "mp_runtime_get_api")) {
    g_api_ptr = get_api();
    if (!g_api_ptr) {
      g_last_error = "mp_runtime_get_api() returned null";
      return false;
    }
    if (g_api_ptr->api_version < MP_RUNTIME_API_MIN_VERSION ||
        g_api_ptr->api_version > MP_RUNTIME_API_MAX_VERSION) {
      g_last_error = "mp_runtime API version mismatch";
      return false;
    }
    return true;
  }

  // Fallback: look for flat C symbols and build a table
  auto v_fn   = sym<int (*)(void)>(g_handle, "mp_runtime_version");
  auto b_fn   = sym<const char* (*)(void)>(g_handle, "mp_runtime_build");

  auto fc = sym<int (*)(const MpFaceLandmarkerOptions*, MpFaceCtx**)>(g_handle, "mp_face_landmarker_create");
  auto fd = sym<int (*)(MpFaceCtx*, const MpImage*, int64_t, MpFaceResult*)>(g_handle, "mp_face_landmarker_detect");
  auto ff = sym<void (*)(MpFaceResult*)>(g_handle, "mp_face_landmarker_free_result");
  auto fx = sym<void (*)(MpFaceCtx**)>(g_handle, "mp_face_landmarker_close");

  // Also accept short names
  if (!fc) fc = sym<int (*)(const MpFaceLandmarkerOptions*, MpFaceCtx**)>(g_handle, "face_create");
  if (!fd) fd = sym<int (*)(MpFaceCtx*, const MpImage*, int64_t, MpFaceResult*)>(g_handle, "face_detect");
  if (!ff) ff = sym<void (*)(MpFaceResult*)>(g_handle, "face_free_result");
  if (!fx) fx = sym<void (*)(MpFaceCtx**)>(g_handle, "face_close");

  if (!fc || !fd || !ff || !fx) {
    g_last_error = "mp_runtime: neither API table nor flat C symbols were found";
    return false;
  }

  g_api_fallback.api_version   = MP_RUNTIME_API_VERSION;
  g_api_fallback.runtime_version = v_fn ? v_fn : [](){ return 0; };
  g_api_fallback.runtime_build   = b_fn ? b_fn : [](){ return "unknown"; };
  g_api_fallback.face_create     = fc;
  g_api_fallback.face_detect     = fd;
  g_api_fallback.face_free_result= ff;
  g_api_fallback.face_close      = fx;

  g_api_ptr = &g_api_fallback;
  return true;
}

} // namespace

extern "C" bool mp_runtime_loader_init(const char* optional_so_path) {
  std::call_once(g_once, [&](){ init_impl(optional_so_path); });
  return g_api_ptr != nullptr;
}

extern "C" const MpRuntimeApi* mp_runtime_loader_api(void) {
  if (!g_api_ptr) mp_runtime_loader_init(nullptr);
  return g_api_ptr;
}

extern "C" const char* mp_runtime_loader_last_error(void) {
  return g_last_error.c_str();
}