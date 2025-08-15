// gstshared/mp_runtime_loader.h
#ifndef MP_RUNTIME_LOADER_H_
#define MP_RUNTIME_LOADER_H_

#include <stdbool.h>
#include "mp_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the loader (idempotent). If path is NULL, it tries:
//   1) getenv("MP_RUNTIME_PATH")
//   2) "libmp_runtime.so" via default search paths
bool mp_runtime_loader_init(const char* optional_so_path);

// Get a reference to the API (after init). Never returns null if init() succeeded.
const MpRuntimeApi* mp_runtime_loader_api(void);

// Most recent error (owned by loader; valid until next init attempt).
const char* mp_runtime_loader_last_error(void);

#ifdef __cplusplus
} // extern "C"
#endif

// ---------- Convenience for C++ plugins ----------
#ifdef __cplusplus
namespace mp_runtime_loader {
  inline bool Init(const char* path = nullptr) { return mp_runtime_loader_init(path); }
  inline const MpRuntimeApi& MpApi()           { return *mp_runtime_loader_api(); }
  inline const char* last_error()              { return mp_runtime_loader_last_error(); }

  // Back-compat shim for older code that called mp_runtime_loader::MpApi::last_error()
  struct MpApi {
    static const char* last_error() { return mp_runtime_loader_last_error(); }
  };
}
// Global helpers used by your plugin code
inline bool MpApiOK()                       { return mp_runtime_loader::Init(nullptr); }
inline const MpRuntimeApi& MpApi()          { return mp_runtime_loader::MpApi(); }
#endif  // __cplusplus

#endif  // MP_RUNTIME_LOADER_H_