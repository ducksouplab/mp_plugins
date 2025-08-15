// gstmozzamp/mp_runtime_loader.h  (use the exact same file in gstfacelandmarks/)
// Header-only, no external deps beyond libc + libdl + your mp_runtime.h
#pragma once

#include <dlfcn.h>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <cstdio>

// Must come from your shared runtime target (gstshared/mp_runtime.h)
#include "mp_runtime.h"

// If your mp_runtime.h defines an API table and advertises it, keep this macro on.
// Otherwise the loader will fall back to resolving flat symbols one by one.
#ifndef MP_RUNTIME_TRY_TABLE_API
#define MP_RUNTIME_TRY_TABLE_API 1
#endif

namespace mp_runtime_loader {

// Small helper to hold a dlopen() handle and resolved function pointers.
class MpApi {
public:
  // --- Function pointer typedefs (match your mp_runtime.h prototypes) ---
  using fn_runtime_version = int (*)(void);
  using fn_runtime_build   = const char* (*)(void);

  using fn_face_create     = int (*)(const MpFaceLandmarkerOptions* opts, MpFaceCtx** out);
  using fn_face_detect     = int (*)(MpFaceCtx* ctx, const MpImage* frame, int64_t ts_us, MpFaceResult* out);
  using fn_face_free_res   = void (*)(MpFaceResult* out);
  using fn_face_close      = void (*)(MpFaceCtx* ctx);

#if MP_RUNTIME_TRY_TABLE_API
  using fn_get_api_table   = const MpRuntimeApi* (*)();  // declared in mp_runtime.h
#endif

  struct Funcs {
    // General
    fn_runtime_version runtime_version = nullptr;
    fn_runtime_build   runtime_build   = nullptr;

    // Face landmarker
    fn_face_create     face_create     = nullptr;
    fn_face_detect     face_detect     = nullptr;
    fn_face_free_res   face_free_res   = nullptr;
    fn_face_close      face_close      = nullptr;

    bool valid() const noexcept {
      return runtime_version && runtime_build &&
             face_create && face_detect && face_free_res && face_close;
    }
  };

  // Get the resolved function table (loads once on first call).
  static const Funcs& get() {
    std::call_once(init_once_, [] { instance_.init(); });
    return instance_.funcs_;
  }

  // True if library is loaded and all required symbols are resolved.
  static bool ok() {
    return get().valid();
  }

  // Optional: unload at process end (normally unnecessary).
  static void unload() {
    std::lock_guard<std::mutex> lock(instance_.mu_);
    if (instance_.handle_) {
      dlclose(instance_.handle_);
      instance_.handle_ = nullptr;
    }
    instance_.funcs_ = Funcs{};
    // do not reset once_flag; further get() calls won’t re-init.
  }

  // For diagnostics
  static const char* last_error() { return instance_.last_error_.c_str(); }

private:
  MpApi() = default;

  void init() {
    // 1) Choose candidate paths
    std::vector<std::string> candidates;
    if (const char* p = std::getenv("MP_RUNTIME_SO"))            candidates.emplace_back(p);
    if (const char* p = std::getenv("DUCKSOUP_MP_RUNTIME_SO"))   candidates.emplace_back(p);
    candidates.emplace_back("libmp_runtime.so");                       // search via ld paths
    candidates.emplace_back("/app/lib/libmp_runtime.so");              // Ducksoup default
    candidates.emplace_back("/usr/local/lib/libmp_runtime.so");
    candidates.emplace_back("/usr/lib/x86_64-linux-gnu/libmp_runtime.so");

    // 2) Try to dlopen one that works
    for (const auto& path : candidates) {
      if (try_open(path.c_str())) {
        // success
        return;
      }
    }

    // 3) Give up: keep funcs_ null, record an error string
    if (last_error_.empty()) {
      last_error_ = "mp_runtime_loader: failed to locate libmp_runtime.so";
    }
  }

  bool try_open(const char* soname) {
    // Clear any stale error state
    (void)dlerror();
    void* h = dlopen(soname, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
      save_dlerr("dlopen", soname);
      return false;
    }

#if MP_RUNTIME_TRY_TABLE_API
    // Preferred: resolve a single API table
    {
      (void)dlerror();
      auto* sym = dlsym(h, "mp_runtime_get_api");
      if (sym) {
        auto get_api = reinterpret_cast<fn_get_api_table>(sym);
        const MpRuntimeApi* api = get_api ? get_api() : nullptr;
        if (api && api->api_version >= MP_RUNTIME_API_MIN_VERSION &&
            api->api_version <= MP_RUNTIME_API_MAX_VERSION) {
          if (fill_from_table(api)) {
            // Keep handle and we’re done.
            std::lock_guard<std::mutex> lock(mu_);
            handle_ = h;
            last_error_.clear();
            return true;
          }
        } else {
          save_err("mp_runtime_get_api: incompatible or null API table");
        }
      }
      // fall through to flat symbols if not present
    }
#endif

    // Fallback: resolve flat symbols one by one
    Funcs f;
    if (!resolve_flat(h, f)) {
      dlclose(h);
      return false;
    }

    std::lock_guard<std::mutex> lock(mu_);
    handle_ = h;
    funcs_  = f;
    last_error_.clear();
    return true;
  }

  bool resolve_flat(void* h, Funcs& out) {
    auto req = [&](auto& fn, const char* name) -> bool {
      (void)dlerror();
      void* s = dlsym(h, name);
      if (!s) { save_dlerr("dlsym", name); return false; }
      fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(s);
      return true;
    };

    Funcs f;
    if (!req(f.runtime_version, "mp_runtime_version"))   return false;
    if (!req(f.runtime_build,   "mp_runtime_build"))     return false;

    if (!req(f.face_create,     "mp_face_landmarker_create")) return false;
    if (!req(f.face_detect,     "mp_face_landmarker_detect")) return false;
    if (!req(f.face_free_res,   "mp_face_landmarker_free_result")) return false;
    if (!req(f.face_close,      "mp_face_landmarker_close"))  return false;

    if (!f.valid()) { save_err("mp_runtime_loader: flat symbol set incomplete"); return false; }
    out = f;
    return true;
  }

#if MP_RUNTIME_TRY_TABLE_API
  bool fill_from_table(const MpRuntimeApi* api) {
    // Map fields from your MpRuntimeApi (from mp_runtime.h) into Funcs.
    // Adjust if your MpRuntimeApi names differ.
    Funcs f;

    // General
    f.runtime_version = api->runtime_version;
    f.runtime_build   = api->runtime_build;

    // Face
    f.face_create     = api->face_create;
    f.face_detect     = api->face_detect;
    f.face_free_res   = api->face_free_result;
    f.face_close      = api->face_close;

    if (!f.valid()) {
      save_err("mp_runtime_loader: API table missing required entries");
      return false;
    }
    std::lock_guard<std::mutex> lock(mu_);
    funcs_ = f;
    return true;
  }
#endif

  void save_err(const char* msg) {
    std::lock_guard<std::mutex> lock(mu_);
    if (last_error_.empty()) last_error_ = msg;
  }

  void save_dlerr(const char* what, const char* name) {
    const char* e = dlerror();
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s(%s): %s", what, name ? name : "?", e ? e : "unknown");
    save_err(buf);
  }

  // --- Data members ---
  static MpApi instance_;
  static std::once_flag init_once_;

  std::mutex mu_;
  void* handle_ = nullptr;
  Funcs funcs_{};
  std::string last_error_;
};

// Static storage
inline MpApi MpApi::instance_{};
inline std::once_flag MpApi::init_once_;

} // namespace mp_runtime_loader

// ---------------------------
// Convenience inline helpers
// ---------------------------
inline const mp_runtime_loader::MpApi::Funcs& MpApi() {
  return mp_runtime_loader::MpApi::get();
}

inline bool MpApiOK() {
  return mp_runtime_loader::MpApi::ok();
}