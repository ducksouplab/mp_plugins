// task_model_extractor.cpp
// The .task file is a standard ZIP archive containing:
//   - face_detector.tflite  (BlazeFace short-range)
//   - face_landmarks_detector.tflite  (478 landmarks)
// We use minizip (from zlib) to extract them.

#include "task_model_extractor.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <algorithm>

// minizip header (ships with zlib on Debian: libminizip-dev)
#include <minizip/unzip.h>

static std::optional<std::vector<uint8_t>> read_zip_entry(unzFile zf,
                                                           const char* name) {
  if (unzLocateFile(zf, name, /*case_sensitive=*/0) != UNZ_OK) {
    // Try alternate names
    return std::nullopt;
  }

  unz_file_info info;
  if (unzGetCurrentFileInfo(zf, &info, nullptr, 0, nullptr, 0, nullptr, 0) !=
      UNZ_OK) {
    return std::nullopt;
  }

  if (unzOpenCurrentFile(zf) != UNZ_OK) return std::nullopt;

  std::vector<uint8_t> buf(info.uncompressed_size);
  int bytes_read =
      unzReadCurrentFile(zf, buf.data(), static_cast<unsigned>(buf.size()));
  unzCloseCurrentFile(zf);

  if (bytes_read < 0 || static_cast<size_t>(bytes_read) != buf.size()) {
    return std::nullopt;
  }
  return buf;
}

// Try multiple known filenames for each model
static std::optional<std::vector<uint8_t>> find_detector(unzFile zf) {
  static const char* names[] = {
      "face_detector.tflite",
      "FaceDetectorShortRange.tflite",
      "detector.tflite",
      nullptr,
  };
  for (int i = 0; names[i]; ++i) {
    auto r = read_zip_entry(zf, names[i]);
    if (r) return r;
  }
  return std::nullopt;
}

static std::optional<std::vector<uint8_t>> find_landmarks(unzFile zf) {
  static const char* names[] = {
      "face_landmarks_detector.tflite",
      "FaceLandmarksDetector.tflite",
      "landmarks.tflite",
      nullptr,
  };
  for (int i = 0; names[i]; ++i) {
    auto r = read_zip_entry(zf, names[i]);
    if (r) return r;
  }
  return std::nullopt;
}

std::optional<TaskModels> extract_task_models(const std::string& task_path) {
  unzFile zf = unzOpen(task_path.c_str());
  if (!zf) {
    std::fprintf(stderr, "[task_extractor] Cannot open ZIP: %s\n",
                 task_path.c_str());
    return std::nullopt;
  }

  // List all entries for debugging
  if (unzGoToFirstFile(zf) == UNZ_OK) {
    do {
      char fname[256];
      unz_file_info info;
      unzGetCurrentFileInfo(zf, &info, fname, sizeof(fname), nullptr, 0,
                            nullptr, 0);
      std::fprintf(stderr, "[task_extractor] ZIP entry: %s (%lu bytes)\n",
                   fname, (unsigned long)info.uncompressed_size);
    } while (unzGoToNextFile(zf) == UNZ_OK);
  }

  TaskModels models;

  auto det = find_detector(zf);
  if (!det) {
    std::fprintf(stderr,
                 "[task_extractor] face_detector.tflite not found in %s\n",
                 task_path.c_str());
    unzClose(zf);
    return std::nullopt;
  }
  models.face_detector = std::move(*det);

  auto lm = find_landmarks(zf);
  if (!lm) {
    std::fprintf(stderr,
                 "[task_extractor] face_landmarks_detector.tflite not found "
                 "in %s\n",
                 task_path.c_str());
    unzClose(zf);
    return std::nullopt;
  }
  models.face_landmarks = std::move(*lm);

  unzClose(zf);

  std::fprintf(stderr,
               "[task_extractor] Extracted detector=%zu bytes, "
               "landmarks=%zu bytes\n",
               models.face_detector.size(), models.face_landmarks.size());
  return models;
}
