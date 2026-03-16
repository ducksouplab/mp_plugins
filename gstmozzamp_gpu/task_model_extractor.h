// task_model_extractor.h
// Extracts .tflite models from a MediaPipe .task bundle (ZIP archive).
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct TaskModels {
  std::vector<uint8_t> face_detector;        // BlazeFace short-range detector
  std::vector<uint8_t> face_landmarks;       // 478-landmark regressor
};

// Extract the two TFLite models from a .task file.
// Returns nullopt on failure.
std::optional<TaskModels> extract_task_models(const std::string& task_path);
