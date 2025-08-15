// dfm.hpp
#pragma once
#include <string>
#include <vector>
#include <optional>

struct DfmEntry {
  int   group;        // group id
  int   idx;          // landmark index to move (MediaPipe index)
  int   t0, t1, t2;   // triangle indices
  float a, b, c;      // barycentric weights
};

struct Deformations {
  std::vector<DfmEntry> entries;
};

std::optional<Deformations> load_dfm(const std::string& path);