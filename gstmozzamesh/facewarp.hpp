// facewarp.hpp: displacement-field face warping on the MediaPipe mesh.
//
// A *basis* is a set of named fields; a field gives, for each of the 468
// face-mesh landmarks, a displacement (dx, dy) at amplitude 1, in face units:
// x along the line from the outer eye corner 33 to 263, y perpendicular to it
// (pointing down), 1 unit = the distance between those corners. Fields are
// loaded (with json-glib) from JSON files exported by face-transforms (au_basis_v1.json,
// trait_basis_v1.json: {"meta": {...}, "fields": {"AU12": [[dx, dy], ...]}}).
//
// Each frame: displacement = sum_k amplitude_k * field_k, converted to pixels
// with the face's own eye-corner frame; the image is then warped with a
// piecewise-affine map on the mesh triangles plus two fixed rings around the
// face (pixels beyond the outer ring are untouched). Mirrors facekit/meshwarp.py
// (topology="strip") in face-transforms.
#pragma once

#include <map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace facewarp {

using Field = std::vector<cv::Point2f>;  // 468 displacements, face units

class Basis {
 public:
  // Loads every field of a basis JSON file; fields with the same name as an
  // existing one replace it. Returns false (and sets error) on failure.
  bool load(const std::string& path, std::string* error);
  bool has(const std::string& name) const { return fields_.count(name) > 0; }
  const std::map<std::string, Field>& fields() const { return fields_; }
  void clear() { fields_.clear(); }

 private:
  std::map<std::string, Field> fields_;
};

// Parses "AU12=0.5, DOM_o=2" into a map. Returns false on a malformed entry.
bool parse_amplitudes(const std::string& text, std::map<std::string, float>* out, std::string* error);
std::string format_amplitudes(const std::map<std::string, float>& amps);

// Sum of amplitude * field over the fields present in the basis, in face units.
// Names in `amps` that the basis lacks are appended to `missing` (if given).
Field combine(const Basis& basis, const std::map<std::string, float>& amps,
              std::vector<std::string>* missing = nullptr);

// The face frame of a set of landmarks (pixels).
struct FaceFrame {
  cv::Point2f ex, ey;  // unit vectors: along the eyes, perpendicular (down)
  float unit = 0.f;    // eye span in pixels
  explicit FaceFrame(const std::vector<cv::Point2f>& L);
  cv::Point2f to_px(const cv::Point2f& v) const { return (v.x * ex + v.y * ey) * unit; }
};

class MeshWarper {
 public:
  // Builds the mesh for landmarks L (>= 468, pixels) in an image of size W x H.
  // ring_scales: outline scaled about its centroid, one ring per scale.
  MeshWarper(const std::vector<cv::Point2f>& L, int W, int H,
             std::vector<float> ring_scales = {1.35f, 1.8f});

  // Total area (px^2) of triangles that flip orientation under disp_px
  // (468 per-landmark pixel displacements). ~0 = no fold. With interior_only,
  // triangles touching the face outline or the rings are ignored: when the
  // head turns, those become slivers that flip invisibly under small motions.
  double folded_area(const std::vector<cv::Point2f>& disp_px, bool interior_only = false) const;

  // Largest s in [0, 1] (to ~1%) with folded_area(s * disp, interior_only) <= max_area.
  float safe_scale(const std::vector<cv::Point2f>& disp_px, double max_area, bool interior_only = false) const;

  // Warps img (CV_8UC3 or CV_8UC4) in place. Only the bounding box of the
  // mesh is touched.
  void warp(cv::Mat& img, const std::vector<cv::Point2f>& disp_px) const;

  const std::vector<cv::Point2f>& src() const { return src_; }
  const std::vector<cv::Vec3i>& tris() const { return tris_; }

 private:
  std::vector<cv::Point2f> dst_points(const std::vector<cv::Point2f>& disp_px) const;
  int W_, H_;
  std::vector<cv::Point2f> src_;  // mesh landmarks then ring points
  std::vector<cv::Vec3i> tris_;
  std::vector<bool> interior_;  // per triangle: no vertex on the outline or the rings
};

}  // namespace facewarp
