// facewarp.cpp: see facewarp.hpp.
#include "facewarp.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include <json-glib/json-glib.h>
#include <opencv2/imgproc.hpp>

#include "face_mesh_data.hpp"

namespace facewarp {

// ── Basis ────────────────────────────────────────────────────────────────────

bool Basis::load(const std::string& path, std::string* error) {
  // json-glib (part of the GStreamer tree of the DuckSoup images)
  GError* gerr = nullptr;
  JsonParser* parser = json_parser_new();
  auto fail = [&](const std::string& msg) {
    if (error) *error = path + ": " + msg;
    if (gerr) g_error_free(gerr);
    g_object_unref(parser);
    return false;
  };
  if (!json_parser_load_from_file(parser, path.c_str(), &gerr)) return fail(gerr ? gerr->message : "cannot read");
  JsonNode* root = json_parser_get_root(parser);
  if (!root || !JSON_NODE_HOLDS_OBJECT(root)) return fail("not a JSON object");
  JsonObject* obj = json_node_get_object(root);
  if (!json_object_has_member(obj, "fields")) return fail("no \"fields\" object");
  JsonNode* fnode = json_object_get_member(obj, "fields");
  if (!JSON_NODE_HOLDS_OBJECT(fnode)) return fail("\"fields\" is not an object");
  JsonObject* fields = json_node_get_object(fnode);
  std::map<std::string, Field> loaded;
  GList* names = json_object_get_members(fields);
  for (GList* it = names; it; it = it->next) {
    const char* name = static_cast<const char*>(it->data);
    JsonNode* node = json_object_get_member(fields, name);
    if (!JSON_NODE_HOLDS_ARRAY(node) || json_array_get_length(json_node_get_array(node)) != (guint)kNumMeshLandmarks) {
      g_list_free(names);
      return fail(std::string("field ") + name + " must be an array of " + std::to_string(kNumMeshLandmarks) + " [dx, dy] pairs");
    }
    JsonArray* rows = json_node_get_array(node);
    Field f(kNumMeshLandmarks);
    for (int i = 0; i < kNumMeshLandmarks; ++i) {
      JsonNode* r = json_array_get_element(rows, i);
      if (!JSON_NODE_HOLDS_ARRAY(r) || json_array_get_length(json_node_get_array(r)) != 2) {
        g_list_free(names);
        return fail(std::string("field ") + name + " row " + std::to_string(i) + " is not [dx, dy]");
      }
      JsonArray* xy = json_node_get_array(r);
      f[i] = cv::Point2f((float)json_array_get_double_element(xy, 0), (float)json_array_get_double_element(xy, 1));
    }
    loaded[name] = std::move(f);
  }
  g_list_free(names);
  g_object_unref(parser);
  for (auto& [name, f] : loaded) fields_[name] = std::move(f);
  return true;
}

static std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\n\r"), b = s.find_last_not_of(" \t\n\r");
  return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

bool parse_amplitudes(const std::string& text, std::map<std::string, float>* out, std::string* error) {
  std::map<std::string, float> amps;
  std::string item;
  std::stringstream ss(text);
  while (std::getline(ss, item, ',')) {
    item = trim(item);
    if (item.empty()) continue;
    size_t eq = item.find('=');
    if (eq == std::string::npos) {
      if (error) *error = "expected name=value, got '" + item + "'";
      return false;
    }
    std::string name = trim(item.substr(0, eq)), value = trim(item.substr(eq + 1));
    char* end = nullptr;
    float v = std::strtof(value.c_str(), &end);
    if (name.empty() || value.empty() || *end != '\0' || !std::isfinite(v)) {
      if (error) *error = "bad amplitude '" + item + "'";
      return false;
    }
    amps[name] = v;
  }
  *out = std::move(amps);
  return true;
}

std::string format_amplitudes(const std::map<std::string, float>& amps) {
  std::ostringstream ss;
  bool first = true;
  for (const auto& [name, v] : amps) {
    if (v == 0.f) continue;
    ss << (first ? "" : ",") << name << "=" << v;
    first = false;
  }
  return ss.str();
}

Field combine(const Basis& basis, const std::map<std::string, float>& amps, std::vector<std::string>* missing) {
  Field out(kNumMeshLandmarks, cv::Point2f(0.f, 0.f));
  for (const auto& [name, a] : amps) {
    if (a == 0.f) continue;
    auto it = basis.fields().find(name);
    if (it == basis.fields().end()) {
      if (missing) missing->push_back(name);
      continue;
    }
    for (int i = 0; i < kNumMeshLandmarks; ++i) out[i] += a * it->second[i];
  }
  return out;
}

// ── Face frame ───────────────────────────────────────────────────────────────

FaceFrame::FaceFrame(const std::vector<cv::Point2f>& L) {
  cv::Point2f d = L[263] - L[33];
  unit = std::sqrt(d.dot(d));
  ex = unit > 0 ? d / unit : cv::Point2f(1.f, 0.f);
  ey = cv::Point2f(-ex.y, ex.x);  // +90 degrees: points down in image coordinates
}

// ── Mesh ─────────────────────────────────────────────────────────────────────

static inline double signed_area(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& c) {
  const double ux = b.x - a.x, uy = b.y - a.y, vx = c.x - a.x, vy = c.y - a.y;
  return ux * vy - uy * vx;
}

MeshWarper::MeshWarper(const std::vector<cv::Point2f>& L, int W, int H, std::vector<float> ring_scales)
    : W_(W), H_(H) {
  src_.assign(L.begin(), L.begin() + kNumMeshLandmarks);
  cv::Point2f c(0.f, 0.f);
  for (int i = 0; i < kNumOval; ++i) c += src_[kFaceOval[i]];
  c *= 1.f / kNumOval;
  for (float s : ring_scales)
    for (int i = 0; i < kNumOval; ++i) {
      cv::Point2f p = c + s * (src_[kFaceOval[i]] - c);
      p.x = std::min(std::max(p.x, 0.f), (float)(W - 1));
      p.y = std::min(std::max(p.y, 0.f), (float)(H - 1));
      src_.push_back(p);
    }

  std::vector<cv::Vec3i> tris;
  for (int t = 0; t < kNumFaceTris; ++t)
    tris.emplace_back(kFaceTris[3 * t], kFaceTris[3 * t + 1], kFaceTris[3 * t + 2]);
  // Strips: outline -> ring 1 -> ring 2 ...
  std::vector<int> inner(kFaceOval, kFaceOval + kNumOval);
  for (size_t k = 0; k < ring_scales.size(); ++k) {
    std::vector<int> outer(kNumOval);
    for (int i = 0; i < kNumOval; ++i) outer[i] = kNumMeshLandmarks + (int)k * kNumOval + i;
    for (int i = 0; i < kNumOval; ++i) {
      int j = (i + 1) % kNumOval;
      tris.emplace_back(inner[i], inner[j], outer[i]);
      tris.emplace_back(inner[j], outer[j], outer[i]);
    }
    inner = outer;
  }
  // Drop degenerate triangles (e.g. along the closed lip seam): they cover no
  // source pixels, and their singular affine maps draw 1px tears.
  std::vector<bool> on_outline(src_.size(), false);
  for (int i = 0; i < kNumOval; ++i) on_outline[kFaceOval[i]] = true;
  for (size_t i = kNumMeshLandmarks; i < src_.size(); ++i) on_outline[i] = true;
  for (const auto& t : tris)
    if (std::abs(signed_area(src_[t[0]], src_[t[1]], src_[t[2]])) > 1.0) {
      tris_.push_back(t);
      interior_.push_back(!on_outline[t[0]] && !on_outline[t[1]] && !on_outline[t[2]]);
    }
}

std::vector<cv::Point2f> MeshWarper::dst_points(const std::vector<cv::Point2f>& disp_px) const {
  std::vector<cv::Point2f> dst = src_;
  for (int i = 0; i < kNumMeshLandmarks; ++i) dst[i] += disp_px[i];
  return dst;
}

double MeshWarper::folded_area(const std::vector<cv::Point2f>& disp_px, bool interior_only) const {
  const auto dst = dst_points(disp_px);
  double total = 0.0;
  for (size_t k = 0; k < tris_.size(); ++k) {
    if (interior_only && !interior_[k]) continue;
    const auto& t = tris_[k];
    double a0 = signed_area(src_[t[0]], src_[t[1]], src_[t[2]]);
    double a1 = signed_area(dst[t[0]], dst[t[1]], dst[t[2]]);
    if ((a0 > 0) != (a1 > 0)) total += std::abs(a1);
  }
  return total;
}

float MeshWarper::safe_scale(const std::vector<cv::Point2f>& disp_px, double max_area, bool interior_only) const {
  if (folded_area(disp_px, interior_only) <= max_area) return 1.f;
  float lo = 0.f, hi = 1.f;
  std::vector<cv::Point2f> d(disp_px.size());
  for (int it = 0; it < 7; ++it) {
    float mid = 0.5f * (lo + hi);
    for (size_t i = 0; i < d.size(); ++i) d[i] = mid * disp_px[i];
    (folded_area(d, interior_only) <= max_area ? lo : hi) = mid;
  }
  return lo;
}

void MeshWarper::warp(cv::Mat& img, const std::vector<cv::Point2f>& disp_px) const {
  const auto dst = dst_points(disp_px);
  // Region touched: bounding box of all mesh points, before and after.
  float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
  for (const auto* pts : {&src_, &dst})
    for (const auto& p : *pts) {
      x0 = std::min(x0, p.x); y0 = std::min(y0, p.y);
      x1 = std::max(x1, p.x); y1 = std::max(y1, p.y);
    }
  cv::Rect roi(cv::Point((int)std::floor(x0) - 1, (int)std::floor(y0) - 1),
               cv::Point((int)std::ceil(x1) + 2, (int)std::ceil(y1) + 2));
  roi &= cv::Rect(0, 0, img.cols, img.rows);
  if (roi.area() <= 0) return;

  // Triangle id per output pixel (-1 = outside the mesh: identity). Same
  // rasterisation as the Python reference (fixed point, 4 fractional bits).
  cv::Mat tid(roi.size(), CV_32S, cv::Scalar(-1));
  std::vector<cv::Matx23d> A(tris_.size());
  for (size_t k = 0; k < tris_.size(); ++k) {
    const auto& t = tris_[k];
    cv::Point2f d3[3] = {dst[t[0]], dst[t[1]], dst[t[2]]};
    cv::Point2f s3[3] = {src_[t[0]], src_[t[1]], src_[t[2]]};
    cv::Mat M = cv::getAffineTransform(d3, s3);  // dst -> src, CV_64F
    A[k] = cv::Matx23d((const double*)M.data);
    cv::Point poly[3];
    for (int v = 0; v < 3; ++v)
      poly[v] = cv::Point((int)std::lround((d3[v].x - roi.x) * 16.0), (int)std::lround((d3[v].y - roi.y) * 16.0));
    cv::fillConvexPoly(tid, poly, 3, cv::Scalar((int)k), cv::LINE_8, 4);
  }

  // Backward map, in ROI coordinates of a copy of the source region.
  cv::Mat mx(roi.size(), CV_32F), my(roi.size(), CV_32F);
  for (int y = 0; y < roi.height; ++y) {
    const int* t = tid.ptr<int>(y);
    float* px = mx.ptr<float>(y);
    float* py = my.ptr<float>(y);
    const double Y = y + roi.y;
    for (int x = 0; x < roi.width; ++x) {
      const double X = x + roi.x;
      if (t[x] < 0) {
        px[x] = (float)x; py[x] = (float)y;
      } else {
        const cv::Matx23d& a = A[t[x]];
        px[x] = (float)(a(0, 0) * X + a(0, 1) * Y + a(0, 2) - roi.x);
        py[x] = (float)(a(1, 0) * X + a(1, 1) * Y + a(1, 2) - roi.y);
      }
    }
  }
  cv::Mat srcroi = img(roi).clone();
  cv::Mat out;
  cv::remap(srcroi, out, mx, my, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
  out.copyTo(img(roi));
}

}  // namespace facewarp
