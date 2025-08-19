// gstmozzamp/deform_utils.cpp
#include "deform_utils.hpp"
#include <algorithm>
#include <cmath>

// Compute a padded bounding rectangle for a set of points, clamped to the
// image dimensions.
static cv::Rect tight_bounds(const std::vector<cv::Point2f>& pts,
                             int W, int H, int pad = 16) {
  if (pts.empty()) return cv::Rect();
  float xmin =  1e9f, ymin =  1e9f;
  float xmax = -1e9f, ymax = -1e9f;
  for (const auto& p : pts) {
    xmin = std::min(xmin, p.x); ymin = std::min(ymin, p.y);
    xmax = std::max(xmax, p.x); ymax = std::max(ymax, p.y);
  }
  int x = std::max(0, (int)std::floor(xmin) - pad);
  int y = std::max(0, (int)std::floor(ymin) - pad);
  int X = std::min(W - 1, (int)std::ceil (xmax) + pad);
  int Y = std::min(H - 1, (int)std::ceil (ymax) + pad);
  return cv::Rect(x, y, std::max(1, X - x + 1), std::max(1, Y - y + 1));
}

// --- dfm → control groups --------------------------------------------------

void build_groups_from_dfm(const Deformations& dfm,
                           const std::vector<cv::Point2f>& L, float alpha,
                           std::vector<std::vector<cv::Point2f>>& srcGroups,
                           std::vector<std::vector<cv::Point2f>>& dstGroups,
                           std::vector<cv::Rect>& bounds,
                           int W, int H) {
  int gmax = -1; for (auto& e : dfm.entries) gmax = std::max(gmax, e.group);
  srcGroups.assign(gmax + 1, {}); dstGroups.assign(gmax + 1, {});
  auto safe = [&](int i)->cv::Point2f {
    if (i < 0 || i >= (int)L.size()) return cv::Point2f(0,0);
    return L[i];
  };

  for (auto& e : dfm.entries) {
    if (e.idx < 0 || e.idx >= (int)L.size()) continue;
    const cv::Point2f cur = safe(e.idx);
    const cv::Point2f T   = e.a * safe(e.t0) + e.b * safe(e.t1) + e.c * safe(e.t2);
    const cv::Point2f dst = cur + alpha * (T - cur);
    srcGroups[e.group].push_back(cur);
    dstGroups[e.group].push_back(dst);
  }

  // compact (remove empty groups)
  std::vector<std::vector<cv::Point2f>> s2, d2;
  for (size_t i = 0; i < srcGroups.size(); ++i)
    if (!srcGroups[i].empty()) {
      s2.push_back(std::move(srcGroups[i]));
      d2.push_back(std::move(dstGroups[i]));
    }
  srcGroups.swap(s2); dstGroups.swap(d2);

  // compute per-group bounding rectangles
  bounds.clear();
  bounds.reserve(srcGroups.size());
  for (size_t i = 0; i < srcGroups.size(); ++i) {
    bounds.push_back(tight_bounds(srcGroups[i], W, H, /*pad=*/18));
  }
}

// --- MLS warp on ROI ------------------------------------------------------

void compute_MLS_on_ROI(cv::Mat& imgRGBA, ImgWarp_MLS_Rigid& mls,
                        const std::vector<cv::Point2f>& src,
                        const std::vector<cv::Point2f>& dst,
                        const cv::Rect& roi) {
  if (src.empty() || dst.empty()) return;
  cv::Rect r = roi & cv::Rect(0, 0, imgRGBA.cols, imgRGBA.rows);
  if (r.width <= 1 || r.height <= 1) return;

  // Extract ROI and convert to BGR for the MLS library.
  cv::Mat patch_rgba = imgRGBA(r).clone();
  cv::Mat patch_bgr; cv::cvtColor(patch_rgba, patch_bgr, cv::COLOR_RGBA2BGR);

  // shift control points into ROI-local coordinates
  std::vector<cv::Point2f> sL, dL;
  sL.reserve(src.size()); dL.reserve(dst.size());
  for (size_t i = 0; i < src.size(); ++i) {
    sL.emplace_back(src[i].x - r.x, src[i].y - r.y);
    dL.emplace_back(dst[i].x - r.x, dst[i].y - r.y);
  }

  // Identity anchors should already be present in src/dst. Generate warp.
  cv::Mat warped = mls.setAllAndGenerate(patch_bgr, sL, dL,
                                         patch_bgr.cols, patch_bgr.rows);
  if (!warped.empty()) warped.copyTo(patch_bgr);

  // Convert back to RGBA and copy into the source image.
  cv::cvtColor(patch_bgr, patch_rgba, cv::COLOR_BGR2RGBA);
  patch_rgba.copyTo(imgRGBA(r));
}

