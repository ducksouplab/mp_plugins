// gstmozzamp/deform_utils.cpp
#include "deform_utils.hpp"
#include <algorithm>
#include <cmath>

// --- helpers ---------------------------------------------------------------

static cv::Rect tight_bounds_union(const std::vector<cv::Point2f>& A,
                                   const std::vector<cv::Point2f>& B,
                                   int W, int H, int pad = 16)
{
  if (A.empty() && B.empty()) return cv::Rect();
  float xmin =  1e9f, ymin =  1e9f;
  float xmax = -1e9f, ymax = -1e9f;

  auto accum = [&](const std::vector<cv::Point2f>& P){
    for (const auto& p : P) {
      xmin = std::min(xmin, p.x); ymin = std::min(ymin, p.y);
      xmax = std::max(xmax, p.x); ymax = std::max(ymax, p.y);
    }
  };
  accum(A); accum(B);

  const int x0 = std::max(0, (int)std::floor(xmin) - pad);
  const int y0 = std::max(0, (int)std::floor(ymin) - pad);
  const int x1 = std::min(W-1, (int)std::ceil (xmax) + pad);
  const int y1 = std::min(H-1, (int)std::ceil (ymax) + pad);
  return cv::Rect(x0, y0, std::max(1, x1 - x0 + 1), std::max(1, y1 - y0 + 1));
}

static inline std::vector<cv::Point2f>
to_local(const std::vector<cv::Point2f>& pts, const cv::Rect& r)
{
  std::vector<cv::Point2f> out; out.reserve(pts.size());
  for (auto p : pts) out.emplace_back(p.x - (float)r.x, p.y - (float)r.y);
  return out;
}

// --- dfm → control groups --------------------------------------------------

void build_groups_from_dfm(const Deformations& dfm,
                           const std::vector<cv::Point2f>& L, float alpha,
                           std::vector<std::vector<cv::Point2f>>& srcGroups,
                           std::vector<std::vector<cv::Point2f>>& dstGroups)
{
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
    if (!srcGroups[i].empty()) { s2.push_back(std::move(srcGroups[i])); d2.push_back(std::move(dstGroups[i])); }
  srcGroups.swap(s2); dstGroups.swap(d2);
}

// --- MLS apply on ROI (public API only) ------------------------------------
// --- MLS apply on ROI ------------------------------------------------------

void compute_MLS_on_ROI(cv::Mat& frameRGBA, ImgWarp_MLS_Rigid& mls,
                        const std::vector<cv::Point2f>& src,
                        const std::vector<cv::Point2f>& dst)
{
  if (src.empty()) return;

  // ROI that covers both where points are and where they move to.
  const cv::Rect roi = tight_bounds_union(src, dst, frameRGBA.cols, frameRGBA.rows, /*pad*/18);
  if (roi.width <= 1 || roi.height <= 1) return;

  // Views / conversions (MLS expects 3-channel)
  cv::Mat roi_rgba = frameRGBA(roi);   // CV_8UC4 view into the frame
  cv::Mat roi_bgr;
  cv::cvtColor(roi_rgba, roi_bgr, cv::COLOR_RGBA2BGR); // copy to CV_8UC3

  // Control points in ROI-local coordinates
  const std::vector<cv::Point2f> sL = to_local(src, roi);
  const std::vector<cv::Point2f> dL = to_local(dst, roi);

  // Let the library do its usual thing: calc grid deltas + bilinear densify
  cv::Mat warped_bgr = mls.setAllAndGenerate(roi_bgr, sL, dL,
                                             roi_bgr.cols, roi_bgr.rows,
                                             /*transRatio=*/1.0);
  if (warped_bgr.empty()) return;

  // Copy back into the frame (BGR→RGBA)
  cv::cvtColor(warped_bgr, roi_rgba, cv::COLOR_BGR2RGBA);
}
