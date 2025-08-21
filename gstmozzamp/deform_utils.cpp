// gstmozzamp/deform_utils.cpp
#include "deform_utils.hpp"
#include <algorithm>
#include <cmath>


// --- dfm → control groups --------------------------------------------------

static inline bool valid_idx(int i, int n) { return i >= 0 && i < n; }

void build_groups_from_dfm(const Deformations& dfm,
                           const std::vector<cv::Point2f>& L, float alpha,
                           std::vector<std::vector<cv::Point2f>>& srcGroups,
                           std::vector<std::vector<cv::Point2f>>& dstGroups)
{
  int gmax = -1; for (auto& e : dfm.entries) gmax = std::max(gmax, e.group);
  srcGroups.assign(gmax + 1, {}); dstGroups.assign(gmax + 1, {});

  const int N = (int)L.size();

  for (auto& e : dfm.entries) {
    // must have a valid control point AND all three anchors
    if (!valid_idx(e.idx, N) ||
        !valid_idx(e.t0, N) || !valid_idx(e.t1, N) || !valid_idx(e.t2, N)) {
      // optional: log once per frame or behind a flag
      // GST_WARNING_OBJECT(self, "DFM: skipping row (idx=%d,t0=%d,t1=%d,t2=%d) for N=%d",
      //                    e.idx, e.t0, e.t1, e.t2, N);
      continue;
    }

    const cv::Point2f cur = L[e.idx];
    const cv::Point2f T   = e.a * L[e.t0] + e.b * L[e.t1] + e.c * L[e.t2];

    const cv::Point2f dst = cur + alpha * (T - cur);
    srcGroups[e.group].push_back(cur);
    dstGroups[e.group].push_back(dst);
  }

  // compact groups
  std::vector<std::vector<cv::Point2f>> s2, d2;
  for (auto& g : srcGroups) if (!g.empty()) s2.emplace_back(std::move(g));
  for (auto& g : dstGroups) if (!g.empty()) d2.emplace_back(std::move(g));
  srcGroups.swap(s2); dstGroups.swap(d2);
}

// --- per-group ROI MLS -------------------------------------------------------

static cv::Rect tight_bounds(const std::vector<cv::Point2f>& pts,
                             int W, int H, int pad) {
  if (pts.empty()) return cv::Rect();
  float xmin =  1e9f, ymin =  1e9f;
  float xmax = -1e9f, ymax = -1e9f;
  for (auto& p : pts) {
    xmin = std::min(xmin, p.x); ymin = std::min(ymin, p.y);
    xmax = std::max(xmax, p.x); ymax = std::max(ymax, p.y);
  }
  int x = std::max(0, (int)std::floor(xmin) - pad);
  int y = std::max(0, (int)std::floor(ymin) - pad);
  int X = std::min(W - 1, (int)std::ceil(xmax) + pad);
  int Y = std::min(H - 1, (int)std::ceil(ymax) + pad);
  return cv::Rect(x, y, std::max(1, X - x + 1), std::max(1, Y - y + 1));
}

void compute_MLS_on_ROI(cv::Mat& imgRGBA, mp_imgwarp::ImgWarp_MLS_Rigid& mls,
                        const std::vector<cv::Point2f>& src,
                        const std::vector<cv::Point2f>& dst,
                        int pad) {
  if (src.empty()) return;
  const cv::Rect roi = tight_bounds(src, imgRGBA.cols, imgRGBA.rows, pad);
  if (roi.width <= 1 || roi.height <= 1) return;

  cv::Mat patch = imgRGBA(roi).clone();

  std::vector<cv::Point2f> sL, dL;
  sL.reserve(src.size());
  dL.reserve(dst.size());
  for (size_t i = 0; i < src.size(); ++i) {
    sL.emplace_back(src[i].x - roi.x, src[i].y - roi.y);
    dL.emplace_back(dst[i].x - roi.x, dst[i].y - roi.y);
  }

  mls.setAllAndGenerate(patch, sL, dL, patch.cols, patch.rows);
  patch.copyTo(imgRGBA(roi));
}