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

// --- per-group ROI MLS -------------------------------------------------------
static cv::Rect tight_bounds(const std::vector<cv::Point2f>& pts,
                             int W, int H, int pad) {
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

static inline void add_identity_anchors_roi(
    int w, int h, std::vector<cv::Point2f>& s, std::vector<cv::Point2f>& d, int inset = 1) {
  const float x0 = (float) inset,           y0 = (float) inset;
  const float x1 = (float)(w - 1 - inset),  y1 = (float)(h - 1 - inset);
  const cv::Point2f c[4] = { {x0,y0}, {x1,y0}, {x1,y1}, {x0,y1} };
  for (int i=0;i<4;++i) { s.push_back(c[i]); d.push_back(c[i]); }
}

void compute_MLS_on_ROI(cv::Mat& img,
                        mp_imgwarp::ImgWarp_MLS_Rigid& mls,
                        const std::vector<cv::Point2f>& srcIn,
                        const std::vector<cv::Point2f>& dstIn,
                        int /*pad_unused*/)              // pad not used in old logic
{
  if (srcIn.empty() || srcIn.size()!=dstIn.size()) return;

  // 1) ROI = union(src,dst), then expand 2×
  cv::Rect rS = cv::boundingRect(srcIn);
  cv::Rect rD = cv::boundingRect(dstIn);
  cv::Rect roi = rS | rD;
  cv::Point c = roi.tl() + cv::Point(roi.width/2, roi.height/2);
  roi = cv::Rect(c.x - roi.width, c.y - roi.height, roi.width*2, roi.height*2);

  // 2) Minimum ROI size: (2*grid+1)^2
  const int g = mls.gridSize;
  cv::Rect roiMin(c.x - g, c.y - g, 2*g + 1, 2*g + 1);
  roi |= roiMin;

  // 3) Clamp to image
  roi &= cv::Rect(0, 0, img.cols, img.rows);
  if (roi.width <= 1 || roi.height <= 1) return;

  // 4) Shift points to ROI-local coords
  std::vector<cv::Point2f> src, dst;
  src.reserve(srcIn.size() + 64);
  dst.reserve(dstIn.size() + 64);
  for (size_t i=0;i<srcIn.size();++i) {
    src.emplace_back(srcIn[i].x - roi.x, srcIn[i].y - roi.y);
    dst.emplace_back(dstIn[i].x - roi.x, dstIn[i].y - roi.y);
  }

  // 5) Identity border ring (step = 2*grid)
  const int step = std::max(2*g, 4);
  for (int x=0; x<roi.width; x+=step) {
    src.push_back({(float)x, 0.f});               dst.push_back(src.back());
    src.push_back({(float)x, (float)roi.height}); dst.push_back(src.back());
  }
  for (int y=step; y<roi.height-step; y+=step) {
    src.push_back({0.f, (float)y});               dst.push_back(src.back());
    src.push_back({(float)roi.width, (float)y});  dst.push_back(src.back());
  }

  // 6) Warp and paste back
  cv::Mat warped = mls.setAllAndGenerate(img(roi), src, dst, roi.width, roi.height);
  if (!warped.empty()) warped.copyTo(img(roi));
}