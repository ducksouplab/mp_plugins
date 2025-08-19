// gstmozzamp/deform_utils.cpp
#include "deform_utils.hpp"
#include <algorithm>
#include <cmath>


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

