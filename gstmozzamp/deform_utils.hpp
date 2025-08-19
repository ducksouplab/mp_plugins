// gstmozzamp/deform_utils.hpp
#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include "dfm.hpp"
#include "imgwarp/imgwarp_mls_rigid.h"

// Build per-group src/dst point sets according to DFM rules and output tight
// bounding boxes for each group. The returned rectangles are clamped to the
// image bounds (W x H).
void build_groups_from_dfm(const Deformations& dfm,
                           const std::vector<cv::Point2f>& L, float alpha,
                           std::vector<std::vector<cv::Point2f>>& srcGroups,
                           std::vector<std::vector<cv::Point2f>>& dstGroups,
                           std::vector<cv::Rect>& bounds,
                           int W, int H);

// Apply MLS on a specific ROI (in-place on RGBA frame).
void compute_MLS_on_ROI(cv::Mat& imgRGBA, ImgWarp_MLS_Rigid& mls,
                        const std::vector<cv::Point2f>& src,
                        const std::vector<cv::Point2f>& dst,
                        const cv::Rect& roi);

