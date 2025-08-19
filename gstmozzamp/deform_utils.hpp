// gstmozzamp/deform_utils.hpp
#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include "dfm.hpp"

// Build per-group src/dst point sets according to DFM rules
void build_groups_from_dfm(const Deformations& dfm,
                           const std::vector<cv::Point2f>& L, float alpha,
                           std::vector<std::vector<cv::Point2f>>& srcGroups,
                           std::vector<std::vector<cv::Point2f>>& dstGroups);

