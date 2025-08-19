#include "imgwarp_mls.h"
#include <cstdio>
#include <cmath>
#include <limits>

using cv::Vec3b;

// Small helper: env flag to toggle diagnostics - Change when debuging is done
static inline bool IMGWARP_DIAG() {
  static int on = 1;
  // Uncomment the following to make this printing accessible through an env var
  //static int on = -1;
  //if (on == -1) {
  //  const char* e = std::getenv("IMGWARP_DEBUG");
  //  on = (e && *e && e[0] != '0') ? 1 : 0;
  // }
  return on == 1;
}

static inline double mean_l1(const cv::Mat& a, const cv::Mat& b) {
  if (a.empty() || b.empty() || a.size() != b.size() || a.type() != b.type()) return -1.0;
  cv::Mat diff; cv::absdiff(a, b, diff);
  cv::Scalar s = cv::sum(diff);
  const double denom = static_cast<double>(a.total()) * a.channels();
  return (s[0] + s[1] + s[2] + (a.channels() == 4 ? s[3] : 0.0)) / denom;
}

static inline void point_stats(const std::vector<cv::Point_<double>>& oldDotL,
                               const std::vector<cv::Point_<double>>& newDotL,
                               double& max_disp, double& mean_disp) {
  max_disp = 0.0; mean_disp = 0.0;
  const int n = std::min(oldDotL.size(), newDotL.size());
  for (int i = 0; i < n; ++i) {
    const double dx = newDotL[i].x - oldDotL[i].x;
    const double dy = newDotL[i].y - oldDotL[i].y;
    const double d  = std::sqrt(dx*dx + dy*dy);
    max_disp = std::max(max_disp, d);
    mean_disp += d;
  }
  if (n > 0) mean_disp /= n;
}

ImgWarp_MLS::ImgWarp_MLS() { gridSize = 5; }

inline double bilinear_interp(double x, double y, double v11, double v12,
                              double v21, double v22) {
    return (v11 * (1 - y) + v12 * y) * (1 - x) + (v21 * (1 - y) + v22 * y) * x;
}

Mat ImgWarp_MLS::setAllAndGenerate(const Mat &oriImg,
                                   const vector<Point_<int> > &qsrc,
                                   const vector<Point_<int> > &qdst,
                                   const int outW, const int outH,
                                   const double transRatio) {
    setSize(oriImg.cols, oriImg.rows);
    setTargetSize(outW, outH);
    setSrcPoints(qsrc);
    setDstPoints(qdst);

    if (IMGWARP_DIAG()) {
      std::fprintf(stderr,
        "[imgwarp][setAll/int] in=%dx%d c=%d out=%dx%d n=%d grid=%d alpha=%.3f\n",
        oriImg.cols, oriImg.rows, oriImg.channels(), outW, outH,
        (int)qsrc.size(), gridSize, alpha);
      if ((int)qsrc.size() != (int)qdst.size()) {
        std::fprintf(stderr, "[imgwarp][setAll/int] WARNING: qsrc.size()=%zu != qdst.size()=%zu\n",
                     qsrc.size(), qdst.size());
      }
      double md = 0, mean = 0;
      point_stats(/*old*/oldDotL, /*new*/newDotL, md, mean);
      std::fprintf(stderr, "[imgwarp][setAll/int] ctrl max|new-old|=%.2f mean=%.2f preScale=%s\n",
                   md, mean, "N/A (rigid calcDelta decides)");
    }

    calcDelta();
    Mat out = genNewImg(oriImg, transRatio);

    if (IMGWARP_DIAG()) {
      double d = mean_l1(oriImg, out);
      double minDx=0, maxDx=0, minDy=0, maxDy=0;
      if (!rDx.empty()) cv::minMaxLoc(rDx, &minDx, &maxDx);
      if (!rDy.empty()) cv::minMaxLoc(rDy, &minDy, &maxDy);
      std::fprintf(stderr,
        "[imgwarp][setAll/int] rDx[min,max]=[%.3f,%.3f] rDy[min,max]=[%.3f,%.3f] meanΔ=%.3f\n",
        minDx, maxDx, minDy, maxDy, d);
    }
    return out;
}

Mat ImgWarp_MLS::setAllAndGenerate(const Mat &oriImg,
                                   const vector<Point_<float> > &qsrc,
                                   const vector<Point_<float> > &qdst,
                                   const int outW, const int outH,
                                   const double transRatio) {
    setSize(oriImg.cols, oriImg.rows);
    setTargetSize(outW, outH);
    setSrcPoints(qsrc);
    setDstPoints(qdst);

    if (IMGWARP_DIAG()) {
      std::fprintf(stderr,
        "[imgwarp][setAll/float] in=%dx%d c=%d out=%dx%d n=%d grid=%d alpha=%.3f\n",
        oriImg.cols, oriImg.rows, oriImg.channels(), outW, outH,
        (int)qsrc.size(), gridSize, alpha);
      if ((int)qsrc.size() != (int)qdst.size()) {
        std::fprintf(stderr, "[imgwarp][setAll/float] WARNING: qsrc.size()=%zu != qdst.size()=%zu\n",
                     qsrc.size(), qdst.size());
      }
      double md = 0, mean = 0;
      point_stats(/*old*/oldDotL, /*new*/newDotL, md, mean);
      std::fprintf(stderr, "[imgwarp][setAll/float] ctrl max|new-old|=%.2f mean=%.2f preScale=%s\n",
                   md, mean, "N/A (rigid calcDelta decides)");
    }

    calcDelta();
    Mat out = genNewImg(oriImg, transRatio);

    if (IMGWARP_DIAG()) {
      double d = mean_l1(oriImg, out);
      double minDx=0, maxDx=0, minDy=0, maxDy=0;
      if (!rDx.empty()) cv::minMaxLoc(rDx, &minDx, &maxDx);
      if (!rDy.empty()) cv::minMaxLoc(rDy, &minDy, &maxDy);
      std::fprintf(stderr,
        "[imgwarp][setAll/float] rDx[min,max]=[%.3f,%.3f] rDy[min,max]=[%.3f,%.3f] meanΔ=%.3f\n",
        minDx, maxDx, minDy, maxDy, d);
    }
    return out;
}

Mat ImgWarp_MLS::genNewImg(const Mat &oriImg, double transRatio) {
    int i, j;
    double di, dj;
    double nx, ny;
    int nxi, nyi, nxi1, nyi1;
    double deltaX, deltaY;
    double w, h;
    int ni, nj;

    Mat newImg(tarH, tarW, oriImg.type());
    for (i = 0; i < tarH; i += gridSize)
        for (j = 0; j < tarW; j += gridSize) {
            ni = i + gridSize, nj = j + gridSize;
            w = h = gridSize;
            if (ni >= tarH) ni = tarH - 1, h = ni - i + 1;
            if (nj >= tarW) nj = tarW - 1, w = nj - j + 1;
            for (di = 0; di < h; di++)
                for (dj = 0; dj < w; dj++) {
                    deltaX =
                        bilinear_interp(di / h, dj / w, rDx(i, j), rDx(i, nj),
                                        rDx(ni, j), rDx(ni, nj));
                    deltaY =
                        bilinear_interp(di / h, dj / w, rDy(i, j), rDy(i, nj),
                                        rDy(ni, j), rDy(ni, nj));
                    nx = j + dj + deltaX * transRatio;
                    ny = i + di + deltaY * transRatio;
                    if (nx > srcW - 1) nx = srcW - 1;
                    if (ny > srcH - 1) ny = srcH - 1;
                    if (nx < 0) nx = 0;
                    if (ny < 0) ny = 0;
                    nxi = int(nx);
                    nyi = int(ny);
                    nxi1 = std::ceil(nx);
                    nyi1 = std::ceil(ny);

                    if (oriImg.channels() == 1)
                        newImg.at<uchar>(i + di, j + dj) = bilinear_interp(
                            ny - nyi, nx - nxi, oriImg.at<uchar>(nyi, nxi),
                            oriImg.at<uchar>(nyi, nxi1),
                            oriImg.at<uchar>(nyi1, nxi),
                            oriImg.at<uchar>(nyi1, nxi1));
                    else {
                        // NOTE: library only handles 3-channel in this branch; callers should pass BGR.
                        for (int ll = 0; ll < 3; ll++)
                            newImg.at<Vec3b>(i + di, j + dj)[ll] =
                                bilinear_interp(
                                    ny - nyi, nx - nxi,
                                    oriImg.at<Vec3b>(nyi, nxi)[ll],
                                    oriImg.at<Vec3b>(nyi, nxi1)[ll],
                                    oriImg.at<Vec3b>(nyi1, nxi)[ll],
                                    oriImg.at<Vec3b>(nyi1, nxi1)[ll]);
                    }
                }
        }
    return newImg;
}
