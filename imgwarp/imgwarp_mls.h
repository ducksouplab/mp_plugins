#ifndef IMGTRANS_MLS_H
#define IMGTRANS_MLS_H

#include "opencv2/opencv.hpp"
#include <vector>

namespace mp_imgwarp {

using std::vector;
using cv::Mat;
using cv::Mat_;
using cv::Point_;
using cv::Point;

//! The base class for Moving Least Square image warping.
/*!
 * Choose one of the subclasses, the easiest interface to generate
 * an output is to use setAllAndGenerate function.
 */
class ImgWarp_MLS {
public:
    ImgWarp_MLS();
    virtual ~ImgWarp_MLS() {}

    //! Set all and generate an output.
    Mat setAllAndGenerate(const Mat &oriImg,
                          const vector<Point_<int> >   &qsrc,
                          const vector<Point_<int> >   &qdst,
                          const int outW, const int outH,
                          const double transRatio = 1);
    Mat setAllAndGenerate(const Mat &oriImg,
                          const vector<Point_<float> > &qsrc,
                          const vector<Point_<float> > &qdst,
                          const int outW, const int outH,
                          const double transRatio = 1);

    //! Generate the warped image (requires prior setAllAndGenerate()).
    Mat genNewImg(const Mat &oriImg, double transRatio);

    //! Calculate delta fields (implemented by subclasses).
    virtual void calcDelta() = 0;

    //! MLS parameters.
    double alpha;
    int    gridSize;

    //! Set source/target points
    inline void setDstPoints(const vector<Point_<int> >   &qdst);
    inline void setDstPoints(const vector<Point_<float> > &qdst);
    inline void setSrcPoints(const vector<Point_<int> >   &qsrc);
    inline void setSrcPoints(const vector<Point_<float> > &qsrc);

    //! Original and target sizes
    inline void setSize(int w, int h)            { srcW = w; srcH = h; }
    inline void setTargetSize(int outW, int outH){ tarW = outW; tarH = outH; }

    //! Read-only access to displacement fields after calcDelta()
    inline const Mat_<double>& deltaX() const { return rDx; }
    inline const Mat_<double>& deltaY() const { return rDy; }

protected:
    vector<Point_<double> > oldDotL, newDotL; // old = dst, new = src (library naming)
    int nPoint = 0;

    Mat_<double> rDx, rDy;  // displacement fields

    int srcW = 0, srcH = 0;
    int tarW = 0, tarH = 0;
};

// ---- inline definitions ----------------------------------------------------

inline void ImgWarp_MLS::setDstPoints(const vector<Point_<int> > &qdst) {
    nPoint = static_cast<int>(qdst.size());
    oldDotL.clear(); oldDotL.reserve(nPoint);
    for (size_t i = 0; i < qdst.size(); ++i) oldDotL.push_back(qdst[i]);
}

inline void ImgWarp_MLS::setDstPoints(const vector<Point_<float> > &qdst) {
    nPoint = static_cast<int>(qdst.size());
    oldDotL.clear(); oldDotL.reserve(nPoint);
    for (size_t i = 0; i < qdst.size(); ++i) oldDotL.push_back(qdst[i]);
}

inline void ImgWarp_MLS::setSrcPoints(const vector<Point_<int> > &qsrc) {
    nPoint = static_cast<int>(qsrc.size());
    newDotL.clear(); newDotL.reserve(nPoint);
    for (size_t i = 0; i < qsrc.size(); ++i) newDotL.push_back(qsrc[i]);
}

inline void ImgWarp_MLS::setSrcPoints(const vector<Point_<float> > &qsrc) {
    nPoint = static_cast<int>(qsrc.size());
    newDotL.clear(); newDotL.reserve(nPoint);
    for (size_t i = 0; i < qsrc.size(); ++i) newDotL.push_back(qsrc[i]);
}

}  // namespace mp_imgwarp

#endif // IMGTRANS_MLS_H
