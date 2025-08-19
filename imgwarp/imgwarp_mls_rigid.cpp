#include "imgwarp_mls_rigid.h"
#include <cstdio>
#include <cmath>
#include <limits>

static inline bool IMGWARP_DIAG() {
  static int on = -1;
  if (on == -1) {
    const char* e = std::getenv("IMGWARP_DEBUG");
    on = (e && *e && e[0] != '0') ? 1 : 0;
  }
  return on == 1;
}

ImgWarp_MLS_Rigid::ImgWarp_MLS_Rigid() { preScale = false; }

static double calcArea(const vector<cv::Point_<double> > &V) {
    cv::Point_<double> lt(+1e10, +1e10), rb(-1e10, -1e10);
    for (const auto& p : V) {
        if (p.x < lt.x) lt.x = p.x;
        if (p.y < lt.y) lt.y = p.y;
        if (p.x > rb.x) rb.x = p.x;
        if (p.y > rb.y) rb.y = p.y;
    }
    return std::max(0.0, (rb.x - lt.x) * (rb.y - lt.y));
}


void ImgWarp_MLS_Rigid::calcDelta() {
    int i, j, k;

    cv::Point_<double> swq, qstar, newP, tmpP;
    double sw;

    // Optional pre-scaling to unify scale
    double ratio = 1.0;
    if (preScale) {
        const double a_old = calcArea(oldDotL);
        const double a_new = calcArea(newDotL);
        if (a_old > 1e-12 && a_new > 1e-12) {
            ratio = std::sqrt(a_new / a_old);
            if (std::isfinite(ratio) && ratio > 1e-12) {
                for (i = 0; i < nPoint; ++i) newDotL[i] *= (1.0 / ratio);
            } else {
                ratio = 1.0;
            }
        }
    }

    rDx.create(tarH, tarW);
    rDy.create(tarH, tarW);

    if (nPoint < 2) {
        rDx.setTo(0);
        rDy.setTo(0);
        return;
    }

    cv::Point_<double> swp, pstar, curV, curVJ, Pi, PiJ, Qi;
    double miu_r;
    std::vector<double> w(nPoint);

    for (i = 0;; i += gridSize) {
        if (i >= tarW && i < tarW + gridSize - 1) i = tarW - 1;
        else if (i >= tarW) break;

        for (j = 0;; j += gridSize) {
            if (j >= tarH && j < tarH + gridSize - 1) j = tarH - 1;
            else if (j >= tarH) break;

            sw = 0.0;
            swp.x = swp.y = 0.0;
            swq.x = swq.y = 0.0;
            newP.x = newP.y = 0.0;

            cv::Point_<double> curV_local;
            curV_local.x = i; // x
            curV_local.y = j; // y

            for (k = 0; k < nPoint; ++k) {
                if ((i == oldDotL[k].x) && (j == oldDotL[k].y)) break;
                const double dx = (i - oldDotL[k].x);
                const double dy = (j - oldDotL[k].y);
                const double d2 = dx*dx + dy*dy;
                w[k] = (alpha == 1.0) ? (1.0 / d2) : std::pow(d2, -alpha);
                sw  += w[k];
                swp += w[k] * oldDotL[k];
                swq += w[k] * newDotL[k];
            }

            if (k == nPoint) {
                pstar = (1.0 / sw) * swp;
                qstar = (1.0 / sw) * swq;

                // Calc miu_r
                double s1 = 0.0, s2 = 0.0;
                for (k = 0; k < nPoint; ++k) {
                    if (i == oldDotL[k].x && j == oldDotL[k].y) continue;
                    Pi = oldDotL[k] - pstar;
                    PiJ.x = -Pi.y; PiJ.y = Pi.x;
                    Qi = newDotL[k] - qstar;
                    s1 += w[k] * Qi.dot(Pi);
                    s2 += w[k] * Qi.dot(PiJ);
                }
                miu_r = std::sqrt(s1*s1 + s2*s2);

                // ---- SAFETY GUARD: avoid divide-by-zero / NaN
                if (miu_r < 1e-12 || !std::isfinite(miu_r)) {
                    if (preScale) {
                        rDx(j, i) = qstar.x * ratio - i;
                        rDy(j, i) = qstar.y * ratio - j;
                    } else {
                        rDx(j, i) = qstar.x - i;
                        rDy(j, i) = qstar.y - j;
                    }
                    continue;
                }
                // -------------------------------------------

                curV = curV_local - pstar;
                curVJ.x = -curV.y; curVJ.y = curV.x;

                for (k = 0; k < nPoint; ++k) {
                    if (i == oldDotL[k].x && j == oldDotL[k].y) continue;

                    Pi = oldDotL[k] - pstar;
                    PiJ.x = -Pi.y; PiJ.y = Pi.x;

                    tmpP.x =  Pi.dot(curV)  * newDotL[k].x - PiJ.dot(curV)  * newDotL[k].y;
                    tmpP.y = -Pi.dot(curVJ) * newDotL[k].x + PiJ.dot(curVJ) * newDotL[k].y;
                    tmpP *= (w[k] / miu_r);
                    newP += tmpP;
                }
                newP += qstar;
            } else {
                // Exactly at a control point
                newP = newDotL[k];
            }

            if (preScale) {
                rDx(j, i) = newP.x * ratio - i;
                rDy(j, i) = newP.y * ratio - j;
            } else {
                rDx(j, i) = newP.x - i;
                rDy(j, i) = newP.y - j;
            }
        }
    }

    if (preScale && ratio != 1.0) {
        for (i = 0; i < nPoint; ++i) newDotL[i] *= ratio; // restore
    }
}

