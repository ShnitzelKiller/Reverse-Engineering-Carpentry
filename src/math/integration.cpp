//
// Created by James Noeckel on 1/8/20.
//

#include "integration.h"

using namespace Eigen;

/**
 * Integrate a single pixel under the assumption of bilinear interpolation with the given starting offset in the specified direction, from t=0 to t=dt
 * May compute integral with additional factors for computing the jacobian: for this, the additional arguments
 * t and complement are used, where t is the global value of t at the local starting point, and complement is whether to consider this
 * global t to decrease along direction d.
 * @param img color or grayscale image, or x derivative image if computing derivatives (use sobel or scharr)
 * @param pix
 * @param offset
 * @param d
 * @param dt
 * @param derivative
 * @param t position along global ray, between 0 and 1.
 * @param scale ray length in pixels
 * @param complement
 * @return
 */
VectorXd integrate_pixel(cv::Mat img, const cv::Point &pix, const Array2d &offset, Array2d d, double dt, bool derivative=false, double t=0, double scale = 0, bool complement=false) {
    Array2d xdtydt = offset + 0.5 * dt * d;
    double dt2 = dt * dt;
    double mixterm = offset.x() * d.y() + offset.y() * d.x();
    double xydt = offset.x() * offset.y() + dt2 * d.x() * d.y() / 3.0 +
                  0.5 * dt * mixterm;
    VectorXd a00(img.channels());
    VectorXd a01(img.channels());
    VectorXd a10(img.channels());
    VectorXd a11(img.channels());
    if (derivative) {
        if (img.channels() == 3) {
            cv::Vec3s a00_b = img.at<cv::Vec3s>(pix);
            cv::Vec3s a01_b = img.at<cv::Vec3s>(cv::Point(pix.x + 1, pix.y));
            cv::Vec3s a10_b = img.at<cv::Vec3s>(cv::Point(pix.x, pix.y + 1));
            cv::Vec3s a11_b = img.at<cv::Vec3s>(cv::Point(pix.x + 1, pix.y + 1));
            a00 = Vector3d(a00_b[0], a00_b[1], a00_b[2]);
            a01 = Vector3d(a01_b[0], a01_b[1], a01_b[2]);
            a10 = Vector3d(a10_b[0], a10_b[1], a10_b[2]);
            a11 = Vector3d(a11_b[0], a11_b[1], a11_b[2]);
        } else {
            a00(0) = img.at<short>(pix);
            a01(0) = img.at<short>(cv::Point(pix.x + 1, pix.y));
            a10(0) = img.at<short>(cv::Point(pix.x, pix.y + 1));
            a11(0) = img.at<short>(cv::Point(pix.x + 1, pix.y + 1));
        }
    } else {
        if (img.channels() == 3) {
            cv::Vec3b a00_b = img.at<cv::Vec3b>(pix);
            cv::Vec3b a01_b = img.at<cv::Vec3b>(cv::Point(pix.x+1, pix.y));
            cv::Vec3b a10_b = img.at<cv::Vec3b>(cv::Point(pix.x, pix.y+1));
            cv::Vec3b a11_b = img.at<cv::Vec3b>(cv::Point(pix.x+1, pix.y+1));
            a00 = Vector3d(a00_b[0], a00_b[1], a00_b[2]);
            a01 = Vector3d(a01_b[0], a01_b[1], a01_b[2]);
            a10 = Vector3d(a10_b[0], a10_b[1], a10_b[2]);
            a11 = Vector3d(a11_b[0], a11_b[1], a11_b[2]);
        } else {
            a00(0) = img.at<uchar>(pix);
            a01(0) = img.at<uchar>(cv::Point(pix.x + 1, pix.y));
            a10(0) = img.at<uchar>(cv::Point(pix.x, pix.y + 1));
            a11(0) = img.at<uchar>(cv::Point(pix.x + 1, pix.y + 1));
        }
    }
    VectorXd xterm = a01-a00;
    VectorXd yterm = a10-a00;
    VectorXd xyterm = a11+a00-a01-a10;
    VectorXd constant_factor = dt * (a00 + xterm * xdtydt.x() + yterm * xdtydt.y() + xyterm*xydt);
    if (derivative) {
        Array2d xtdtytdt = 0.5 * dt * offset + dt2 * d / 3.0;
        double xytdt = 0.5 * dt * offset.x() * offset.y() + 0.25 * dt2 * dt * d.x() * d.y() +
                        dt2 * mixterm/3.0;
        VectorXd linear_factor = dt * (0.5 * dt * a00 + xterm * xtdtytdt.x() + yterm * xtdtytdt.y() + xyterm * xytdt) / scale;
        if (complement) {
            return constant_factor * (1-t/scale) - linear_factor;
        }  else {
            return constant_factor * t/scale + linear_factor;
        }
    } else {
        return constant_factor;
    }
}

VectorXd integrate_image(cv::Mat img, const Ref<const Vector2d> &a, const Ref<const Vector2d> &b, bool derivative, const Ref<const Vector2d> &dadt, const Ref<const Vector2d> &dbdt, cv::Mat img_y, bool draw) {
    int chans = img.channels();
    if ((img.depth() != CV_8U && !derivative) || (img.depth() != CV_16S && derivative) ||
        !(chans == 1 || chans==3) ||
            (derivative && img_y.empty())) {
        return VectorXd();
    }
    Array2d d = b - a;
    double dist = d.matrix().norm();
    d/=dist;
    Array2d d_inv = Array2d(d.x() == 0 ? 0 : 1.0/d.x(), d.y() == 0 ? 0 : 1.0/d.y());
    //boundaries
    Array2d halfres = Array2d(img.cols-1, img.rows-1)/2;
    Array2d a_centered = a.array() - halfres;
    Array2d p = a_centered * d_inv;
    Array2d k = halfres * d_inv.abs();
    Array2d ts_enter = -k - p;
    Array2d ts_exit = k - p;
    for (int i=0; i<2; i++) {
        if (d[i] == 0) {
            ts_enter[i] = std::numeric_limits<double>::lowest();
            ts_exit[i] = std::numeric_limits<double>::max();
        }
    }
    double clip_dist = std::min(dist, ts_exit.minCoeff()) - std::max(0.0, ts_enter.maxCoeff());
    if (clip_dist <= 0) return VectorXd(img.channels());
    //DDA
    double intpartx, intparty;
    Array2d a_offset(modf(a.x(), &intpartx), modf(a.y(), &intparty));
    Array2d tr = (0.5 * d.sign() - (a_offset - 0.5)) * d_inv; //exit dists
    if (d.x() == 0) {
        tr.x() = std::numeric_limits<double>::max();
    }
    if (d.y() == 0) {
        tr.y() = std::numeric_limits<double>::max();
    }
    Array2i cellID(static_cast<int>(intpartx), static_cast<int>(intparty));
    Array2i end(static_cast<int>(b.x()), static_cast<int>(b.y()));
    double t = std::min(tr.x(), tr.y());
    double dt = t;
    VectorXd accum = VectorXd::Zero(img.channels());
    VectorXd accumY;
    VectorXd accumComplementX;
    VectorXd accumComplementY;
    if (derivative) {
        //initialize other accumulators
        accumComplementX = VectorXd::Zero(img.channels());
        accumComplementY = VectorXd::Zero(img.channels());
        accumY = VectorXd::Zero(img.channels());
    }
    if (cellID.x() >= 0 && cellID.x() < img.cols-1 && cellID.y() >= 0 && cellID.y() < img.rows-1) {
        accum += integrate_pixel(img, cv::Point(cellID.x(), cellID.y()), a_offset, d, dt, derivative, 0, dist, false);
        if (derivative) {
            accumY += integrate_pixel(img_y, cv::Point(cellID.x(), cellID.y()), a_offset, d, dt, derivative, 0, dist, false);
            accumComplementX += integrate_pixel(img, cv::Point(cellID.x(), cellID.y()), a_offset, d, dt, derivative, 0, dist, true);
            accumComplementY += integrate_pixel(img_y, cv::Point(cellID.x(), cellID.y()), a_offset, d, dt, derivative, 0, dist, true);
        }
    }
    size_t steps = 0;
    while (t < dist) {
        Array2i n = Array2i(tr.y() > tr.x() ? 1 : 0, tr.x() >= tr.y() ? 1 : 0) * d.sign().cast<int>();
        a_offset = Array2d(n.x() == 0 ? a_offset.x() + dt * d.x() : (1-n.x())/2, n.y() == 0 ? a_offset.y() + dt * d.y() : (1-n.y())/2);
        tr += n.cast<double>() * d_inv;
        double newt = std::min(tr.x(), tr.y());
        cellID += n;
        dt = std::min(dist, newt) - t;
        if (cellID.x() < 0 || cellID.y() < 0 || cellID.x() >= img.cols-1 || cellID.y() >= img.rows-1) {
            t = newt;
            continue;
        }
        accum += integrate_pixel(img, cv::Point(cellID.x(), cellID.y()), a_offset, d, dt);
        if (derivative) {
            accumY += integrate_pixel(img_y, cv::Point(cellID.x(), cellID.y()), a_offset, d, dt, derivative, t, dist, false);
            accumComplementX += integrate_pixel(img, cv::Point(cellID.x(), cellID.y()), a_offset, d, dt, derivative, t, dist, true);
            accumComplementY += integrate_pixel(img_y, cv::Point(cellID.x(), cellID.y()), a_offset, d, dt, derivative, t, dist, true);
        }
        t = newt;
        steps++;
        if (draw && !derivative) img.at<cv::Vec3b>(cv::Point(cellID.x(), cellID.y())) = cv::Vec3b(255, 0, 0);
    }
    if (derivative) {
        Eigen::Vector4d dab(dadt.x(), dadt.y(), dbdt.x(), dbdt.y());
        Eigen::Matrix<double, -1, 4> jacobian(img.channels(), 4);
        jacobian << accumComplementX, accumComplementY, accum, accumY;
        return (jacobian * dab) / clip_dist;
    } else {
        return accum/clip_dist;
    }
}