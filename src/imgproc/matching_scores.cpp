//
// Created by James Noeckel on 10/1/20.
//

#include "matching_scores.h"

#define DIV_EPS 1e-9f

template <typename T>
float norm_cross_correlation_template(const cv::Mat &img1, const cv::Mat &img2, int i1, int j1, int i2, int j2, int radius, bool separate_channels, const cv::Mat &mask1, const cv::Mat &mask2) {
    cv::Vec3f mean1(0, 0, 0);
    cv::Vec3f mean2(0, 0, 0);
    int covered_pixels1 = 0;
    int covered_pixels2 = 0;
    for (int r=-radius; r<=radius; ++r) {
        for (int c=-radius; c<=radius; ++c) {
            int i1o = i1 + r;
            int j1o = j1 + c;
            int i2o = i2 + r;
            int j2o = j2 + c;
            if (i1o >= 0 && j1o >= 0 && i1o < img1.rows && j1o < img1.cols && (mask1.empty() || mask1.at<uchar>(i1o, j1o))) {
                mean1 += img1.at<T>(i1o, j1o);
                ++covered_pixels1;
            }
            if (i2o >= 0 && j2o >= 0 && i2o < img2.rows && j2o < img2.cols && (mask2.empty() || mask2.at<uchar>(i2o, j2o))) {
                mean2 += img2.at<T>(i2o, j2o);
                ++covered_pixels2;
            }
        }
    }
    mean1 /= covered_pixels1;
    mean2 /= covered_pixels2;
    cv::Vec3f norm1(0, 0, 0);
    cv::Vec3f norm2(0, 0, 0);
    cv::Vec3f corr(0, 0, 0);
    for (int r=-radius; r<=radius; ++r) {
        for (int c=-radius; c<=radius; ++c) {
            int i1o = i1 + r;
            int j1o = j1 + c;
            int i2o = i2 + r;
            int j2o = j2 + c;
            if (i1o >= 0 && j1o >= 0 && i1o < img1.rows && j1o < img1.cols && (mask1.empty() || mask1.at<uchar>(i1o, j1o))
                && i2o >= 0 && j2o >= 0 && i2o < img2.rows && j2o < img2.cols && (mask2.empty() || mask2.at<uchar>(i2o, j2o))) {
                cv::Vec3f centered1 = img1.at<T>(i1o, j1o);
                centered1 -= mean1;
                cv::Vec3f centered2 = img2.at<T>(i2o, j2o);
                centered2 -= mean2;
                norm1 += centered1.mul(centered1);
                norm2 += centered2.mul(centered2);
                corr += centered1.mul(centered2);
            }
        }
    }
    if (separate_channels) {
        cv::Vec3f normCorrs;
        normCorrs[0] = corr[0] / std::sqrt(norm1[0] * norm2[0]);
        normCorrs[1] = corr[1] / std::sqrt(norm1[1] * norm2[1]);
        normCorrs[2] = corr[2] / std::sqrt(norm1[2] * norm2[2]);
        return (normCorrs[0] + normCorrs[1] + normCorrs[2]) / 3.0f;
    } else {
        float norm1All = norm1[0] + norm1[1] + norm1[2];
        float norm2All = norm2[0] + norm2[1] + norm2[2];
        float corrAll = corr[0] + corr[1] + corr[2];
//    std::cout << mean1 << std::endl;
//    std::cout << mean2 << std::endl;
//    std::cout << norm1All << std::endl;
//    std::cout << norm2All << std::endl;
//    std::cout << corrAll << std::endl;
        return corrAll / (std::sqrt(norm1All * norm2All) + DIV_EPS);
    }
}

float norm_cross_correlation(const cv::Mat &img1, const cv::Mat &img2, int i1, int j1, int i2, int j2, int radius, bool separate_channels, const cv::Mat &mask1, const cv::Mat &mask2) {
    if (img1.type() == CV_32FC3) {
        return norm_cross_correlation_template<cv::Vec3f>(img1, img2, i1, j1, i2, j2, radius, separate_channels, mask1,
                                                          mask2);
    } else {
        return norm_cross_correlation_template<cv::Vec3b>(img1, img2, i1, j1, i2, j2, radius, separate_channels, mask1,
                                                          mask2);
    }
}

template <typename T>
float ssd_template(const cv::Mat &img1, const cv::Mat &img2, int i1, int j1, int i2, int j2, int radius, const cv::Mat &mask1, const cv::Mat &mask2) {
    cv::Vec3f distanceSquared;
    for (int r=-radius; r<=radius; ++r) {
        for (int c=-radius; c<=radius; ++c) {
            int i1o = i1 + r;
            int j1o = j1 + c;
            int i2o = i2 + r;
            int j2o = j2 + c;
            if (i1o >= 0 && j1o >= 0 && i1o < img1.rows && j1o < img1.cols
                && i2o >= 0 && j2o >= 0 && i2o < img2.rows && j2o < img2.cols && (mask1.empty() || mask1.at<uchar>(i1o, j1o)) && (mask2.empty() || mask2.at<uchar>(i2o, j2o))) {
                cv::Vec3f diff = img1.at<T>(i1o, j1o);
                diff -= img2.at<T>(i2o, j2o);
                distanceSquared += diff.mul(diff);
            }
        }
    }
    return distanceSquared[0] + distanceSquared[1] + distanceSquared[2];
}

float ssd(const cv::Mat &img1, const cv::Mat &img2, int i1, int j1, int i2, int j2, int radius, const cv::Mat &mask1, const cv::Mat &mask2) {
    if (img1.type() == CV_32FC3) {
        return ssd_template<cv::Vec3f>(img1, img2, i1, j1, i2, j2, radius, mask1, mask2);
    } else {
        return ssd_template<cv::Vec3b>(img1, img2, i1, j1, i2, j2, radius, mask1, mask2);
    }
}