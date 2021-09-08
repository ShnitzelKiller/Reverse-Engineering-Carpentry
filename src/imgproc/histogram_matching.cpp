//
// Created by James Noeckel on 10/7/20.
//

#include "histogram_matching.h"
#include <vector>

void histogram_matching(const cv::Mat &target, cv::Mat &img, int channel, const cv::Mat &targetMask, const cv::Mat &imgMask) {
    bool allChannels = channel < 0;
    const int loopBound = allChannels ? 3 : 1;
    std::vector<std::vector<unsigned int>> targetHistograms(loopBound, std::vector<unsigned int>(255, 0));
    std::vector<std::vector<unsigned int>> imgHistograms(loopBound, std::vector<unsigned int>(255, 0));
    size_t N_target = 0;
    size_t N_img = 0;
    for (size_t i=0; i<target.total(); ++i) {
        if (targetMask.empty() || targetMask.at<uchar>(i)) {
            const auto &color = target.at<cv::Vec3b>(i);
            for (int c=0; c<loopBound; ++c) {
                ++targetHistograms[c][color[allChannels ? c : channel]];
            }
            ++N_target;
        }
    }
    for (size_t i=0; i<img.total(); ++i) {
        if (imgMask.empty() || imgMask.at<uchar>(i)) {
            const auto &color = img.at<cv::Vec3b>(i);
            for (int c=0; c<loopBound; ++c) {
                ++imgHistograms[c][color[allChannels ? c : channel]];
            }
            ++N_img;
        }
    }
    //convert to CDFs
    for (size_t i=1; i<255; ++i) {
        for (int c=0; c < loopBound; ++c) {
            targetHistograms[c][i] += targetHistograms[c][i - 1];
            imgHistograms[c][i] += imgHistograms[c][i - 1];
        }
    }
    std::vector<std::vector<uchar>> mapping(loopBound, std::vector<uchar>(255));
    std::vector<uchar> targetVals(loopBound, 0);
    //float conversion = static_cast<float>(N_target) / N_img;
    for (size_t i = 0; i < 255; ++i) {
        for (int c=0; c<loopBound; ++c) {
            unsigned int imgCDF = (imgHistograms[c][i] * N_target) / N_img;
            while (targetVals[c] < 255 && targetHistograms[c][targetVals[c]] < imgCDF) ++targetVals[c];
            mapping[c][i] = targetVals[c];
        }
    }
    for (size_t i=0; i<img.total(); ++i) {
        for (int c=0; c<loopBound; ++c) {
            int cc = allChannels ? c : channel;
            img.at<cv::Vec3b>(i)[cc] = mapping[c][img.at<cv::Vec3b>(i)[cc]];
        }
    }
}