//
// Created by James Noeckel on 10/7/20.
//

#pragma once
#include <opencv2/opencv.hpp>

/**
 * Transform the colors of img1 to match the histogram of target
 * @param target BGR 8-bit image
 * @param img BGR 8-bit image
 * @param targetMask CV_8UC1 mask
 * @param imgMask CV_8UC1 mask
 */
void histogram_matching(const cv::Mat &target, cv::Mat &img, int channel=-1, const cv::Mat &targetMask=cv::Mat(), const cv::Mat &imgMask=cv::Mat());