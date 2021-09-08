//
// Created by James Noeckel on 10/1/20.
//

#pragma once
#include <opencv2/opencv.hpp>

/**
 * Normalized cross correlation of patches
 * @param img1
 * @param img2
 * @param i1 row in img1
 * @param j1 col in img1
 * @param i2 row in img2
 * @param j2 col in img2
 * @param radius radius of patches, where ksize = radius*2+1
 */
float norm_cross_correlation(const cv::Mat &img1, const cv::Mat &img2, int i1, int j1, int i2, int j2, int radius, bool separate_channels=true, const cv::Mat &mask1 = cv::Mat(), const cv::Mat &mask2=cv::Mat());

/**
 * SSD distance between patches
 * @param img1
 * @param img2
 * @param i1 row in img1
 * @param j1 col in img1
 * @param i2 row in img2
 * @param j2 col in img2
 * @param radius radius of patches, where ksize = radius*2+1
 */
float ssd(const cv::Mat &img1, const cv::Mat &img2, int i1, int j1, int i2, int j2, int radius, const cv::Mat &mask1 = cv::Mat(), const cv::Mat &mask2 = cv::Mat());