//
// Created by James Noeckel on 10/1/20.
//

#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <random>
#include "utils/settings.h"

/**
 *
 * @param allImages
 * @param allMasks
 * @param labels initial guess, and output. For input, 0 == known cluster 0, 1 == known cluster 1, and 2 == unknown
 * @param settings
 * @param groups list of lists of indices in allImages and allMasks belonging to separate groups
 * @param random_engine
 */
void joint_segmentation(const std::vector<cv::Mat> &allImages, const std::vector<cv::Mat> &allMasks, cv::Mat &labels, const Settings &settings, const std::vector<std::vector<int>> &groups=std::vector<std::vector<int>>(), std::mt19937 random_engine=std::mt19937(std::random_device{}()));