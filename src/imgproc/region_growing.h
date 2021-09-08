//
// Created by James Noeckel on 1/21/20.
//
#pragma once

#include <opencv2/opencv.hpp>

void region_growing(const cv::Mat &image, cv::Mat &seed, float gmm_weight=1.0);