//
// Created by James Noeckel on 1/31/20.
//

#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

void display_segmentation(const cv::Mat &segmentation, cv::Mat &display, const Eigen::Ref<const Eigen::MatrixX3d> &colors);