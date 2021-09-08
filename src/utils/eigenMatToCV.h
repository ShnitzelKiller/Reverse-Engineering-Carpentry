//
// Created by James Noeckel on 9/29/20.
//

#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

/**
 *
 * @param matrix input matrix
 * @return
 */
cv::Mat eigenMatToCV(const Eigen::Ref<Eigen::Matrix<double, -1, -1>> &matrix);
cv::Mat eigenMatToCV(const Eigen::Ref<Eigen::Matrix<float, -1, -1>> &matrix);