//
// Created by James Noeckel on 9/29/20.
//

#include "eigenMatToCV.h"

cv::Mat eigenMatToCV(const Eigen::Ref<Eigen::Matrix<double, -1, -1>> &matrix) {
    cv::Mat output(matrix.rows(), matrix.cols(), CV_64FC1);
    for (size_t i=0; i<matrix.rows(); ++i) {
        for (size_t j=0; j<matrix.cols(); ++j) {
            output.at<double>(i, j) = matrix(i, j);
        }
    }
    return output;
}

cv::Mat eigenMatToCV(const Eigen::Ref<Eigen::Matrix<float, -1, -1>> &matrix) {
    cv::Mat output(matrix.rows(), matrix.cols(), CV_32FC1);
    for (size_t i=0; i<matrix.rows(); ++i) {
        for (size_t j=0; j<matrix.cols(); ++j) {
            output.at<float>(i, j) = matrix(i, j);
        }
    }
    return output;
}