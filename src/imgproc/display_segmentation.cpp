//
// Created by James Noeckel on 1/31/20.
//

#include "display_segmentation.h"

void display_segmentation(const cv::Mat &segmentation, cv::Mat &display, const Eigen::Ref<const Eigen::MatrixX3d> &colors) {
    if (display.type() != CV_8UC3 || display.size() != segmentation.size()) {
        display = cv::Mat(segmentation.size(), CV_8UC3);
    }
    for (int i=0; i<segmentation.rows; i++) {
        for (int j=0; j<segmentation.cols; j++) {
            int segment = segmentation.at<int32_t>(i, j);
            if (segment < 0) {
                display.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
            } else if (segment >= colors.rows()) {
                display.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
            } else {
                Eigen::RowVector3i color = (colors.row(segment) * 255).cast<int>();
                display.at<cv::Vec3b>(i, j) = cv::Vec3b(color.x(), color.y(), color.z());
            }
        }
    }
}