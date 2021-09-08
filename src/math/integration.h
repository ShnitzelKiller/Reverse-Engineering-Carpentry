//
// Created by James Noeckel on 1/8/20.
//

#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

/**
 * Compute the line integral of the bilinearly interpolated image with start and end points a and b, divided by the length of ab
 * If derivative is enabled, computes the derivative of the integral with respect to the rate of change vectors of a and b
 * Expects a CV_8UC if derivative is false, and CV_16S otherwise
 * @param img color image, or x derivative image
 * @param a
 * @param b
 * @param derivative if true, the following arguments are required: dadt, dbdt, img_y
 * @param dadt
 * @param dbdt
 * @param img_y y derivative image
 * @param draw debug draw the traced line on the input image
 * @return
 */
Eigen::VectorXd integrate_image(cv::Mat img,
                                const Eigen::Ref<const Eigen::Vector2d> &a,
                                const Eigen::Ref<const Eigen::Vector2d> &b,
                                bool derivative=false,
                                const Eigen::Ref<const Eigen::Vector2d> &dadt=Eigen::Vector2d(),
                                const Eigen::Ref<const Eigen::Vector2d> &dbdt=Eigen::Vector2d(),
                                cv::Mat img_y = cv::Mat(),
                                bool draw=false);