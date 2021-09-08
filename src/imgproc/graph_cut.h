//
// Created by James Noeckel on 10/5/20.
//

#pragma once
#include <opencv2/opencv.hpp>
#include "utils/typedefs.hpp"

/**
 *
 * @param data_costs CV_32FC2 image where channel 0 is the energy for label 0 and channel 1 is the energy for label 1
 * @param labels CV_8UC1 output
 * @param smoothness smoothness factor, applied to the pairwise term
 * @param contrast_image optional image for contrast sensitive smoothness term
 */
float graph_cut(const cv::Mat &data_costs, cv::Mat &labels, float smoothness, const std::vector<cv::Mat> &contrast_image={}, float sigma=10, const std::vector<cv::Mat> &contrast_masks={}, const std::vector<Edge2d> &guides={}, double guideExtension=0, bool diagonalEdges=false, float precision=1.0f);