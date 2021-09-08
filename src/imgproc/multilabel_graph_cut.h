//
// Created by James Noeckel on 1/29/20.
//

#include <opencv2/opencv.hpp>

#pragma once

/**
 * Compute the segmentation of image using some initial labels using energy minimization via graph cuts.
 * @param image image for which to compute the segmentation
 * @param labels initial label map of type CV_32SC1, where -1 indicates an unknown value
 * @param data_weight weight of the data term
 * @param smoothness_weight weight of the smoothness term (should be greater than unity for full expressivity of the integral energy term)
 * @param label_penalty cost of reassigning a label provided in the input
 * @param sigma standard deviation of the Gaussian falloff in the smoothness term with respect to the difference of neighboring colors
 * @param components number of mixture components to use in the learned Gaussian mixture models
 * @param iterations number of iterations of global optimization
 * @param levels number of Gaussian pyramid levels to use as features
 */
void multilabel_graph_cut(const cv::Mat &image, cv::Mat &labels, float data_weight, float smoothness_weight, float label_penalty, float sigma, int components=5, int iterations=2, int levels=1);