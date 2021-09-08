//
// Created by James Noeckel on 1/16/21.
//
#pragma once
#include <memory>
#include <vector>
#include <Eigen/Dense>
#include "Primitive.h"

/** return the primitive corresponding to the contour hierarchy provided, including sufficiently large holes (according to minSizeRatio)*/
std::shared_ptr<Primitive> convertContour(const std::vector<std::vector<Eigen::Vector2d>> &marching_squares_contours, std::vector<std::vector<int>> &hierarchy, double minSizeRatio);
