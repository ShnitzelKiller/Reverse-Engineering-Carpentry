//
// Created by James Noeckel on 1/5/21.
//

#pragma once
#include <Eigen/Dense>

/**
 * Measure the largest inscribed circle diameter for a densely sampled counter-clockwise contour
 * @param contour
 * @return
 */
double polygon_thickness(const Eigen::Ref<const Eigen::MatrixX2d> &contour);