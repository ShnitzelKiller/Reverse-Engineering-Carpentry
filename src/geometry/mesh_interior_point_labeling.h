//
// Created by James Noeckel on 3/26/20.
//

#pragma once
#include <vector>
#include <Eigen/Dense>

std::vector<bool> mesh_interior_point_labeling(const Eigen::MatrixX3d &points, const Eigen::MatrixX3d &V, const Eigen::MatrixX3i &F);