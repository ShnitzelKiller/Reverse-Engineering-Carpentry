//
// Created by James Noeckel on 11/3/20.
//

#pragma once
#include <Eigen/Dense>

void find_affine(const Eigen::Ref<const Eigen::MatrixX3d> &P, const Eigen::Ref<const Eigen::MatrixX3d> &Q, double &scale, Eigen::Vector3d &trans, Eigen::Quaterniond &rot);