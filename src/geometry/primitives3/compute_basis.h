//
// Created by James Noeckel on 2/11/20.
//

#pragma once
#include <Eigen/Dense>
#include "utils/typedefs.hpp"



Eigen::Matrix<double, 2, 3> compute_basis(const Eigen::Ref<const Eigen::Vector3d> &normal);
