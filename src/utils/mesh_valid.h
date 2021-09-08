//
// Created by James Noeckel on 1/11/21.
//

#pragma once
#include <Eigen/Dense>

/**
 * Check if the mesh has valid element indices
 * @param V
 * @param F
 * @return element with faulty indices, or -1 if no errors
 */
int mesh_valid(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);