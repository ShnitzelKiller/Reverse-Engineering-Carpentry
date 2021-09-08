//
// Created by James Noeckel on 2/11/20.
//

#include "compute_basis.h"
Eigen::Matrix<double, 2, 3> compute_basis(const Eigen::Ref<const Eigen::Vector3d> &norm) {
    Eigen::Vector3d up(0, 0, 0);
    int ind = 0;
    double mincomp = std::abs(norm[0]);
    for (int i = 1; i < 3; i++) {
        double abscomp = std::abs(norm[i]);
        if (abscomp < mincomp) {
            ind = i;
            mincomp = abscomp;
        }
    }
    up[ind] = 1.0f;
    Eigen::Matrix<double, 2, 3> basis;
    basis.row(1) = norm.cross(up).normalized();
    basis.row(0) = basis.row(1).transpose().cross(norm).normalized();
    return basis;
}