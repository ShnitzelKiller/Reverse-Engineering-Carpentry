//
// Created by James Noeckel on 1/11/21.
//

#include "mesh_valid.h"
#include <iostream>

int mesh_valid(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) {
    for (size_t i=0; i<F.rows(); ++i) {
        for (int j=0; j<3; ++j) {
            if (F(i, j) < 0 || F(i, j) >= V.rows()) {
                std::cout << "out of bounds " << F.row(i) << std::endl;
                return i;
            }
        }
        if (F(i, 0) == F(i, 1) || F(i, 0) == F(i, 2) || F(i, 1) == F(i, 2)) {
            std::cout << "degenerate triangle " << F.row(i) << std::endl;
            return i;
        }
    }
    return -1;
}