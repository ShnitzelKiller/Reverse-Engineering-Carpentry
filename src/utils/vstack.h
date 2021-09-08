//
// Created by James Noeckel on 11/6/20.
//

#pragma once

#include <Eigen/Dense>

template <typename T, int N>
Eigen::Matrix<T, -1, N> vstack(const std::vector<Eigen::Matrix<T, N, 1>> &vectors) {
    Eigen::Matrix<T, -1, N> mat(vectors.size(), N);
    for (int p = 0; p < vectors.size(); p++) {
        mat.row(p) = vectors[p].transpose();
    }
    return mat;
}