//
// Created by James Noeckel on 10/19/20.
//

#pragma once
#include <Eigen/Dense>
template <typename T, int ND>
Eigen::Matrix<T, -1, ND> cyclicPermuteRows(const Eigen::Matrix<T, -1, ND> &mat, int offset) {
    int N = mat.rows();
    int D = mat.cols();
    Eigen::Matrix<T, -1, -1> mat2(N, D);
    if (offset > 0) {
        mat2.block(0, 0, offset, D) = mat.block(N-offset, 0, offset, D);
        mat2.block(offset, 0, N-offset, D) = mat.block(0, 0, N-offset, D);
    } else if (offset < 0) {
        offset = -offset;
        mat2.block(N-offset, 0, offset, D) = mat.block(0, 0, offset, D);
        mat2.block(0, 0, N-offset, D) = mat.block(offset, 0, N-offset, D);
    } else mat2 = mat;
    return mat2;
}