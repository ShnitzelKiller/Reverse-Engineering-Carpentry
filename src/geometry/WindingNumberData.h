//
// Created by James Noeckel on 9/2/20.
//

#pragma once
#include <Eigen/Dense>
#include <vector>

struct WindingNumberData {
    Eigen::MatrixX3d P;
    Eigen::MatrixX3d N;
    Eigen::VectorXd A;
    std::vector<std::vector<size_t>> octree_indices;
    Eigen::MatrixXi CH;
    Eigen::MatrixX3d CM;
    Eigen::VectorXd R;
    Eigen::MatrixXd EC;
    typedef std::shared_ptr<WindingNumberData> Handle;
};