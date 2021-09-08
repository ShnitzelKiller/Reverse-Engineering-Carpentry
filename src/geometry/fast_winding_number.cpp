//
// Created by James Noeckel on 4/12/20.
//

#include "fast_winding_number.h"
#include <iostream>
#include <igl/copyleft/cgal/point_areas.h>
#include <igl/octree.h>
#include <igl/knn.h>
#include <igl/fast_winding_number.h>

void precompute_fast_winding_number(const PointCloud3 &cloud, int k_n, WindingNumberData &data, int winding_number_stride) {
    if (winding_number_stride != 1) {
        data.P.resize(cloud.P.rows() / winding_number_stride, 3);
        data.N.resize(cloud.P.rows() / winding_number_stride, 3);
        for (size_t i = 0; i < data.P.rows(); i++) {
            data.P.row(i) = cloud.P.row(i * winding_number_stride);
            data.N.row(i) = cloud.N.row(i * winding_number_stride);
        }
    } else {
        data.P = cloud.P;
        data.N = cloud.N;
    }
    Eigen::MatrixX3d CN;
    Eigen::VectorXd W;
    std::cout << "computing octree" << std::endl;
    igl::octree(data.P, data.octree_indices, data.CH, CN, W);
    std::cout << data.octree_indices.size() << " octree cells; CH: " << data.CH.rows() << "x" << data.CH.cols() << "; CN: "
              << CN.rows() << "x" << CN.cols()
              << "; W: " << W.rows() << "x" << W.cols() << std::endl;
    Eigen::MatrixXi I(data.P.rows(), k_n);
    std::cout << "computing " << k_n << " nearest neighbors" << std::endl;
    igl::knn(data.P, k_n, data.octree_indices, data.CH, CN, W, I);
    std::cout << "computing areas" << std::endl;
    igl::copyleft::cgal::point_areas(data.P, I, data.N, data.A);
    std::cout << "Areas: " << data.A.size() << std::endl;
    std::cout << "precomputing winding number" << std::endl;
    igl::fast_winding_number(data.P, data.N, data.A, data.octree_indices, data.CH, 2, data.CM, data.R, data.EC);
}

Eigen::VectorXd fast_winding_number(const WindingNumberData &data, const Eigen::MatrixXd &Q) {
    Eigen::VectorXd windingNumber;
    igl::fast_winding_number(data.P, data.N, data.A, data.octree_indices, data.CH, data.CM, data.R, data.EC, Q, 2, windingNumber);
    return windingNumber;
}

/*Eigen::VectorXd fast_winding_number (PointCloud::Ref cloud, const Eigen::MatrixXd &Q, int k_n, int winding_number_stride) {
    WindingNumberData data = precompute_fast_winding_number(cloud, k_n, winding_number_stride);
    return fast_winding_number(data, Q);
}*/