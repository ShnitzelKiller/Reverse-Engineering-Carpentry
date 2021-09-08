//
// Created by James Noeckel on 4/12/20.
//

#pragma once
#include <Eigen/Dense>
#include <utils/typedefs.hpp>
#include "WindingNumberData.h"

/**
 * Generate precomputed data for fast winding number computation
 * @param cloud
 * @param k
 * @param data precomputed winding number data
 * @param winding_number_stride ratio of original points to points used in winding number computation
 */
void precompute_fast_winding_number(const PointCloud3 &cloud, int k_n, WindingNumberData &data, int winding_number_stride=1);

/**
 * Compute winding number at query points given precomputed data
 * @param data result of precompute_fast_winding_number()
 * @param Q query points (nx3)
 * @return
 */
Eigen::VectorXd fast_winding_number(const WindingNumberData &data, const Eigen::MatrixXd &Q);

/**
 * compute winding number from scratch
 * @param cloud
 * @param Q
 * @param k_n
 * @param winding_number_stride
 * @return
 */
//Eigen::VectorXd fast_winding_number(PointCloud::Ref cloud, const Eigen::MatrixXd &Q, int k_n, int winding_number_stride=1);