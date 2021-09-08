//
// Created by James Noeckel on 9/2/20.
//

#pragma once
#include "ScalarField.h"
#include <vector>
#include "utils/typedefs.hpp"

/**
 * Kernel density field, where the value at any point is approximately Sum[Exp[-d_i^2/(2*sigma*sigma)],{i,1,N}]
 * where d_i is the distance of point i from the query point and sigma is the standard deviation of the gaussian kernel
 */
class PointDensityField : public ScalarField<2> {
public:
    PointDensityField(PointCloud2::Handle points, double stdev);
    Eigen::VectorXd operator()(const Eigen::Ref<const Eigen::Matrix<double, -1, 2>> &Q) const override;
private:
    Eigen::RowVector2i getGridIndex(const Eigen::Ref<const Eigen::RowVector2d> &point) const;
    std::vector<std::vector<unsigned>> indexGrid_;
    PointCloud2::Handle points_;
    Eigen::RowVector2d minPt;
    Eigen::RowVector2d dims;
    Eigen::RowVector2i res; //x (cols), y (rows)
    double denom_;
    double radius_;
};

