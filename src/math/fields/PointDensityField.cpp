//
// Created by James Noeckel on 9/2/20.
//

#include "PointDensityField.h"
#include "utils/sorted_data_structures.hpp"

using namespace Eigen;

RowVector2i PointDensityField::getGridIndex(const Ref<const RowVector2d> &point) const {
    RowVector2i ind = ((point - minPt)/radius_).array().floor().cast<int>();
    return ind;
}

PointDensityField::PointDensityField(PointCloud2::Handle points, double stdev) : points_(std::move(points)), denom_(1.0/(2.0 * stdev * stdev)), radius_(3 * stdev) {
    minPt = points_->P.colwise().minCoeff();
    RowVector2d maxPt = points_->P.colwise().maxCoeff();
    dims = maxPt - minPt;
    res = (dims/radius_).array().ceil().cast<int>();
    indexGrid_.resize(res.prod());
    if (!indexGrid_.empty()) {
        for (unsigned i = 0; i < points_->P.rows(); ++i) {
            RowVector2i ind = getGridIndex(points_->P.row(i));
            indexGrid_[ind.y() * res.x() + ind.x()].push_back(i);
        }
    }
}


VectorXd PointDensityField::operator()(const Ref<const Matrix<double, -1, 2>> &Q) const {
    sorted_map<size_t, std::vector<size_t>> gridToQuery; //grid index -> query point index(es)
    size_t n = Q.rows();
    for (size_t q=0; q<n; ++q) {
        RowVector2i ind = getGridIndex(Q.row(q));

        for (int xoffset=-1; xoffset<=1; ++xoffset) {
            for (int yoffset=-1; yoffset<=1; ++yoffset) {
                RowVector2i indo = ind + RowVector2i(xoffset, yoffset);
                if (indo.x() >= 0 && indo.x() < res.x() && indo.y() >= 0 && indo.y() < res.y()) {
                    size_t flatIndex = indo.y() * res.x() + indo.x();
                    auto it = sorted_find(gridToQuery, flatIndex);
                    if (it == gridToQuery.end()) {
                        sorted_insert(gridToQuery, flatIndex, std::vector<size_t>(1, q));
                    } else {
                        it->second.push_back(q);
                    }
                }
            }
        }
    }
    VectorXd results = VectorXd::Zero(n);
    for (const auto &pair : gridToQuery) {
        const auto &cell = indexGrid_[pair.first];
        for (unsigned int i : cell) {
            for (size_t q : pair.second) {
                double distSquared = (points_->P.row(i) - Q.row(q)).squaredNorm();
                double w = std::exp(-distSquared*denom_);
                results(q) += w;
            }
        }
    }
    return results;
}
