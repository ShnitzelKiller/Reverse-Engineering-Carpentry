//
// Created by James Noeckel on 9/2/20.
//

#include "FieldSlice.h"
#include <cassert>

Eigen::VectorXd FieldSlice::operator()(const Eigen::Ref<const Eigen::Matrix<double, -1, 2>> &Q) const {
    size_t n = Q.rows();
    size_t numPoints = samples_ * n;
    Eigen::MatrixX3d Q2(numPoints, 3);
    size_t offset = 0;
    for (size_t i=0; i<samples_; ++i) {
        for (size_t j=0; j < n; ++j) {
            Q2.row(j + offset) = rot_ * Eigen::Vector3d(Q(j, 0), Q(j, 1), -(i * spacing_) - offset_);
        }
        offset += n;
    }
    Eigen::MatrixXd allW = (*field3D_)(Q2);
    assert(allW.size() == numPoints);
    allW.resize(n, samples_);
    return allW.rowwise().maxCoeff();
}

FieldSlice::FieldSlice(ScalarField<3>::Handle field3D, const Eigen::Quaterniond& rot, double offset,
                       double depth, unsigned samplingRate)
                                                 : field3D_(std::move(field3D)), rot_(rot), offset_(offset), spacing_(samplingRate > 1 ? depth/(samplingRate-1) : 0), samples_(samplingRate) {

}
