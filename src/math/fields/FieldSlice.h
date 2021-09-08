//
// Created by James Noeckel on 9/2/20.
//

#pragma once
#include "ScalarField.h"

class FieldSlice : public ScalarField<2> {
public:
    /**
     * @param field3D 3D field of which this is a slice
     * @param rot rotation from local to world space, where x and y in the input space are the plane coordinates and any depth is in the -z direction
     * @param offset points sampled in 3D space are given by rot * (x, y, -offset).
     * @param depth thickness of the slice to be averaged for each 2D point
     * @param samplingRate number of samples in the thickness dimension (should be >1 if depth>0)
     */
    FieldSlice(ScalarField<3>::Handle field3D, const Eigen::Quaterniond& rot, double offset, double depth=0, unsigned samplingRate=1);
    Eigen::VectorXd operator()(const Eigen::Ref<const Eigen::Matrix<double, -1, 2>> &Q) const override;
private:
    const ScalarField<3>::Handle field3D_;
    double offset_;
    double spacing_;
    Eigen::Quaterniond rot_;
    unsigned samples_;
};

