//
// Created by James Noeckel on 9/2/20.
//

#pragma once
#include "ScalarField.h"
#include "geometry/WindingNumberData.h"

class WindingNumberField3D : public ScalarField<3> {
public:
    /**
     * Constructs a field from existing precomputed data. Note that this only refers to the passed data, so the data
     * must not be destroyed for this object's lifespan.
     * @param data
     */
    WindingNumberField3D(WindingNumberData::Handle data);
    Eigen::VectorXd operator()(const Eigen::Ref<const Eigen::Matrix<double, -1, 3>> &Q) const override;
private:
    WindingNumberData::Handle data_;
};