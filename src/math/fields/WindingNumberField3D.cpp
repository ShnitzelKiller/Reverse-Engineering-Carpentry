//
// Created by James Noeckel on 9/2/20.
//

#include "WindingNumberField3D.h"
#include "geometry/fast_winding_number.h"

Eigen::VectorXd WindingNumberField3D::operator()(const Eigen::Ref<const Eigen::Matrix<double, -1, 3>> &Q) const {
    return fast_winding_number(*data_, Q);
}

WindingNumberField3D::WindingNumberField3D(WindingNumberData::Handle data) : data_(std::move(data)) {

}
