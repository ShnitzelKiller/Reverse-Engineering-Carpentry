//
// Created by James Noeckel on 12/9/20.
//

#include "MultiRay3d.h"

MultiRay3d::MultiRay3d(Eigen::Vector3d origin, Eigen::Vector3d direction) : o(std::move(origin)), d(std::move(direction)) {

}

Edge3d MultiRay3d::getEdge(size_t i) const {
    return Edge3d(o + ranges[i].first * d, o + ranges[i].second * d);
}

size_t MultiRay3d::size() const {
    return ranges.size();
}