//
// Created by James Noeckel on 12/9/20.
//

#pragma once

#include <Eigen/Dense>
#include <vector>
#include "utils/typedefs.hpp"

struct MultiRay3d {
    MultiRay3d() = default;
    MultiRay3d(Eigen::Vector3d origin, Eigen::Vector3d direction);
    Eigen::Vector3d o, d;
    std::vector<std::pair<double, double>> ranges;
    Edge3d getEdge(size_t i) const;
    size_t size() const;
};
