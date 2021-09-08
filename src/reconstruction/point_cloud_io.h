//
// Created by James Noeckel on 3/13/20.
//

#pragma once
#include <Eigen/Dense>
#include <string>
#include <utils/typedefs.hpp>

bool load_pointcloud(const std::string &filename, PointCloud3::Handle &cloud);