//
// Created by James Noeckel on 1/16/21.
//

#pragma once

#include <vector>
#include <Eigen/Dense>
#include "utils/typedefs.hpp"

/**
 * Extract a density-based boundary representation of the shape approximating the input 2D point cloud
 * @param cloud2d
 * @param hierarchy output hierarchy of contours
 * @param densityThreshold
 * @param voxel_width
 * @param max_resolution maximum grid resolution in any dimension
 * @return list of contours
 */
std::vector<std::vector<Eigen::Vector2d>> density_contour(PointCloud2::Handle &cloud2d, std::vector<std::vector<int>> &hierarchy, double densityThreshold, double voxel_width, int max_resolution=1000);