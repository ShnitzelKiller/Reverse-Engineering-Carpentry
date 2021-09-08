//
// Created by James Noeckel on 1/16/21.
//

#include "density_contour.h"
#include "math/fields/PointDensityField.h"
#include "geometry/shapes2/VoxelGrid.hpp"

using namespace Eigen;

std::vector<std::vector<Eigen::Vector2d>> density_contour(PointCloud2::Handle &cloud2d, std::vector<std::vector<int>> &hierarchy, double densityThreshold, double voxel_width, int max_resolution) {
    double stdev = voxel_width/4.0;
    RowVector2d min2d = cloud2d->P.colwise().minCoeff().array() - (3*stdev);
    RowVector2d max2d = cloud2d->P.colwise().maxCoeff().array() + (3*stdev);
    ScalarField<2>::Handle field(new PointDensityField(cloud2d, stdev));
    VoxelGrid2D voxels(std::move(field), min2d.x(), min2d.y(), max2d.x(), max2d.y(), voxel_width, max_resolution);
    return voxels.marching_squares(hierarchy, densityThreshold);
}