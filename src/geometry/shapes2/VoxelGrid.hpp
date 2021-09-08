#pragma once

#include <memory>
#include <Eigen/Dense>
#include <vector>
#include "math/fields/ScalarField.h"

class VoxelGrid2D {
public:
    /**
     * Initialize from edge soup using extremely fast winding number
     * @param edges
     * @param grid_spacing
     * @param max_resolution
     */
    VoxelGrid2D(const std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> &edges, double grid_spacing, int max_resolution=std::numeric_limits<int>::max());

    /**
     * Initialize from 2D points
     * @param points
     * @param grid_spacing
     * @param max_resolution
     */
    VoxelGrid2D(const Eigen::Ref<const Eigen::MatrixX2d> &points, double grid_spacing, int max_resolution=std::numeric_limits<int>::max());

    /**
     * Initialize directly from data (x,y) -> data[x + y * x_res]
     * @param data data in row-major order
     * @param width number of columns
     * @param height number of rows (width * height must equal the number of elements in data)
     * @param min_x x coordinate of element 0
     * @param min_y y coordinate of element 0
     * @param spacing spacing between grid elements
     */
    VoxelGrid2D(std::vector<double> data, int width, int height, double min_x, double min_y, double spacing);
    double query(const Eigen::Ref<const Eigen::Vector2d> &pt) const;
    double query(int row, int col) const;
    /**
     * Compute marching squares contours from grid data. Contour is counter-clockwise encircling regions greater than the threshold.
     * @param hierarchy Map from indices of returned contours to lists of interior contours, if any. If there are N contours,
     * the index N gives the list of outer contours not contained by any other contours.
     * @param threshold value at which to extract an isosurface from the voxel grid.
     * @return
     */
    std::vector<std::vector<Eigen::Vector2d>> marching_squares(std::vector<std::vector<int>> &hierarchy, double threshold=NAN, bool bisect=true) const;

    /**
     * Initialize from a scalar field
     * @param field 2D scalar field
     * See data constructor for other argument descriptions
     */
    VoxelGrid2D(ScalarField<2>::Handle field, double min_x, double min_y, double max_x, double max_y, double spacing, int max_resolution=std::numeric_limits<int>::max());

    /**
     * Compute marching squares contours from grid data.
     * @param threshold value at which to extract an isosurface from the voxel grid.
     * @return
     */
    std::vector<std::vector<Eigen::Vector2d>> marching_squares(double threshold=NAN, bool bisect=true) const;
    Eigen::Vector2i resolution() const;
    void dilate(double threshold, bool erode, const Eigen::Ref<const Eigen::MatrixXi> &kernel=Eigen::Matrix3i::Ones(), int centerRow=1, int centerCol=1);
    void discretize(double threshold);
private:
    /**
     * Sets the resolution and grid spacing based on a maximum resolution and the current dimensions
     * @param grid_spacing
     * @param max_resolution
     * @param minPt
     * @param maxPt
     * @param clear
     */
    void set_resolution(double grid_spacing, int max_resolution, const Eigen::Vector2d &minPt, const Eigen::Vector2d &maxPt, bool clear);
    std::vector<std::vector<Eigen::Vector2d>> marching_squares(std::vector<std::vector<int>> &hierarchy, bool compute_hierarchy, double threshold, bool bisect=true) const;
    std::vector<double> data_;
    Eigen::Vector2d minPt_;
    Eigen::Array2i res_; //x, y
    double spacing_;
    double min_value_;
    ScalarField<2>::Handle field_;
};