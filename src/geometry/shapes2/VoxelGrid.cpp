//
// Created by James Noeckel on 4/2/20.
//

#include "VoxelGrid.hpp"
#include <unordered_map>
#include <utility>
#include "utils/IntervalTree.h"
#include "utils/sorted_data_structures.hpp"

#define BISECTION_STEPS 5

using namespace Eigen;

static const std::vector<std::vector<std::pair<int, int>>> database =
        {{},
        {{3, 0}},
        {{0, 1}},
        {{3, 1}},
        {{1, 2}},
        {{1, 0}, {3, 2}},
        {{0, 2}},
        {{3, 2}},
        {{2, 3}},
        {{2, 0}},
        {{2, 1}, {0, 3}},
        {{2, 1}},
        {{1, 3}},
        {{1, 0}},
        {{0, 3}},
        {}};

void VoxelGrid2D::set_resolution(double grid_spacing, int max_resolution, const Eigen::Vector2d &minPt, const Eigen::Vector2d &maxPt, bool clear) {
    Eigen::Array2d dims = maxPt - minPt;
    if (grid_spacing <= 0) {
        grid_spacing = dims.maxCoeff() / max_resolution;
    }
    res_ = (dims / grid_spacing).ceil().cast<int>();
    int ind;
    int biggest_res = res_.maxCoeff(&ind);
    if (biggest_res > max_resolution) {
        grid_spacing = dims[ind] / max_resolution;
        res_ = (dims / grid_spacing).ceil().cast<int>();
    }
    spacing_ = grid_spacing;
    if (clear) {
        data_.resize(res_.prod(), 0.0f);
    } else {
        data_.resize(res_.prod());
    }
    if (res_.x() < 0) res_.x() = 0;
    if (res_.y() < 0) res_.y() = 0;
}

VoxelGrid2D::VoxelGrid2D(const std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> &edges, double grid_spacing,
                         int max_resolution) : min_value_(0.0f) {
    IntervalTree<double, Interval<double>> tree;
    std::vector<std::pair<Interval<double>, Interval<double>>> intervals;
    minPt_ = Eigen::Vector2d(std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
    Eigen::Vector2d maxPt(std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest());
    for (const auto & edge : edges) {
        double a = edge.first.x();
        double b = edge.second.x();
        if (a == b) continue;
        double depth1 = edge.first.y();
        double depth2 = edge.second.y();
        if (b < a) {
            std::swap(a, b);
            std::swap(depth1, depth2);
        }
        intervals.emplace_back(Interval<double>(a, b), Interval<double>(depth1, depth2));
        minPt_ = minPt_.cwiseMin(edge.first);
        minPt_ = minPt_.cwiseMin(edge.second);
        maxPt = maxPt.cwiseMax(edge.first);
        maxPt = maxPt.cwiseMax(edge.second);
    }
    tree.build(intervals.begin(), intervals.end());
    set_resolution(grid_spacing, max_resolution, minPt_, maxPt, false);
    for (size_t i=0; i<res_.y(); i++) {
        for (size_t j=0; j<res_.x(); j++) {
            double pos = minPt_.x() + (0.5 + j) * grid_spacing;
            double depth = minPt_.y() + (0.5 + i) * grid_spacing;
            auto result = tree.query(pos);
            int crossings = 0;
            for (const auto& pair : result) {
                double real_depth =
                        (pos - pair.first.start) / (pair.first.end - pair.first.start) *
                        (pair.second.end - pair.second.start) + pair.second.start;
                if (real_depth >= depth) {
                    crossings++;
                }
            }
            data_[i * res_.x() + j] = crossings % 2 == 1 ? 1.0f : 0.0f;
        }
    }
}

VoxelGrid2D::VoxelGrid2D(const Eigen::Ref<const Eigen::MatrixX2d> &points, double grid_spacing, int max_resolution) : min_value_(0.0f) {
    minPt_ = points.colwise().minCoeff().transpose();
    Eigen::Vector2d maxPt = points.colwise().maxCoeff().transpose();
    set_resolution(grid_spacing, max_resolution, minPt_, maxPt, true);
    for (int i=0; i<points.rows(); i++) {
        Eigen::Array2i ind = ((points.row(i).transpose()-minPt_).array()/spacing_).floor().cast<int>();
        if (ind.x() >= 0 && ind.y() >= 0 && ind.x() < res_.x() && ind.y() < res_.y()) {
            data_[ind.y() * res_.x() + ind.x()] += 1.0f;
        }
    }
}

VoxelGrid2D::VoxelGrid2D(std::vector<double> data, int width, int height, double min_x, double min_y, double spacing)
: spacing_(spacing), minPt_(min_x, min_y), res_(width, height), data_(std::move(data)) {
    if (!data_.empty())
        min_value_ = *std::min_element(data_.begin(), data_.end());
}

VoxelGrid2D::VoxelGrid2D(ScalarField<2>::Handle field, double min_x, double min_y, double max_x, double max_y, double spacing, int max_resolution)
  : spacing_(spacing), minPt_(min_x, min_y), field_(std::move(field)) {
    Vector2d maxPt(max_x, max_y);
    set_resolution(spacing, max_resolution, minPt_, maxPt, false);
    int n = res_.prod();
    MatrixX2d Q(n, 2);
    for (int i=0; i<res_.y(); ++i) {
        for (int j=0; j<res_.x(); ++j) {
            Q.row(i * res_.x() + j) = (minPt_.array() + spacing_ * (0.5 + Array2d(j, i))).transpose();
        }
    }
    VectorXd values = (*field_)(Q);
    for (int i=0; i<n; ++i) {
        data_[i] = values(i);
    }
    /*for (int i=0; i<height; ++i) {
        for (int j=0; j<width; ++j) {
            RowVector2d pt = (minPt_.array() + spacing_ * (0.5 + Array2d(j, i))).transpose();
            data_[i*width+j] = (*field_)(pt)(0);
        }
    }*/
}


double VoxelGrid2D::query(const Eigen::Ref<const Eigen::Vector2d> &pt) const {
    Eigen::Array2i ind = ((pt-minPt_).array()/spacing_).floor().cast<int>();
    if (ind.x() < 0 || ind.y() < 0 || ind.x() >= res_.x() || ind.y() >= res_.y()) {
        return min_value_;
    } else {
        return data_[ind.y() * res_.x() + ind.x()];
    }
}

double VoxelGrid2D::query(int row, int col) const {
    if (row < 0 || col < 0 || row >= res_.y() || col >= res_.x()) return min_value_;
    return data_[row * res_.x() + col];
}

/**
 * Compute a unique ID for the edge with the given index in the given marching squares cell.
 *
 * @param cols number of columns in marching squares grid (1 more than number of columns of data)
 * @param index Index of edge within cell. 0 -> bottom, 1 -> right, 2 -> top, 3 -> left
 * @return
 */
size_t cell_edge_to_id(size_t row, size_t col, size_t cols, size_t num_horizontal_edges, int index) {
    size_t global_edge_id;
    if (index == 0) {
        size_t cell_id = row * cols + col;
        global_edge_id = cell_id;
    } else if (index == 1) {
        size_t cell_id = row * (cols+1) + col;
        global_edge_id = num_horizontal_edges + cell_id + 1;
    } else if (index == 2) {
        size_t cell_id = row * cols + col;
        global_edge_id = cell_id + cols;
    } else if (index == 3) {
        size_t cell_id = row * (cols+1) + col;
        global_edge_id = num_horizontal_edges + cell_id;
    } else {
        assert(false);
    }
    return global_edge_id;
}

std::vector<std::vector<Eigen::Vector2d>> VoxelGrid2D::marching_squares(std::vector<std::vector<int>> &hierarchy, double threshold, bool bisect) const {
    return marching_squares(hierarchy, true, threshold, bisect);
}

std::vector<std::vector<Eigen::Vector2d>> VoxelGrid2D::marching_squares(double threshold, bool bisect) const {
    std::vector<std::vector<int>> hierarchy;
    return marching_squares(hierarchy, false, threshold, bisect);
}

std::vector<std::vector<Eigen::Vector2d>>
VoxelGrid2D::marching_squares(std::vector<std::vector<int>> &hierarchy, bool compute_hierarchy, double threshold, bool bisect) const {
    //default threshold: sort data values, and find the median in the range after the first non-minimum value as the max density,
    //then take the midpoint between the min and max density.
    if (!std::isfinite(threshold)) {
        std::vector<int> indices(data_.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {return data_[a] < data_[b];});
        double minval = data_[indices[0]];
        auto it = std::upper_bound(indices.begin(), indices.end(), minval, [&] (double a, int b) {return a < data_[b];});
        if (it != indices.end()) {
            /*double median = data_[*(it + std::distance(it, indices.end())/2)];
            threshold = 0.5 * (minval + median);*/
            double weight = 0.0;
            double mean = 0.0;
            for (auto it2 = it; it2 != indices.end(); it2++) {
                double newweight = weight + 1.0;
                mean = (weight * mean + data_[*it2]) / newweight;
                weight = newweight;
            }
            threshold = 0.5 * (minval + mean);
        } else {
            return {};
        }
    }

    size_t num_horizontal_edges = (res_.x() + 1) * (res_.y() + 2);
    size_t num_vertical_edges = (res_.x() + 2) * (res_.y() + 1);
    size_t total_edges = num_horizontal_edges + num_vertical_edges;
    std::vector<int> edge_to_segment(total_edges, -1);

    for (int i=0; i<res_.y()+1; i++) {
        for (int j=0; j<res_.x()+1; j++) {
            unsigned char index =
                    (query(i-1, j-1) > threshold ? 1U : 0U) |
                    ((query(i-1, j) > threshold ? 1U : 0U) << 1) |
                    ((query(i, j) > threshold ? 1U : 0U) << 2) |
                    ((query(i, j-1) > threshold ? 1U : 0U) << 3);

            std::vector<std::pair<int, int>> cell_edges = database[index];
            for (const auto &edge : cell_edges) {
                size_t global_edge_id_1 = cell_edge_to_id(i, j, res_.x()+1, num_horizontal_edges, edge.first);
                size_t global_edge_id_2 = cell_edge_to_id(i, j, res_.x()+1, num_horizontal_edges, edge.second);
                edge_to_segment[global_edge_id_1] = global_edge_id_2;
            }
        }
    }

    std::vector<int> edge_to_contour(total_edges, -1);
    std::vector<std::vector<Eigen::Vector2d>> contours;
    for (size_t loop_base_index = 0; loop_base_index < total_edges; loop_base_index++) {
        if (edge_to_segment[loop_base_index] >= 0 && edge_to_contour[loop_base_index] < 0) {
            size_t start_index = loop_base_index;
            bool closed = false;
            contours.emplace_back();
            for (size_t i=0; i<total_edges; i++) {
                edge_to_contour[start_index] = contours.size()-1;
                /** index (x, y) */
                Eigen::Array2i indA, indB;
                /** axis along which the segment lies, s.t. indA(axis) != indB(axis) */
                int axis;
                if (start_index >= num_horizontal_edges) {
                    // vertical segment intersected
                    axis = 1;
                    size_t col = (start_index-num_horizontal_edges) % (res_.x() + 2);
                    size_t row = (start_index-num_horizontal_edges) / (res_.x() + 2);
                    indA = Eigen::Array2i(col - 1, row - 1);
                    indB = Eigen::Array2i(col - 1, row);
                } else {
                    // horizontal segment intersected
                    axis = 0;
                    size_t col = start_index % (res_.x() + 1);
                    size_t row = start_index / (res_.x() + 1);
                    indA = Eigen::Array2i(col - 1, row - 1);
                    indB = Eigen::Array2i(col, row - 1);
                }
                double valA = query(indA.y(), indA.x());
                double valB = query(indB.y(), indB.x());

                if (bisect && field_) {
                    // bisection in world space
                    bool switched = valA > valB;
                    Vector2d ptA = (minPt_.array() + (0.5 + indA.cast<double>()) * spacing_).matrix();
                    Vector2d ptB = (minPt_.array() + (0.5 + indB.cast<double>()) * spacing_).matrix();
                    /*if (switched) {
                        std::swap(ptA, ptB);
                    }*/
                    Vector2d midpoint;
                    midpoint(1-axis) = ptA(1-axis);
                    midpoint(axis) = 0.5 * (ptA(axis) + ptB(axis));
                    for (unsigned iter=0; iter<BISECTION_STEPS; ++iter) {
                        double valMid = (*field_)(midpoint.transpose())(0);
                        if ((valMid < threshold) ^ switched) {
                            ptA(axis) = midpoint(axis);
                        } else {
                            ptB(axis) = midpoint(axis);
                        }
                        midpoint(axis) = 0.5 * (ptA(axis) + ptB(axis));
                    }
                    contours.back().emplace_back(std::move(midpoint));
                } else {
                    double weightB = (threshold - valA)/(valB - valA);
                    Eigen::Vector2d midpointTransformed = (minPt_.array() + (0.5 + (1.0f - weightB) * indA.cast<double>() + weightB * indB.cast<double>()) * spacing_).matrix();
                    contours.back().emplace_back(std::move(midpointTransformed));
                }
                start_index = edge_to_segment[start_index];
                if (start_index == loop_base_index) {
                    closed = true;
                    break;
                }
            }
            assert(closed);
        }
    }
    if (compute_hierarchy) {
        hierarchy.clear();
        hierarchy.resize(contours.size() + 1);
        //traverse scan lines to find hierarchy
        std::vector<int> contour_stack = {static_cast<int>(contours.size())};
        for (int row = 0; row < res_.y(); row++) {
            for (int col = 0; col < res_.x() + 1; col++) {
                size_t edge_index = (res_.x() + 1) * (row + 1) + col;
                int contour_id = edge_to_contour[edge_index];
                if (contour_id >= 0) {
                    if (contour_id == contour_stack.back()) {
                        contour_stack.pop_back();
                    } else {
                        sorted_insert(hierarchy[contour_stack.back()], contour_id);
                        contour_stack.push_back(contour_id);
                    }
                }
            }
            assert(contour_stack.size() == 1);
        }
    }
    return contours;
}

Eigen::Vector2i VoxelGrid2D::resolution() const {
    return res_;
}
void VoxelGrid2D::discretize(double threshold) {
    field_.reset();
    for (int i=0; i<res_.y(); ++i) {
        for (int j=0; j<res_.x(); ++j) {
            size_t ind = i * res_.x() + j;
            data_[ind] = data_[ind] > threshold ? 1 : 0;
        }
    }
}
void VoxelGrid2D::dilate(double threshold, bool erode, const Eigen::Ref<const Eigen::MatrixXi> &kernel, int centerRow, int centerCol) {
    field_.reset();
    std::vector<double> newGrid(data_.size());
    for (int i=0; i<res_.y(); ++i) {
        for (int j=0; j<res_.x(); ++j) {
            size_t ind = i * res_.x() + j;
            bool contained = erode;
            for (int di=0; di<kernel.rows(); ++di) {
                for (int dj=0; dj<kernel.cols(); ++dj) {
                    int pi = i + di - centerRow;
                    int pj = j + dj - centerCol;
                    size_t ind2 = pi * res_.x() + pj;
                    if (kernel(di, dj)) {
                        if (pi >= 0 && pi < res_.y() && pj >= 0 && pj < res_.x()) {
                            if (erode ^ (data_[ind2] > threshold)) {
                                contained = !erode;
                                break;
                            }
                        } else if (erode) contained = false;
                    }
                }
            }
            newGrid[ind] = contained ? 1 : 0;
        }
    }
    data_ = newGrid;
}

