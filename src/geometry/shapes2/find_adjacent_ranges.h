//
// Created by James Noeckel on 4/23/20.
//

#pragma once
#include <vector>
#include <Eigen/Dense>

/**
 * Smallest looping inclusive range of indices containing all true elements of bitvector
 * interval start is always in the range [0, N-1], interval end may be greater than N-1
 */
std::pair<int, int> smallest_looping_range(const std::vector<bool> &bitvector);

/**
 * Find the list of (looping) indices of crossings where the contour starts and ceases being adjacent to each of the supplied edges.
 * @param contour
 * @param edges
 * @param projectedNeighbors sets of projected points of adjacent clusters. Must not be the same clusters whose adjacencies generated an edge in edges.
 * @param threshold
 * @return list of ((start, end), edgeIndex) where edgeIndex = -1 indicates an unconstrained sub curve
 */
std::vector<std::pair<std::pair<int, int>, int>> find_adjacent_ranges(const Eigen::Ref<const Eigen::MatrixX2d> &contour, const std::vector<Eigen::Matrix2d> &edges, const std::vector<Eigen::MatrixX2d> &projectedNeighbors, const std::vector<double> &thresholds);