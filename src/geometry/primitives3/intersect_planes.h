//
// Created by James Noeckel on 9/4/20.
//

#pragma once

#include <Eigen/Dense>

/**
 * Find the ray representing the plane intersections
 * @param n1 plane 1 normal
 * @param n2 plane 2 normal
 * @param offset1 plane 1 offset
 * @param offset2 plane 2 offset
 * @param o ray origin
 * @param d ray direction
 */
void intersect_planes(const Eigen::Ref<const Eigen::RowVector3d> &n1, const Eigen::Ref<const Eigen::RowVector3d> &n2, double offset1, double offset2, Eigen::Ref<Eigen::Vector3d> o, Eigen::Ref<Eigen::Vector3d> d);