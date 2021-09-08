//
// Created by James Noeckel on 5/4/20.
//

#pragma once
#include <Eigen/Dense>

struct Ray2d {
    Eigen::Vector2d o;
    Eigen::Vector2d d;
    double start;
    double end;

    /**
     * Intersect with another ray
     * @param other
     * @param t
     * @return true if intersection point is in the [start, end) range of both rays
     */
    bool intersect(const Ray2d &other, double &t) const;

    /**
     * Get the t values of a matrix of points
     * @param points Xx2 matrix of 2D points
     */
    Eigen::VectorXd project(const Eigen::Ref<const Eigen::MatrixX2d> &points) const;

    /**
     * Get signed distances of points to the line (right side following ray is positive)
     * @param points Xx2 matrix of 2D points
     */
    Eigen::VectorXd orthDist(const Eigen::Ref<const Eigen::MatrixX2d> &points) const;

    Eigen::VectorXd realDist(const Eigen::Ref<const Eigen::MatrixX2d> &points) const;

    double distBetween(const Ray2d &other) const;

    /**
     * Get the position of the ray at distance t
     * @param t distance value
     */
    Eigen::Vector2d sample(double t) const;

    /**
     *
     * @param origin origin point
     * @param direction normalized direction
     * @param start minimum distance
     * @param end maximum distance
     */
    Ray2d(Eigen::Vector2d origin, Eigen::Vector2d direction, double start=std::numeric_limits<double>::lowest(), double end=std::numeric_limits<double>::max());
    explicit Ray2d(double start=std::numeric_limits<double>::lowest(), double end=std::numeric_limits<double>::max());
    Ray2d(const std::pair<Eigen::Vector2d, Eigen::Vector2d> &edge2D);
};

