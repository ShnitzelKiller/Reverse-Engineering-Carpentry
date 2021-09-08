//
// Created by James Noeckel on 12/2/19.
//

#pragma once

#include "utils/typedefs.hpp"
#include "geometry/shapes2/Primitive.h"
#include <memory>
#include <unordered_map>
#include "geometry/primitives3/MultiRay3d.h"

/**
 * A plane described by (n.dot(x)) + offset = 0
 * where n is the 3rd row of basis()
 */
struct BoundedPlane {
public:
    BoundedPlane(std::shared_ptr<Primitive> shape, const Eigen::Ref<const Eigen::Matrix<double, 3, 3>> &basis, double offset);

    BoundedPlane(const Eigen::Ref<const Eigen::Matrix<double, 3, 1>> &normal, double offset);

    /**
     * @param newbasis 2x2 matrix whose rows are the 2D basis vectors expressed in the current basis
     */
    void changeBasis(const Eigen::Ref<const Eigen::Matrix2d> &newbasis);

    void flip();

    /** 3D points of the current shape (hasShape() must return true) */
    Eigen::MatrixX3d points3D() const;

    /** unproject points in the plane basis into world space */
    Eigen::MatrixX3d points3D(const Eigen::Ref<const Eigen::MatrixX2d> &points) const;

    Eigen::MatrixX2d project(const Eigen::Ref<const Eigen::MatrixX3d> &points) const;

    /** signed normal distance of a set of points */
    Eigen::VectorXd normalDistance(const Eigen::Ref<const Eigen::MatrixX3d> &points) const;

    /**
     * local to world space rotation, last row is the normal
     */
    const Eigen::Matrix<double, 3, 3> &basis() const;

    double offset() const;

    Eigen::Vector3d normal() const;

    bool contains(const Eigen::Ref<const Eigen::Vector2d> &point, double margin=0) const;

    /**
     * Test if a 3D point is both in the plane and inside the bounding box
     * @param point
     * @param threshold maximum distance from the plane
     * @param margin margin around the shape
     * @return
     */
    bool contains3D(const Eigen::Ref<const Eigen::Vector3d> &point, double threshold,  double margin=0, double offset = 0.0) const;

    /**
     * Intersect 3D ray with this plane
     * @param ray_origin
     * @param ray_direction (does not need to be normalized)
     * @param t intersecting ray distance
     * @param margin margin around the shape for detecting intersection
     * @return true if the intersection is within the shape bounds (plus margin)
     */
    bool intersectRay(const Eigen::Ref<const Eigen::Vector3d> &ray_origin, const Eigen::Ref<const Eigen::Vector3d> &ray_direction, double &t, bool ignore_shape=false, double margin= 0.0) const;

    /**
     * Get line segment from intersecting this plane with another, such that both bounding hulls determine the endpoints
     * @param other
     * @param a, @param b segment endpoints, bounded by the hull of @param other and this (in the local basis)
     * @param margin margin around the shape for detecting intersection
     * @return true if there was intersection
     */
    bool intersect(const BoundedPlane &other, MultiRay3d &outRay, double margin=0.0) const;

    /**
     * Detect whether the shapes of both primitives overlap when projected onto each other
     */
    bool overlap(const BoundedPlane &other, double threshold, double margin=0.0, double offset=0.0) const;

    void addShape(int idx, std::shared_ptr<Primitive> shape);
    void setCurrentShape(int idx);
    void clearCurrentShape();
    void clearShapes();
    int getCurrentShape() const;
    const Primitive& getShape(int idx) const;
    Primitive& getShape(int idx);
    int getNumShapes() const;
    bool hasShape(int idx) const;
    bool hasShape() const;
    void serialize(std::ostream &o) const;
private:
    void intersectHelper(const Eigen::Vector3d &p, const Eigen::Vector3d &d, MultiRay3d &outRay, double margin) const;
    Eigen::Matrix<double, 3, 3> basis_;
    double offset_;
    std::unordered_map<int, std::shared_ptr<Primitive>> shapes_;
    //int curr_shape_ = 0;
    Primitive* curr_shape_ptr_ = nullptr;
    int curr_shape_id_ = -1;
};

std::ostream &operator<<(std::ostream &o, const BoundedPlane &plane);
