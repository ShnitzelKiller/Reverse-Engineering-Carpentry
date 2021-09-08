//
// Created by James Noeckel on 12/2/19.
//

#include <Eigen/Dense>
#include <geometry/primitives3/compute_basis.h>
#include "BoundedPlane.h"
#include "intersect_planes.h"
#include "geometry/csg1d.h"

using namespace Eigen;

BoundedPlane::BoundedPlane(std::shared_ptr<Primitive> shape, const Ref<const Matrix<double, 3, 3>> &basis, double offset) : shapes_(1), basis_(basis), offset_(offset) {
    shapes_[0] = (std::move(shape));
    setCurrentShape(0);
}

BoundedPlane::BoundedPlane(const Ref< const Eigen::Matrix<double, 3, 1>> &normal, double offset) : offset_(offset) {
    basis_.row(2) = normal.transpose();
    basis_.block<2, 3>(0, 0) = compute_basis(normal);
}

void BoundedPlane::changeBasis(const Ref< const Eigen::Matrix2d> &newbasis) {
    basis_.block<2, 3>(0, 0) = newbasis * basis_.block<2, 3>(0, 0);
}

void BoundedPlane::flip() {
    basis_.row(2) = -basis_.row(2);
    basis_.row(0) = -basis_.row(0);
    offset_ = -offset_;
}

double BoundedPlane::offset() const {
    return offset_;
}

Vector3d BoundedPlane::normal() const {
    return basis_.row(2).transpose();
}

MatrixX3d BoundedPlane::points3D() const {
    MatrixX2d pts = curr_shape_ptr_->points();
    return points3D(pts);
}

MatrixX3d BoundedPlane::points3D(const Ref<const MatrixX2d> &points) const {
    return (points * basis_.block<2, 3>(0, 0)).rowwise() - offset_ * basis_.row(2);
}

MatrixX2d BoundedPlane::project(const Ref<const MatrixX3d> &points) const {
    return points * basis_.block<2, 3>(0, 0).transpose();
}

VectorXd BoundedPlane::normalDistance(const Eigen::Ref<const Eigen::MatrixX3d> &points) const {
    return (points * basis_.row(2).transpose()).array() + offset_;
}

const Matrix<double, 3, 3> &BoundedPlane::basis() const {
    return basis_;
}

bool BoundedPlane::contains(const Ref<const Vector2d> &point, double margin) const {
    return curr_shape_ptr_->contains(point, margin);
}

bool BoundedPlane::contains3D(const Ref<const Vector3d> &point, double threshold, double margin, double offset) const {
    if (threshold < 0 || std::abs(basis_.row(2) * point + offset_ - offset) <= threshold) {
        RowVector2d p2d = project(point.transpose());
        return contains(p2d.transpose(), margin);
    }
    return false;
}

bool BoundedPlane::intersectRay(const Ref<const Vector3d> &ray_origin, const Ref<const Vector3d> &ray_direction, double &t, bool ignore_shape, double margin) const {
    Vector3d plane_center = - offset_ * basis_.row(2).transpose();
    Vector3d offset = plane_center - ray_origin;
    t = (-offset_- basis_.row(2) * ray_origin)/(basis_.row(2) * ray_direction);
    if (t > 0) {
        if (!ignore_shape && hasShape()) {
            Vector3d projected_pt = ray_origin + t * ray_direction;
            return contains(project(projected_pt.transpose()).transpose());
        } else {
            return true;
        }
    }
    return false;
}

void BoundedPlane::intersectHelper(const Eigen::Vector3d &p, const Eigen::Vector3d &d, MultiRay3d &outRay, double margin) const {
    outRay.o = p;
    outRay.d = d;
    if (hasShape()) {
        Matrix<double, 2, 3> pd;
        pd << p.transpose(), d.transpose();
        Matrix<double, 2, 2> pd2d = project(pd);
        Ray2d ray2d(pd2d.row(0).transpose(), pd2d.row(1).transpose());
        auto intersections = curr_shape_ptr_->intersect(ray2d);
        double intersectionLength = 0;
        if (!intersections.empty() && intersections.size() % 2 == 0) {
            for (size_t j = 0; j < intersections.size(); j += 2) {
                intersectionLength += intersections[j + 1].t - intersections[j].t;
            }
        }
        if (margin > 0) {
            Eigen::Vector2d n(pd2d.row(1).y(), -pd2d.row(1).x());
            for (int i = -1; i <= 1; i += 2) {
                auto offsetIntersections = curr_shape_ptr_->intersect(
                        Ray2d(pd2d.row(0).transpose() + (i * margin) * n, pd2d.row(1).transpose()));
                if (!offsetIntersections.empty() && offsetIntersections.size() % 2 == 0) {
                    double offsetLength = 0;
                    for (int j = 0; j < offsetIntersections.size(); j += 2) {
                        offsetLength += offsetIntersections[j + 1].t - offsetIntersections[j].t;
                    }
                    if (offsetLength > intersectionLength) {
                        intersectionLength = offsetLength;
                        intersections = std::move(offsetIntersections);
                    }
                }
            }
        }
        if (!intersections.empty() && intersections.size() % 2 == 0) {
            for (size_t i = 0; i < intersections.size(); i += 2) {
                outRay.ranges.emplace_back(intersections[i].t, intersections[i + 1].t);
            }
        }
    } else {
        outRay.ranges.emplace_back(std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max());
    }
}

bool BoundedPlane::intersect(const BoundedPlane &other, MultiRay3d &outRay, double margin) const {
    // get point-vector line intersection of planes
    Vector3d p;
    Vector3d d;
    intersect_planes(basis_.row(2), other.basis_.row(2), offset_, other.offset_, p, d);
    MultiRay3d rayThis;
    intersectHelper(p, d, rayThis, margin);
    MultiRay3d rayOther;
    other.intersectHelper(p, d, rayOther, margin);
    outRay = rayThis;
    outRay.ranges = csg1d(rayThis.ranges, rayOther.ranges);
    if (margin > 0) {
        //stitch together ranges closer than margin, and remove ranges smaller than margin
        std::vector<std::pair<double, double>> newRanges;
        newRanges.reserve(outRay.size());
        for (size_t i = 0; i < outRay.size(); ++i) {
            if (i < outRay.size() - 1 && outRay.ranges[i + 1].first - outRay.ranges[i].second < margin) {
                newRanges.emplace_back(outRay.ranges[i].first, outRay.ranges[i + 1].second);
                ++i;
                continue;
            } else {
                newRanges.push_back(outRay.ranges[i]);
            }
        }
        newRanges.erase(std::remove_if(newRanges.begin(), newRanges.end(),
                                       [=](const auto &range) { return range.second - range.first < margin; }),
                        newRanges.end());
        outRay.ranges = std::move(newRanges);
    }
    return !outRay.ranges.empty();

}

void BoundedPlane::addShape(int idx, std::shared_ptr<Primitive> shape) {
    shapes_[idx] = std::move(shape);
}

void BoundedPlane::setCurrentShape(int idx) {
    if (shapes_.find(idx) != shapes_.end()) {
        curr_shape_ptr_ = shapes_[idx].get();
    } else {
        curr_shape_ptr_ = nullptr;
    }
    curr_shape_id_ = idx;
}

void BoundedPlane::clearCurrentShape() {
    curr_shape_ptr_ = nullptr;
}

void BoundedPlane::clearShapes() {

}

int BoundedPlane::getCurrentShape() const {
    return curr_shape_id_;
}

int BoundedPlane::getNumShapes() const {
    return shapes_.size();
}

void BoundedPlane::serialize(std::ostream &o) const {
    Eigen::MatrixXd basisd = basis();
    basisd.resize(1, 9);
    o << "<plane basis=\"" << basisd << "\" offset=\"" << offset() << "\">" << std::endl;
//    if (hasShape())
//        o << *curr_shape_ptr_ << std::endl;
    o << "</plane>";
}

bool BoundedPlane::overlap(const BoundedPlane &other, double threshold, double margin, double offset) const {
    //Eigen::MatrixX2d projected = project(other.points3D());
    Eigen::MatrixX3d points = other.points3D();
    for (int i=0; i < points.rows(); i++) {
        if (contains3D(points.row(i).transpose(), threshold, margin, offset)) {
            return true;
        }
    }
    points = points3D();
    for (int i=0; i < points.rows(); i++) {
        if (other.contains3D((points.row(i) + basis_.row(2) * offset).transpose(), threshold, margin)) {
            return true;
        }
    }
    return false;
}

const Primitive &BoundedPlane::getShape(int idx) const {
    return *(shapes_.find(idx)->second);
}

Primitive &BoundedPlane::getShape(int idx) {
    return *(shapes_.find(idx)->second);
}

bool BoundedPlane::hasShape(int idx) const {
    return shapes_.find(idx) != shapes_.end();
}

bool BoundedPlane::hasShape() const {
    return curr_shape_ptr_ != nullptr;
}

std::ostream &operator<<(std::ostream &o, const BoundedPlane &plane) {
    plane.serialize(o);
    return o;
}


