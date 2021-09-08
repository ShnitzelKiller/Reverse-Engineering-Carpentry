//
// Created by James Noeckel on 5/4/20.
//

#include "Ray2d.h"

using namespace Eigen;

bool Ray2d::intersect(const Ray2d &other, double &t) const {
    const Vector2d &ab = other.d;
    Vector2d n(ab(1), -ab(0));
    Vector2d oc = o - other.o;
    double x = oc.dot(ab);
    t = -n.dot(oc) / n.dot(d);
    if (t >= start && t <= end) {
        double x_end = x + t * ab.dot(d);
        return x_end >= other.start && x_end < other.end;
    } else return false;
}

Ray2d::Ray2d(Vector2d origin, Vector2d direction, double startp,
             double endp) : o(std::move(origin)), d(std::move(direction)), start(startp), end(endp) {

}

Ray2d::Ray2d(double startp, double endp) : start(startp), end(endp) {

}

Ray2d::Ray2d(const std::pair<Eigen::Vector2d, Eigen::Vector2d> &edge2D) {
    d = edge2D.second-edge2D.first;
    end = d.norm();
    d /= end;
    o = edge2D.first;
    start = 0;
}


Eigen::Vector2d Ray2d::sample(double t) const {
    return o + t * d;
}

Eigen::VectorXd Ray2d::project(const Ref<const Eigen::MatrixX2d> &points) const {
    Eigen::VectorXd ts = ((points.rowwise() - o.transpose()) * d);
    return ts;
}

Eigen::VectorXd Ray2d::orthDist(const Ref<const Eigen::MatrixX2d> &points) const {
    return ((points.rowwise() - o.transpose()) * Vector2d(d.y(), -d.x()));
}

Eigen::VectorXd Ray2d::realDist(const Ref<const Eigen::MatrixX2d> &points) const {
    Eigen::MatrixX2d disps = points.rowwise() - o.transpose();
    Eigen::VectorXd ts = disps * d;
    Eigen::Vector2d n(d.y(), -d.x());
    Eigen::VectorXd dists(points.rows());
    double l = end - start;
    Eigen::Vector2d startpt = sample(start);
    Eigen::Vector2d endpt = sample(end);
    for (size_t i=0; i<points.rows(); ++i) {
        if (ts(i) >= 0 && ts(i) <= l) {
            dists(i) = std::abs(disps.row(i).dot(n));
        } else if (ts(i) < 0) {
            dists(i) = (points.row(i).transpose() - startpt).norm();
        } else {
            dists(i) = (points.row(i).transpose() - endpt).norm();
        }
    }
    return dists;
}

double Ray2d::distBetween(const Ray2d &other) const {
    double t;
    if (intersect(other, t)) {
        return 0;
    }
    double dist00 = realDist(other.sample(other.start).transpose())(0);
    double dist01 = realDist(other.sample(other.end).transpose())(0);
    double dist10 = other.realDist(sample(start).transpose())(0);
    double dist11 = other.realDist(sample(end).transpose())(0);
    return std::min(dist00, std::min(dist01, std::min(dist10, dist11)));
}
