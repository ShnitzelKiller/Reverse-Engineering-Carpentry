//
// Created by James Noeckel on 2/12/20.
//

#include "Primitive.h"

#include <utility>
#include "Segment2d.h"
#include <array>
#include <iostream>
#include "utils/exceptions.hpp"

using namespace Eigen;

/*struct IntersectionComp {
    IntersectionComp(const std::vector<std::tuple<double, size_t, double>> &intersections) : intersections_(intersections) {}
    bool operator()(size_t a, size_t b) {

    }
    const std::vector<std::tuple<double, size_t, double>> &intersections_;
};*/

bool operator<(const PrimitiveIntersectionRecord &a, const PrimitiveIntersectionRecord &b) {
    return a.t < b.t;
}

std::vector<std::shared_ptr<Primitive>> Primitive::children() {
    return {};
}

Bbox::Bbox(const Ref<const Vector2d> &minPt, const Ref<const Vector2d> &maxPt) : aa_(minPt), bb_(maxPt) {

}

std::vector<PrimitiveIntersectionRecord> Bbox::intersect(const Ray2d &ray, double margin) const {
    //transform to centered frame
    Array2d p2d_centered = ray.o - 0.5 * (aa_ + bb_);
    Array2d dims = 0.5 * (bb_ - aa_);
    if (margin > 0) {
        dims += margin;
    }
    //compute box intersection
    Array2d d_inv = ray.d.cwiseInverse();
    Array2d n = p2d_centered * d_inv;
    Array2d k = dims * d_inv.abs();
    Array2d pin = -k - n;
    Array2d pout = k - n;
    double t_in = pin.maxCoeff();
    double t_out = pout.minCoeff();
    //TODO: actually implement circumference parameters
    return t_in < t_out ? std::vector<PrimitiveIntersectionRecord>{{t_in, 0, 0., true}, {t_out, 0, 0., false}} : std::vector<PrimitiveIntersectionRecord>();
}

bool Bbox::contains(const Ref<const Vector2d> &point, double margin) const {
    return point[0] >= aa_[0] - margin && point[1] >= aa_[1] - margin && point[0] <= bb_[0] + margin && point[1] <= bb_[1] + margin;
}

MatrixX2d Bbox::points() const {
    Matrix<double, 4, 2> pts;
    pts << aa_.transpose(),
            bb_[0], aa_[1],
            bb_.transpose(),
            aa_[0], bb_[1];
    return pts;
}

/*void Bbox::serialize(std::ostream &o) const {
    o << "<bbox min=\"" << aa_.transpose() << "\" max=\"" << bb_.transpose() << "\"/>";
}*/

CombinedCurve Bbox::curves() const {
    CombinedCurve curves;
    curves.addCurve(std::unique_ptr<Curve>(new LineSegment((Matrix2d() << aa_.transpose(), bb_[0], aa_[1]).finished())));
    curves.addCurve(std::unique_ptr<Curve>(new LineSegment((Matrix2d() << bb_[0], aa_[1], bb_.transpose()).finished())));
    curves.addCurve(std::unique_ptr<Curve>(new LineSegment((Matrix2d() << bb_.transpose(), aa_[0], bb_[1]).finished())));
    curves.addCurve(std::unique_ptr<Curve>(new LineSegment((Matrix2d() << aa_[0], bb_[1], aa_.transpose()).finished())));
    return curves;
}

std::pair<std::vector<std::unique_ptr<Primitive>>, std::vector<std::unique_ptr<Primitive>>>
Bbox::split(const Ray2d &ray) {
    throw NotImplementedException();
}

size_t Bbox::numCurves() const {
    return 4;
}

const Vector2d &Bbox::aa() const {
    return aa_;
}

const Vector2d &Bbox::bb() const {
    return bb_;
}

std::shared_ptr<Primitive> Bbox::clone() const {
    return std::make_shared<Bbox>(*this);
}

void Bbox::transform(const Ref<const Vector2d> &d, double ang, double scale) {
    if (ang == 0) {
        aa_ =  aa_ * scale + d;
        bb_ = bb_ * scale + d;
    } else {
        Rotation2D rot(ang);
        MatrixX2d points;
        points << aa_.transpose() , aa_(0), bb_(1), bb_(0), aa_(1), bb_.transpose();
        for (int i=0; i<4; ++i) {
            points.row(i) = (rot * points.row(i).transpose() * scale + d).transpose();
        }
        aa_ = points.colwise().minCoeff().transpose();
        bb_ = points.colwise().maxCoeff().transpose();
    }
}

double Bbox::area() const {
    return (bb_-aa_).prod();
}

double Bbox::circumference() const {
    return (bb_.y() - aa_.y() + bb_.x() - aa_.x()) * 2;
}


std::vector<std::pair<Interval<double>, Interval<double>>> Polygon::computeIntervals(const Ref<const MatrixX2d> &points) const {
    std::vector<std::pair<Interval<double>, Interval<double>>> intervals;
    for (int i=0; i<points.rows(); i++) {
        double a = points.row(i).dot(axis_);
        double b = points.row((i+1)%points.rows()).dot(axis_);
        if (a == b) continue; //discard zero length projections
        double depth1 = points.row(i).dot(normal_);
        double depth2 = points.row((i+1)%points.rows()).dot(normal_);
        if (b < a) {
            std::swap(a, b);
            std::swap(depth1, depth2);
        }
//        std::cout << "intervals: " << a << ", " << b << "; " << depth1 << ", " << depth2 << std::endl;
        intervals.emplace_back(Interval<double>(a, b), Interval<double>(depth1, depth2));
    }
    return intervals;
}

void Polygon::rebuildTree() {
    Matrix2d cov = points_.transpose() * points_;
    SelfAdjointEigenSolver<Matrix2d> eig(cov);
    axis_ = eig.eigenvectors().col(1);
    normal_ = eig.eigenvectors().col(0);
    auto intervals = computeIntervals(points_);
    tree_.build(intervals.begin(), intervals.end());
}

Polygon::Polygon(const Ref<const MatrixX2d> &points) : points_(points) {
    rebuildTree();
}

bool Polygon::contains(const Ref<const Vector2d> &point, double margin) const {
    double pos = axis_.dot(point);
    double depth = normal_.dot(point);
    auto result = tree_.query(pos);
    int crossings = 0;
    for (const auto &pair : result) {
        double real_depth = (pos - pair.first.start) / (pair.first.end - pair.first.start) *
                            (pair.second.end - pair.second.start) + pair.second.start;
        if (real_depth >= depth) {
            crossings++;
        }
    }
    return crossings % 2 == 1;
}

std::vector<PrimitiveIntersectionRecord> Polygon::intersect(const Ray2d &ray, double margin) const {
    Ray2d infiniteRay = ray;
    infiniteRay.start = std::numeric_limits<double>::lowest();
    infiniteRay.end = std::numeric_limits<double>::max();
    std::vector<PrimitiveIntersectionRecord> hits;
    for (size_t i=0; i<points_.rows(); i++) {
        Vector2d a = points_.row(i).transpose();
        Vector2d b = points_.row((i+1)%points_.rows()).transpose();
        Segment2d segment(a, b);
        double t, t2;
        bool entering;
        if (segment.intersect(infiniteRay, t, t2, entering)) {
            hits.push_back({t, i, t2, entering});
        }
    }
    std::sort(hits.begin(), hits.end());

    for (size_t i=0; i<hits.size(); ++i) {
        hits[i].entering = i % 2 == 0;
    }
    std::vector<PrimitiveIntersectionRecord> newHits;
    newHits.reserve(hits.size());
    for (size_t i=0; i<hits.size(); ++i) {
        if (hits[i].t >= ray.start && hits[i].t < ray.end) {
            newHits.push_back(std::move(hits[i]));
        }
    }
//    auto itStart = std::lower_bound(hits.begin(), hits.end(), ray.start, [](const PrimitiveIntersectionRecord &record, double dist) {return record.t < dist;});
    return newHits;
}

MatrixX2d Polygon::points() const {
    return points_;
}

void Polygon::addPoints(const Ref<const MatrixX2d> &points) {
    MatrixX2d newpoints(points_.rows() + points.rows(), 2);
    newpoints << points_, points;
    points_ = newpoints;
    auto intervals = computeIntervals(points);
    for (const auto &interval : intervals) {
        tree_.insert(interval.first, interval.second);
    }
}

void Polygon::clear() {
    tree_.clear();
    points_.resize(0, 2);
}

//Polygon::~Polygon() {}

/*void Polygon::serialize(std::ostream &o) const {
    o << "<polygon>" << std::endl;
    for (int i=0; i<points_.rows(); i++) {
        o << "<point2 value=\"" << points_.row(i) << "\"/>" << std::endl;
    }
    o << "</polygon>";
}*/

CombinedCurve Polygon::curves() const {
    CombinedCurve curves;
    for (int i=0; i<points_.rows(); i++) {
        curves.addCurve(std::unique_ptr<Curve>(new LineSegment((Matrix2d() << points_.row(i), points_.row((i+1)%points_.rows())).finished())));
    }
    return curves;
}

std::pair<std::vector<std::unique_ptr<Primitive>>, std::vector<std::unique_ptr<Primitive>>>
Polygon::split(const Ray2d &ray) {
    Ray2d raycpy = ray;
    raycpy.start = std::numeric_limits<double>::lowest();
    raycpy.end = std::numeric_limits<double>::max();
    std::vector<PrimitiveIntersectionRecord> intersections = intersect(ray);
    size_t N = intersections.size();
    std::vector<size_t> loop_indices(N);
    std::iota(loop_indices.begin(), loop_indices.end(), 0);
    std::sort(loop_indices.begin(), loop_indices.end(), [&](auto a, auto b) {
        double ta = intersections[a].curveIndex + intersections[a].curveDist / points_.rows();
        double tb = intersections[b].curveIndex + intersections[b].curveDist / points_.rows();
        return ta < tb;
    });
    std::vector<size_t> inv_loop_indices(N);
    for (size_t i=0; i<N; ++i) {
        inv_loop_indices[loop_indices[i]] = i;
    }

    std::array<std::vector<std::vector<size_t>>, 2> contours; //0: left contours 1: right contours

    {
        std::vector<bool> used(N, false);
        for (size_t t=0; t < N; ++t) {
            if (used[t]) continue;
            std::vector<size_t> contour;
            std::cout << "left contour starting at " << t << std::endl;
            size_t ti = t;
            do {
                //push forward
                ++ti;
                std::cout << "t: " << ti << std::endl;
                used[ti] = true;
                contour.push_back(ti);
                //follow contour
                ti = loop_indices[(inv_loop_indices[ti] + 1) % N];
                std::cout << "t: " << ti << std::endl;
                used[ti] = true;
                contour.push_back(ti);
            } while (ti != t);
            contours[0].push_back(std::move(contour));
        }

        for (size_t i=0; i<N; ++i) {used[i] = false;}
        for (size_t rt = 0; rt < N; ++rt) {
            size_t t = N - rt - 1;
            if (used[t]) continue;
            std::vector<size_t> contour;
            std::cout << "right contour starting at " << t << std::endl;
            size_t ti = t;
            do {
                //pull backward
                --ti;
                std::cout << "t: " << ti << std::endl;
                used[ti] = true;
                contour.push_back(ti);
                //follow contour
                ti = loop_indices[(inv_loop_indices[ti] + 1) % N];
                std::cout << "t: " << ti << std::endl;
                used[ti] = true;
                contour.push_back(ti);
            } while (ti != t);
            contours[1].push_back(std::move(contour));
        }
    }

    std::vector<std::unique_ptr<Primitive>> left, right;
    for (size_t s = 0; s < 2; ++s) {
        for (const auto &contour : contours[s]) {
            std::vector<MatrixX2d> allPoints;
            size_t totalRows = 0;
            int lastIndex;
            for (size_t p=0; p < contour.size(); ++p) {
                auto intersection = intersections[contour[p]];
                int index = static_cast<int>(intersection.curveIndex);
                std::cout << "index: " << index << std::endl;
                double fac = intersection.curveDist;
                if (p % 2 == 1) {
                    //find intervening points
                    //std::cout << '(' << lastIndex << '-' << index << ')' << std::endl;
                    int numPoints = index > lastIndex ? index - lastIndex : (index - lastIndex) + points_.rows();
                    std::cout << "num points: " << numPoints << std::endl;
                    MatrixX2d interBlock(numPoints, 2);
                    for (size_t i=0; i<numPoints; ++i) {
                        interBlock.row(i) = points_.row((i+lastIndex+1) % points_.rows());
                    }
                    std::cout << std::endl << interBlock << std::endl;
                    allPoints.push_back(std::move(interBlock));
                    totalRows += numPoints;
                }
                MatrixX2d newPoint = points_.row(index) * (1-fac) + points_.row((index + 1) % points_.rows()) * fac;
                allPoints.push_back(std::move(newPoint));
                totalRows += 1;
                lastIndex = index;
            }
            MatrixX2d allPointsMat(totalRows, 2);
            size_t offset = 0;
            for (const auto &points : allPoints) {
                allPointsMat.block(offset, 0, points.rows(), 2) = points;
                offset += points.rows();
            }
            auto newPrimitive = std::make_unique<Polygon>(allPointsMat);
            if (s == 0) {
                left.push_back(std::move(newPrimitive));
            } else {
                right.push_back(std::move(newPrimitive));
            }
        }
    }
    return std::make_pair(std::move(left), std::move(right));
}

size_t Polygon::numCurves() const {
    return points_.rows();
}

std::shared_ptr<Primitive> Polygon::clone() const {
    return std::make_shared<Polygon>(*this);
}

void Polygon::transform(const Ref<const Vector2d> &d, double ang, double scale) {
    for (size_t i=0; i<points_.rows(); ++i) {
        points_.row(i) = (Rotation2D(ang) * points_.row(i).transpose() * scale + d).transpose();
    }
    rebuildTree();
}

double Polygon::area() const {
    double signedArea = 0.0;
    for (size_t i=0; i<points_.rows(); ++i) {
        size_t j = (i+1)%points_.rows();
        signedArea += points_(i, 0) * points_(j, 1) - points_(i, 1) * points_(j, 0);
    }
    signedArea /= 2;
    return std::abs(signedArea);
}

double Polygon::circumference() const {
    double circ = 0;
    for (size_t i=0; i<points_.rows(); ++i) {
        size_t j = (i+1)%points_.rows();
        circ += (points_.row(j) - points_.row(i)).norm();
    }
    return circ;
}

PolygonWithHoles::PolygonWithHoles(const Ref<const MatrixX2d> &outer_points, std::vector<std::shared_ptr<Primitive>> holes) : Polygon(outer_points), children_(std::move(holes)) {
}

bool PolygonWithHoles::contains(const Ref<const Vector2d> &point, double margin) const {
    int winding_number = 0;
    if (Polygon::contains(point, margin)){
        winding_number++;
    }
    for (const auto &child : children_) {
        if (child->contains(point, margin)) {
            winding_number++;
        }
    }
    return (winding_number % 2) == 1;
}

//PolygonWithHoles::~PolygonWithHoles() {}

/*void PolygonWithHoles::serialize(std::ostream &o) const {
    o << "<polygon>" << std::endl;
    for (int i=0; i<points().rows(); i++) {
        o << "<point2 value=\"" << points().row(i) << "\"/>" << std::endl;
    }
    serializeChildren(o);
    o << "</polygon>";
}*/

/*void PolygonWithHoles::serializeChildren(std::ostream &o) const {
    if (!children_.empty()) {
        o << "<holes>" << std::endl;
        for (const auto &child : children_) {
            child->serialize(o);
            o << std::endl;
        }
        o << "</holes>" << std::endl;
    }
}*/

std::vector<PrimitiveIntersectionRecord>
PolygonWithHoles::intersect(const Ray2d &ray, double margin) const {
    Ray2d infiniteRay = ray;
    infiniteRay.start = std::numeric_limits<double>::lowest();
    infiniteRay.end = std::numeric_limits<double>::max();
    std::vector<PrimitiveIntersectionRecord> intersections = Polygon::intersect(infiniteRay, margin);
    size_t offset = numCurves();
    for (size_t c=0; c<children_.size(); ++c) {
        const auto &child = children_[c];
        std::vector<PrimitiveIntersectionRecord> inner_intersections = child->intersect(infiniteRay, margin);
        for (auto &intersection : inner_intersections) {
            intersection.curveIndex += offset;
        }
        intersections.insert(intersections.end(), inner_intersections.begin(), inner_intersections.end());
        offset += child->numCurves();
    }
    std::sort(intersections.begin(), intersections.end());
    for (size_t i=0; i<intersections.size(); ++i) {
        intersections[i].entering = i % 2 == 0;
    }
    std::vector<PrimitiveIntersectionRecord> newHits;
    newHits.reserve(intersections.size());
    for (size_t i=0; i<intersections.size(); ++i) {
        if (intersections[i].t >= ray.start && intersections[i].t < ray.end) {
            newHits.push_back(std::move(intersections[i]));
        }
    }
//    auto itStart = std::lower_bound(hits.begin(), hits.end(), ray.start, [](const PrimitiveIntersectionRecord &record, double dist) {return record.t < dist;});
    return newHits;
}

std::vector<std::shared_ptr<Primitive>> PolygonWithHoles::children() {
    return children_;
}

std::shared_ptr<Primitive> PolygonWithHoles::clone() const {
    return std::make_shared<PolygonWithHoles>(*this);
}

void PolygonWithHoles::transform(const Ref<const Vector2d> &d, double angle, double scale) {
    Polygon::transform(d, angle, scale);
    for (auto &child : children_) {
        child->transform(d, angle, scale);
    }
}

double PolygonWithHoles::area() const {
    double baseArea = Polygon::area();
    for (const auto &child : children_) {
        baseArea -= child->area();
    }
    return baseArea;
}

/*std::ostream &operator<<(std::ostream &o, const Primitive &p) {
    p.serialize(o);
    return o;
}*/

PolyCurveWithHoles::PolyCurveWithHoles(CombinedCurve curve, std::vector<std::shared_ptr<Primitive>> holes, int resolution) : PolygonWithHoles(
        curve.uniformSample(resolution), std::move(holes)), curve_(std::move(curve)) {

}

/*void PolyCurveWithHoles::serialize(std::ostream &o) const {
    o << "<polycurve>" << std::endl;
    for (int i=0; i<curve_.size(); i++) {
        const auto& subcurve = curve_.getCurve(i);
        if (subcurve.type() == CurveTypes::BEZIER_CURVE) {
            o << "<bezier>" << std::endl;
            for (int j=0; j<4; j++) {
                o << "<point2 value=\"" << subcurve.points().row(j) << "\"/>" << std::endl;
            }
            o << "</bezier>" << std::endl;
        } else if (subcurve.type() == CurveTypes::LINE_SEGMENT) {
            o << "<line>" << std::endl;
            for (int j=0; j<2; j++) {
                o << "<point2 value=\"" << subcurve.points().row(j) << "\"/>" << std::endl;
            }
            o << "</line>" << std::endl;
        }
    }
    serializeChildren(o);
    o << "</polycurve>" << std::endl;
}*/

CombinedCurve PolyCurveWithHoles::curves() const {
    return curve_;
}

std::shared_ptr<Primitive> PolyCurveWithHoles::clone() const {
    return std::make_shared<PolyCurveWithHoles>(*this);
}

//PolyCurveWithHoles::~PolyCurveWithHoles() {}

void PolyCurveWithHoles::transform(const Ref<const Vector2d> &d, double ang, double scale) {
    PolygonWithHoles::transform(d, ang, scale);
    curve_.transform(d, ang, scale);
}
