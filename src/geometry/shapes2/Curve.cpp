//
// Created by James Noeckel on 4/15/20.
//

#include "Curve.h"
#include "graphicsgems/FitCurves.h"
#include <numeric>
#include <algorithm>
#include "Segment2d.h"
#include "math/poly34.h"

#define TWO_TIMES_PI 2 * M_PI

using namespace Eigen;

bool operator<(const CurveIntersectionRecord &a, const CurveIntersectionRecord &b) {
    return a.t < b.t;
}

/** matrices M such that [t^d ... 1] * M * P gives the equation for the curve */
static Matrix4d cubicBezierMatrix = (Matrix4d() <<
        -1, 3, -3, 1,
        3, -6, 3, 0,
        -3, 3, 0, 0,
        1, 0, 0, 0).finished();

static Matrix3d quadraticBezierMatrix = (Matrix3d() <<
        1, -2, 1,
        -2, 2, 0,
        1, 0, 0).finished();

MatrixX2d Curve::uniformSample(int maxPts, int minPts) const {
    MatrixX2d pts(maxPts, 2);
    for (int i=0; i < maxPts; i++) {
        pts.row(i) = sample(static_cast<double>(i) / maxPts).transpose();
    }
    return pts;
}

void Curve::setEndpoints(const Ref<const Vector2d> &left, const Ref<const Vector2d> &right) {
    Vector2d origLeft = sample(0);
    Vector2d trans = left - origLeft;
    Vector2d origDisp = sample(1) - origLeft;
    Vector2d disp = right - left;
    double origAng = std::atan2(origDisp.y(), origDisp.x());
    double newAng = std::atan2(disp.y(), disp.x());
    transform(trans, newAng-origAng, std::sqrt(disp.squaredNorm()/origDisp.squaredNorm()));
}

BezierCurve::BezierCurve(const Ref<const Matrix<double, -1, 2>> &control_points) : points_(control_points) {

}

Vector2d BezierCurve::sample(double t) const {
    return BezierII(degree(), points_, t);
}

void BezierCurve::setRightTangent(const Eigen::Ref<const Eigen::Vector2d> &tangent) {
    if (degree() == 3) {
        points_.row(2) = points_.row(3) - tangent.transpose()/3;
    }
}
void BezierCurve::setLeftTangent(const Eigen::Ref<const Eigen::Vector2d> &tangent) {
    if (degree() == 3) {
        points_.row(1) = points_.row(0) + tangent.transpose()/3;
    }
}

double
BezierCurve::fit(const Ref<const MatrixX2d> &points, int start, int end,
        const Ref<const Vector2d> &leftTangent,
                 const Ref<const Vector2d> &rightTangent) {
    int N = points.rows();
    int nPts = end - start + 1;
    if (nPts < 3) return std::numeric_limits<double>::max();
    int d = degree();
    if (nPts == 3) d = 2;
    std::vector<double> u = ChordLengthParameterize(points, start, end);
    MatrixX2d controlPoints = FitBezier(points, start, end, u, d, leftTangent, rightTangent);
    if (controlPoints.rows() == 0) {
        return std::numeric_limits<double>::max();
    }
//    if (controlPoints.rows() == 4) {
        for (int i = 0; i < 3; i++) {
            u = Reparameterize(points, start, end, u, controlPoints);
            controlPoints = FitBezier(points, start, end, u, d, leftTangent, rightTangent);
        }
//    }
    points_ = controlPoints;
    double totalError = 0.0;
    if (nPts >= 4) {
        for (int i = start; i <= end; i++) {
            Vector2d P = BezierII(d, points_, u[i - start]);
            totalError += (P - points.row(i % N).transpose()).squaredNorm();
        }
    }
    return totalError;
}

BezierCurve::BezierCurve(int degree) : points_(degree+1, 2) {

}

CurveTypes BezierCurve::type() const {
    return CurveTypes::BEZIER_CURVE;
}

std::unique_ptr<Curve> BezierCurve::clone() const {
    return std::make_unique<BezierCurve>(*this);
}

MatrixX2d & BezierCurve::points() {
    return points_;
}

const MatrixX2d & BezierCurve::points() const {
    return points_;
}

double BezierCurve::projectedMinPt(const Ref<const Vector2d> &direction, double &minval) const {
    VectorXd rotpoints = points_ * direction;
    minval = std::numeric_limits<double>::max();
    double t = -1;
    //check endpoints first
    if (rotpoints(0) < rotpoints(3)) {
        minval = rotpoints(0);
        t = 0;
    } else {
        minval = rotpoints(3);
        t = 1;
    }
    //solve for derivative = 0
    if (degree() == 3) {
        Eigen::Vector4d C = (cubicBezierMatrix * rotpoints).reverse();

        double disc = C(2) * C(2) - 3 * C(3) * C(1);
        if (disc >= 0) {
            double sqrtdisc = std::sqrt(disc) / (3 * C(3));
            double c2n = -C(2) / (3 * C(3));
            double roots[2] = {c2n + sqrtdisc, c2n - sqrtdisc};
            for (int i = 0; i < 2; i++) {
                //if t is in (0, 1) and if second derivative is positive
                if (roots[i] > 0 && roots[i] < 1 && 3 * C(3) * roots[i] + C(2) > 0) {
                    double t2 = roots[i] * roots[i];
                    double t3 = roots[i] * t2;
                    double newval = C(3) * t3 + C(2) * t2 + C(1) * roots[i] + C(0);
                    if (newval < minval) {
                        minval = newval;
                        t = roots[i];
                    }
                }
            }
        }
    } else if (degree() == 2) {
        Eigen::Vector3d C = (quadraticBezierMatrix * rotpoints).reverse();

        if (C(2) > 0) {
            double root = -C(1)/(2 * C(2));
            double t2 = root * root;
            double newval = C(2) * t2 + C(1) * root + C(0);
            if (newval < minval) {
                minval = newval;
                t = root;
            }
        }
    }
    return t;
}

Vector2d BezierCurve::tangent(double t) const {
    if (degree() == 3) {
        Eigen::Matrix<double, 4, 2> C = (cubicBezierMatrix * points_).colwise().reverse();
        return ((3 * t * t) * C.row(3) + (2 * t) * C.row(2) + C.row(1)).transpose();
    } else {
        Eigen::Matrix<double, 3, 2> C = (quadraticBezierMatrix * points_).colwise().reverse();
        return ((2 * t) * C.row(2) + C.row(1)).transpose();
    }

}

int BezierCurve::degree() const {
    return points_.rows()-1;
}

std::pair<BezierCurve, BezierCurve> BezierCurve::split(double t) const {
    int N = points_.rows();
    int pyramidSize = N * (N+1)/2;
    MatrixX2d pyramid(pyramidSize, 2);
    pyramid.block(0, 0, N, 2) = points_;
    int prevOffset = 0;
    int offset = N;
    for (int i=degree(); i>=1; i--) {
        for (int j=0; j<i; j++) {
            pyramid.row(offset + j) = (1-t) * pyramid.row(prevOffset + j) + t * pyramid.row(prevOffset + j + 1);
        }
        prevOffset = offset;
        offset += i;
    }
    MatrixX2d leftPoints(N, 2);
    MatrixX2d rightPoints(N, 2);
    offset = 0;
    for (int i=0; i<N; i++) {
        leftPoints.row(i) = pyramid.row(offset);
        rightPoints.row(N-1-i) = pyramid.row(offset + (N-1-i));
        offset += (N-i);
    }
    return {BezierCurve(leftPoints), BezierCurve(rightPoints)};
}

double BezierCurve::curvature(double t) const {
    if (t > 0.5) {
        //TODO: adjust for other degrees
        MatrixX2d subPoints = split(t).first.points();
        //std::cout << "subPoints: " << std::endl << subPoints << std::endl;
        RowVector2d disp = subPoints.row(2) - subPoints.row(3);
        double c = disp.norm();
        RowVector2d n(disp.y(), -disp.x());
        n /= c;
        double perpdist = (subPoints.row(1) - subPoints.row(3)).dot(n);
        return -(static_cast<double>(degree() - 1)/degree()) * (perpdist / (c * c));
    } else {
        MatrixX2d subPoints = split(t).second.points();
        //std::cout << "subPoints: " << std::endl << subPoints << std::endl;
        RowVector2d disp = subPoints.row(1) - subPoints.row(0);
        double c = disp.norm();
        RowVector2d n(disp.y(), -disp.x());
        n /= c;
        double perpdist = (subPoints.row(2) - subPoints.row(0)).dot(n);
        return (static_cast<double>(degree() - 1)/degree()) * (perpdist / (c * c));
    }
}

void BezierCurve::setEndpoints(const Ref<const Vector2d> &left, const Ref<const Vector2d> &right) {
    if (degree() == 3) {
        Vector2d dispLeft = left - points_.row(0).transpose();
        Vector2d dispRight = right - points_.row(points_.rows() - 1).transpose();
        points_.row(1) += dispLeft;
        points_.row(2) += dispRight;
    } else {
        Curve::setEndpoints(left, right);
    }
    points_.row(0) = left.transpose();
    points_.row(points_.rows()-1) = right.transpose();
}

void BezierCurve::transform(const Ref<const Vector2d> &trans, double ang, double scale) {
    Vector2d endpoint = points_.row(0).transpose();
    for (int i=1; i<points_.rows(); ++i) {
        points_.row(i) = (endpoint + Rotation2D(ang) * (points_.row(i) - points_.row(0)).transpose() * scale +
                          trans).transpose();
    }
    points_.row(0) += trans.transpose();
}

std::vector<CurveIntersectionRecord> BezierCurve::intersect(const Ray2d &ray) const {
    Matrix2d rot;
    rot.col(0) = ray.d;
    rot.col(1) = Vector2d(-ray.d.y(), ray.d.x());
    RowVector2d orot = ray.o.transpose() * rot;
    MatrixX2d rotpoints = points_ * rot;
    std::vector<CurveIntersectionRecord> intersections;
    if (degree() == 3) {
        Eigen::Vector4d C = (cubicBezierMatrix * rotpoints.col(1)).reverse();
        C(0) -= orot.y();
        C /= C(3);
        double roots[3];
        int numRoots = SolveP3(roots, C(2), C(1), C(0));
        intersections.reserve(numRoots);
        for (size_t i=0; i<numRoots; i++) {
            //TODO: to make this watertight, would have to use tangents to decide inclusive vs exclusive
            if (roots[i] >= 0 && roots[i] < 1) {
                double t = BezierII(3, rotpoints.col(0), roots[i])(0) - orot.x();
                if (t >= ray.start && t < ray.end) {
                    intersections.push_back({t, roots[i]});
                }
            }
        }
    } else if (degree() == 2) {
        Eigen::Vector3d C = (quadraticBezierMatrix * rotpoints.col(1)).reverse();
        C(0) -= orot.y();
        double disc = C(1) * C(1) - 4 * C(2) * C(0);
        if (disc >= 0) {
            double sqrtdisc = std::sqrt(disc) / (2*C(2));
            double x = -C(1)/(2*C(2));
            double roots[2] = {x + sqrtdisc, x - sqrtdisc};
            for (size_t i=0; i<2; i++) {
                if (roots[i] >= 0 && roots[i] < 1) {
                    double t = BezierII(2, rotpoints.col(0), roots[i])(0) - orot.x();
                    if (t >= ray.start && t < ray.end) {
                        intersections.push_back({t, roots[i]});
                    }
                }
            }
        }
    }
    std::sort(intersections.begin(), intersections.end());
    return intersections;
}

LineSegment::LineSegment(const Ref<const Matrix2d> &points) : points_(points) {

}

Vector2d LineSegment::sample(double t) const {
    return (t * points_.row(1) + (1.0-t) * points_.row(0)).transpose();
}

double
LineSegment::fit(const Ref<const MatrixX2d> &points, int start, int end,
        const Ref<const Vector2d> &leftTangent,
                 const Ref<const Vector2d> &rightTangent) {
    int N = points.rows();
    points_.row(0) = points.row(start % N);
    points_.row(1) = points.row(end % N);
    Vector2d n = (points_.row(1) - points_.row(0)).normalized();
    n = Vector2d(n.y(), -n.x());
    double totalDist = 0.0;
    for (int i=start; i<end; i++) {
        double dist = (points.row(i % N) - points_.row(0)) * n;
        totalDist += dist*dist;
    }
    return totalDist;
}

LineSegment::LineSegment() {

}

CurveTypes LineSegment::type() const {
    return CurveTypes::LINE_SEGMENT;
}

std::unique_ptr<Curve> LineSegment::clone() const {
    return std::make_unique<LineSegment>(*this);
}

Ref<MatrixX2d> LineSegment::points() {
    return points_;
}

Ref<const MatrixX2d> LineSegment::points() const {
    return points_;
}

double LineSegment::projectedMinPt(const Ref<const Vector2d> &direction, double &minVal) const {
    Matrix2d rot;
    rot.col(0) = direction;
    rot.col(1) = Vector2d(-direction.y(), direction.x());
    Matrix2d rotpoints = points_ * rot;
    if (rotpoints(0, 0) < rotpoints(1, 0)) {
        minVal = rotpoints(0, 0);
        return 0;
    } else {
        minVal = rotpoints(1, 0);
        return 1;
    }
}

MatrixX2d LineSegment::uniformSample(int maxPts, int minPts) const {
    MatrixX2d pts(minPts, 2);
    for (int i=0; i<minPts; i++) {
        double t = static_cast<double>(i)/minPts;
        pts.row(i) = sample(t).transpose();
    }
    return pts;
}

void LineSegment::transform(const Ref<const Vector2d> &trans, double ang, double scale) {
    Vector2d endpoint = points_.row(0).transpose();
    points_.row(1) = (endpoint + Rotation2D(ang) * (points_.row(1) - points_.row(0)).transpose() * scale + trans).transpose();
    points_.row(0) += trans.transpose();
}

Vector2d LineSegment::tangent(double t) const {
    return (points_.row(1) - points_.row(0)).transpose();
}

double LineSegment::curvature(double t) const {
    return 0;
}

void LineSegment::setEndpoints(const Ref<const Vector2d> &left, const Ref<const Vector2d> &right) {
    points_.row(0) = left.transpose();
    points_.row(1) = right.transpose();
}

std::vector<CurveIntersectionRecord> LineSegment::intersect(const Ray2d &ray) const {
    Segment2d segment(points_.row(0).transpose(), points_.row(1).transpose());
    double t, t2;
    bool entering;
    std::vector<CurveIntersectionRecord> intersections;
    if (segment.intersect(ray, t, t2, entering)) {
        intersections.push_back({t, t2});
    }
    return intersections;
}

//circular arc

Vector2d CircularArc::sample(double t) const {
    double ang = startAng_ * (1-t) + endAng_ * t;
    return Eigen::Vector2d(radius_ * std::cos(ang), radius_ * std::sin(ang)) + center_;
}

/**
 * the parameter of ang along the arc bounded by ang1 and ang2
 * @param ang1 between 0 and 2pi
 * @param ang2 between -2pi and 2pi
 * @param ang
 * @return
 */
double angleParameter(double ang1, double ang2, double ang) {
    bool swapped = ang1 > ang2;
    if (swapped) std::swap(ang1, ang2);
    if (ang > ang2) {
        ang -= TWO_TIMES_PI;
    } else if (ang < ang1) {
        ang += TWO_TIMES_PI;
    }
    return swapped ? (ang - ang2) / (ang1 - ang2) : (ang - ang1) / (ang2 - ang1);
}


std::vector<CurveIntersectionRecord> CircularArc::intersect(const Ray2d &ray) const {
    Vector2d p = ray.o - center_;
    double pd = p.dot(ray.d);
    double disc = pd*pd + 4 * (radius_ * radius_ - p.dot(p));
    if (disc < 0) {
        return {};
    }
    double sqrdisc = std::sqrt(disc)/2;
    pd = -pd/2;
    double t1 = pd + sqrdisc;
    double t2 = pd - sqrdisc;
    Vector2d disp1 = ray.sample(t1) - center_;
    Vector2d disp2 = ray.sample(t2) - center_;
    double ang1 = std::atan2(disp1.y(), disp1.x());
    double ang2 = std::atan2(disp2.y(), disp2.x());
    std::vector<CurveIntersectionRecord> intersections;
    double param1 = angleParameter(startAng_, endAng_, ang1);
    if (param1 >= 0 && param1 < 1) {
        CurveIntersectionRecord record;
        record.curveT = param1;
        record.t = t1;
        intersections.push_back(record);
    }
    double param2 = angleParameter(startAng_, endAng_, ang2);
    if (param2 >= 0 && param2 < 1) {
        CurveIntersectionRecord record;
        record.curveT = param2;
        record.t = t2;
        intersections.push_back(record);
    }
    return intersections;
}

CurveTypes CircularArc::type() const {
    return CurveTypes::CIRCULAR_ARC;
}

Vector2d CircularArc::tangent(double t) const {
    double ang = startAng_ * (1-t) + endAng_ * t;
    return startAng_ <= endAng_ ? Vector2d(-std::sin(ang), std::cos(ang)) : Vector2d(std::sin(ang), -std::cos(ang));
}

double CircularArc::curvature(double t) const {
    return startAng_ <= endAng_ ? -1./radius_ : 1./radius_;
}

double CircularArc::projectedMinPt(const Ref<const Eigen::Vector2d> &direction, double &distance) const {
    //TODO
    return 0;
}

double CircularArc::fit(const Ref<const Eigen::MatrixX2d> &points, int first, int last,
                        const Ref<const Eigen::Vector2d> &leftTangent, const Ref<const Eigen::Vector2d> &rightTangent) {
    int mid = (first+last)/2;

    Vector2d dir = (points.row(last % points.rows()) - points.row(first % points.rows())).transpose();
    double L = dir.norm();
    Vector2d n(dir.y()/L, -dir.x()/L); //right facing normal
    Vector2d midDisp = (points.row(mid % points.rows()) - points.row(first % points.rows())).transpose();
    double H = midDisp.dot(n);
    double x = midDisp.dot(dir)/L - L/2;
    double L2 = L/2;
    double h = -4 * (L2-x) * (L2+x) / (8*H) + H/2;
    double r = std::sqrt(h*h+L2*L2);
    //TODO: actually optimize, use this as an initialization or do it for more points
    Vector2d center = points.row(first % points.rows()).transpose() + dir/2 + n * h;
    Vector2d disp = points.row(first % points.rows()).transpose() - center;
    double ang1 = std::atan2(disp.y(), disp.x());
    disp = points.row(last % points.rows()).transpose() - center;
    double ang2 = std::atan2(disp.y(), disp.x());
    if (ang1 < 0) {
        ang1 += TWO_TIMES_PI;
        ang2 += TWO_TIMES_PI;
    }
    if (H > 0) {
        while (ang2 < ang1) ang2 += TWO_TIMES_PI;
    } else {
        while (ang2 > ang1) ang2 -= TWO_TIMES_PI;
    }

    double totalError = 0;
    for (int i=first; i<=last; ++i) {
        double err = (points.row(i % points.rows()).transpose() - center).norm() - r;
        totalError += err*err;
    }
    center_ = center;
    radius_ = r;
    startAng_ = ang1;
    endAng_ = ang2;
    return totalError;
}

std::unique_ptr<Curve> CircularArc::clone() const {
    return std::make_unique<CircularArc>(*this);
}

void CircularArc::transform(const Ref<const Eigen::Vector2d> &trans, double ang, double scale) {
    Vector2d endpointToCenter(-radius_ * std::cos(startAng_), -radius_ * std::sin(startAng_));
    Vector2d endpoint = center_ - endpointToCenter;
    endpointToCenter *= scale;
    endpointToCenter = Rotation2D(ang) * endpointToCenter;
    center_ = endpoint + endpointToCenter + trans;

    startAng_ += ang;
    endAng_ += ang;
    if (startAng_ > TWO_TIMES_PI) {
        startAng_ -= TWO_TIMES_PI;
        endAng_ -= TWO_TIMES_PI;
    }
    radius_ *= scale;
}
