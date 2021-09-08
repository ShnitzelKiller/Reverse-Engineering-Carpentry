//
// Created by James Noeckel on 4/15/20.
//

#pragma once
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "Ray2d.h"

enum class CurveTypes {
    LINE_SEGMENT=0,
    BEZIER_CURVE=1,
    CIRCULAR_ARC=2
};

struct CurveIntersectionRecord {
    /** distance along ray */
    double t;
    /** curve parameter value */
    double curveT;
};

bool operator<(const CurveIntersectionRecord &a, const CurveIntersectionRecord &b);

class Curve {
public:
    /**
     * Sample a 2D point from the curve at parameter value t in [0, 1]
     * @param t
     * @return
     */
    virtual Eigen::Vector2d sample(double t) const = 0;

    /**
     * Intersect a ray with this curve
     * @param ray
     * @return vector of pairs of (t along ray, t along curve)
     */
    virtual std::vector<CurveIntersectionRecord> intersect(const Ray2d &ray) const = 0;

    /**
     * Sample a sequence of points on the curve, includes the left endpoint and may have fewer than the specified number of points
     * @param maxPts maximum number of points to use for approximating this curve
     * @param minPts minimum number of points to approximate the curve
     * @return Matrix where each row is a 2D point, in increasing order of t
     */
    virtual Eigen::MatrixX2d uniformSample(int maxPts, int minPts=1) const;
    virtual CurveTypes type() const = 0;
    /**
     * Matrix of 2D control points
     */
    //virtual Eigen::Ref<const Eigen::MatrixX2d> points() const = 0;
    //virtual Eigen::Ref<Eigen::MatrixX2d> points() = 0;

    /**
     * Transform the curve with left endpoint as center
     * @param trans positional offset
     * @param ang rotation about first endpoint
     * @param scale scale about fist endpoint
     */
    virtual void transform(const Eigen::Ref<const Eigen::Vector2d> &trans, double ang, double scale) = 0;

    virtual void setEndpoints(const Eigen::Ref<const Eigen::Vector2d> &left, const Eigen::Ref<const Eigen::Vector2d> &right);

    /**
     * Tangent at point t
     */
    virtual Eigen::Vector2d tangent(double t) const = 0;
    /**
     * Curvature at point t (positive means curving towards the right direction along the curve)
     */
    virtual double curvature(double t) const = 0;

    /**
     * t value of the minimum projection of the curve onto the specified direction vector
     * @param direction
     * @return
     */
    virtual double projectedMinPt(const Eigen::Ref<const Eigen::Vector2d> &direction, double &distance) const = 0;
    /**
     * Compute the optimal parameters of a curve interpolating the endpoints of the point range [first, last]
     * @param points matrix whose rows are 2D points on a digitized curve
     * @return total squared error of the points in the range [first, last)
     */
    virtual double
    fit(const Eigen::Ref<const Eigen::MatrixX2d> &points, int first, int last,
            const Eigen::Ref<const Eigen::Vector2d> &leftTangent = Eigen::Vector2d(0, 0),
        const Eigen::Ref<const Eigen::Vector2d> &rightTangent = Eigen::Vector2d(0, 0)) = 0;
    virtual ~Curve() = default;
    virtual std::unique_ptr<Curve> clone() const = 0;
};

class CircularArc : public Curve {
public:
    Eigen::Vector2d sample(double t) const override;

    std::vector<CurveIntersectionRecord> intersect(const Ray2d &ray) const override;

    CurveTypes type() const override;

    Eigen::Vector2d tangent(double t) const override;

    double curvature(double t) const override;

    double projectedMinPt(const Eigen::Ref<const Eigen::Vector2d> &direction, double &distance) const override;

    void transform(const Eigen::Ref<const Eigen::Vector2d> &trans, double ang, double scale) override;

    double fit(const Eigen::Ref<const Eigen::MatrixX2d> &points, int first, int last,
               const Eigen::Ref<const Eigen::Vector2d> &leftTangent=Eigen::Vector2d(0, 0),
               const Eigen::Ref<const Eigen::Vector2d> &rightTangent=Eigen::Vector2d(0, 0)) override;

    std::unique_ptr<Curve> clone() const override;
private:
    Eigen::Vector2d center_;
    double radius_;
    double startAng_, endAng_;
};

class BezierCurve : public Curve {
public:
    explicit BezierCurve(int degree=3);
    /**
     * @param control_points The rows of this matrix are the bezier control points.
     */
    explicit BezierCurve(const Eigen::Ref<const Eigen::Matrix<double, -1, 2>> &control_points);
    int degree() const;
    Eigen::Vector2d sample(double t) const override;
    double fit(const Eigen::Ref<const Eigen::MatrixX2d> &points, int first, int last,
            const Eigen::Ref<const Eigen::Vector2d> &leftTangent = Eigen::Vector2d(0, 0),
               const Eigen::Ref<const Eigen::Vector2d> &rightTangent = Eigen::Vector2d(0, 0)) override;

    std::vector<CurveIntersectionRecord> intersect(const Ray2d &ray) const override;

    void transform(const Eigen::Ref<const Eigen::Vector2d> &trans, double ang, double scale) override;

    void setEndpoints(const Eigen::Ref<const Eigen::Vector2d> &left, const Eigen::Ref<const Eigen::Vector2d> &right) override;

    Eigen::Vector2d tangent(double t) const override;

    /**
     * Split using de casteljau's algorithm
     */
    std::pair<BezierCurve, BezierCurve> split(double t) const;

    double curvature(double t) const override;

    const Eigen::MatrixX2d &points() const;
    Eigen::MatrixX2d &points();

    //both tangents face in curve direction
    void setRightTangent(const Eigen::Ref<const Eigen::Vector2d> &tangent);
    void setLeftTangent(const Eigen::Ref<const Eigen::Vector2d> &tangent);

    double projectedMinPt(const Eigen::Ref<const Eigen::Vector2d> &direction, double &distance) const override;

    std::unique_ptr<Curve> clone() const override;

    CurveTypes type() const override;

private:
    Eigen::Matrix<double, -1, 2> points_;
};

class LineSegment : public Curve {
public:
    LineSegment();
    /**
     * @param points the rows of this matrix are the start and end points
     */
    explicit LineSegment(const Eigen::Ref<const Eigen::Matrix2d> &points);
    Eigen::Vector2d sample(double t) const override;

    std::vector<CurveIntersectionRecord> intersect(const Ray2d &ray) const override;

    void transform(const Eigen::Ref<const Eigen::Vector2d> &trans, double ang, double scale) override;

    void setEndpoints(const Eigen::Ref<const Eigen::Vector2d> &left, const Eigen::Ref<const Eigen::Vector2d> &right) override;

    Eigen::Vector2d tangent(double t) const override;

    double curvature(double t) const override;

    double fit(const Eigen::Ref<const Eigen::MatrixX2d> &points, int first, int last,
            const Eigen::Ref<const Eigen::Vector2d> &leftTangent = Eigen::Vector2d(0, 0),
               const Eigen::Ref<const Eigen::Vector2d> &rightTangent = Eigen::Vector2d(0, 0)) override;

    Eigen::MatrixX2d uniformSample(int maxPts, int minPts=1) const override;

    Eigen::Ref<const Eigen::MatrixX2d> points() const;
    Eigen::Ref<Eigen::MatrixX2d> points();

    double projectedMinPt(const Eigen::Ref<const Eigen::Vector2d> &direction, double &distance) const override;

    std::unique_ptr<Curve> clone() const override;

    CurveTypes type() const override;

private:
    Eigen::Matrix2d points_;
};