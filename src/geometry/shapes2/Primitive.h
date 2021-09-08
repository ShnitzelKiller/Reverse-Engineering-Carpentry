//
// Created by James Noeckel on 2/12/20.
//

#pragma once

#include <Eigen/Dense>
#include <vector>
#include "CombinedCurve.h"
#include "Ray2d.h"
#include "utils/IntervalTree.h"
#include <ostream>
#include <eigen3/Eigen/src/Core/Matrix.h>

struct PrimitiveIntersectionRecord {
    /** distance along ray */
    double t;
    /** curve index */
    size_t curveIndex;
    /** curve parameter */
    double curveDist;
    bool entering;
};

bool operator<(const PrimitiveIntersectionRecord &a, const PrimitiveIntersectionRecord &b);

class Primitive {
public:
    /**
     * Intersect 2d ray with this bbox
     * @param ray_origin
     * @param ray_direction
     * @param margin margin around the bbox for detecting intersection
     * @return vector of tuples of (hit distance, hit edge, interpolation factor along edge where hit occurred)
     */
    virtual std::vector<PrimitiveIntersectionRecord> intersect(const Ray2d &ray, double margin = 0) const = 0;

    virtual size_t numCurves() const = 0;

    /**
     * Check if point is inside this 2D primitive
     * @param point
     * @param margin how far outside the boundary is still considered inside
     */
    virtual bool contains(const Eigen::Ref<const Eigen::Vector2d> &point, double margin=0) const = 0;

    virtual std::vector<std::shared_ptr<Primitive>> children();

    /**
     * Return a matrix whose rows are points of the primitive
     */
    virtual Eigen::MatrixX2d points() const = 0;

    virtual CombinedCurve curves() const = 0;

    /**
     * Split this primitive into two groups of primitives; those on the left along the direction of the ray,
     * and those on the right
     * @param ray
     * @return
     */
    virtual std::pair<std::vector<std::unique_ptr<Primitive>>, std::vector<std::unique_ptr<Primitive>>> split(const Ray2d &ray) = 0;

    //virtual void serialize(std::ostream &o) const = 0;

    virtual std::shared_ptr<Primitive> clone() const = 0;

    virtual ~Primitive() = default;

    /** transform the primitive
     * @param ang counter-clockwise rotation */
    virtual void transform(const Eigen::Ref<const Eigen::Vector2d> &trans, double ang, double scale) = 0;

    virtual double area() const = 0;
    virtual double circumference() const = 0;
};

class Bbox : public Primitive {
public:
    /**
     * Define an axis aligned bouding box with the specified corners
     * @param minPt corner with minimum coordinates
     * @param maxPt corner with maximum coordinates
     */
    Bbox(const Eigen::Ref<const Eigen::Vector2d> &minPt, const Eigen::Ref<const Eigen::Vector2d> &maxPt);
    std::vector<PrimitiveIntersectionRecord> intersect(const Ray2d &ray, double margin = 0) const override;

    size_t numCurves() const override;

    std::pair<std::vector<std::unique_ptr<Primitive>>, std::vector<std::unique_ptr<Primitive>>>
    split(const Ray2d &ray) override;

    bool contains(const Eigen::Ref<const Eigen::Vector2d> &point, double margin=0) const override;
    //void serialize(std::ostream &o) const override;
    Eigen::MatrixX2d points() const override;

    const Eigen::Vector2d &aa() const;
    const Eigen::Vector2d &bb() const;

    CombinedCurve curves() const override;

    std::shared_ptr<Primitive> clone() const override;

    void transform(const Eigen::Ref<const Eigen::Vector2d> &d, double ang, double scale) override;

    double area() const override;
    double circumference() const override;

private:
    Eigen::Vector2d aa_;
    Eigen::Vector2d bb_;
};

class Polygon : public Primitive {
public:
    /**
     * Defines a polygon with the given points (orientation does not matter)
     * @param points an Nx2 matrix whose rows are sequential points on the boundary
     */
    explicit Polygon(const Eigen::Ref<const Eigen::MatrixX2d> &points);
    bool contains(const Eigen::Ref<const Eigen::Vector2d> &point, double margin=0) const override;
    std::vector<PrimitiveIntersectionRecord> intersect(const Ray2d &ray, double margin = 0) const override;

    size_t numCurves() const override;

    std::pair<std::vector<std::unique_ptr<Primitive>>, std::vector<std::unique_ptr<Primitive>>>
    split(const Ray2d &ray) override;

    Eigen::MatrixX2d points() const override;

    /**
     * Add another boundary to the polygon. Can act as a hole, since it inverts inside/outside area within the new shape.
     * @param points new points, interpreted as a separate closed chain of edges
     */
    void addPoints(const Eigen::Ref<const Eigen::MatrixX2d> &points);

    CombinedCurve curves() const override;

    void clear();
    //void serialize(std::ostream &o) const override;

    std::shared_ptr<Primitive> clone() const override;

    void transform(const Eigen::Ref<const Eigen::Vector2d> &d, double ang, double scale) override;

    double area() const override;
    double circumference() const override;

//    ~Polygon() override;
private:
    Eigen::MatrixX2d points_;
    IntervalTree<double, Interval<double>> tree_;
    Eigen::Vector2d axis_;
    Eigen::Vector2d normal_;
    std::vector<std::pair<Interval<double>, Interval<double>>> computeIntervals(const Eigen::Ref<const Eigen::MatrixX2d> &points) const;
    void rebuildTree();
};

class PolygonWithHoles : public Polygon {
public:
    explicit PolygonWithHoles(const Eigen::Ref<const Eigen::MatrixX2d> &outer_points, std::vector<std::shared_ptr<Primitive>> holes={});
    bool contains(const Eigen::Ref<const Eigen::Vector2d> &point, double margin=0) const override;

    std::vector<PrimitiveIntersectionRecord>
    intersect(const Ray2d &ray, double margin) const override;

    std::vector<std::shared_ptr<Primitive>> children() override;

    //void serialize(std::ostream &o) const override;

    std::shared_ptr<Primitive> clone() const override;

    void transform(const Eigen::Ref<const Eigen::Vector2d> &d, double ang, double scale) override;

    double area() const override;

//    ~PolygonWithHoles() override;
protected:
    //void serializeChildren(std::ostream &o) const;
private:
    std::vector<std::shared_ptr<Primitive>> children_;
};

class PolyCurveWithHoles : public PolygonWithHoles {
public:
    explicit PolyCurveWithHoles(CombinedCurve curve, std::vector<std::shared_ptr<Primitive>> holes={}, int resolution=20);
    //void serialize(std::ostream &o) const override;

    CombinedCurve curves() const override;

    std::shared_ptr<Primitive> clone() const override;

    void transform(const Eigen::Ref<const Eigen::Vector2d> &d, double ang, double scale) override;

//    ~PolyCurveWithHoles() override;
private:
    CombinedCurve curve_;
};

std::ostream &operator<<(std::ostream &o, const Primitive &p);