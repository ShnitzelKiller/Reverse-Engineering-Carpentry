//
// Created by James Noeckel on 9/16/20.
//

#pragma once
#include <Eigen/Dense>
#include "Curve.h"
#include <random>
#include "utils/typedefs.hpp"

struct LineConstraint {
    int uniqueId;
    Edge2d edge;
    double threshold;
    bool alignable;
    double tiltAngCos=0;
    LineConstraint() = default;
    LineConstraint(const Eigen::Matrix2d &mat, double threshold);
};

class CombinedCurve {
public:
    CombinedCurve();
    ~CombinedCurve() = default;

    CombinedCurve(const CombinedCurve &other);

    CombinedCurve(CombinedCurve &&other) noexcept;

    CombinedCurve &operator=(const CombinedCurve &other);

    CombinedCurve &operator=(CombinedCurve &&other) noexcept;

    /*template <class InputIterator>
    CombinedCurve(InputIterator it, InputIterator end) {
        for (auto itt=it; itt != end; ++itt) {
            curves_.push_back(std::move(*it));
        }
    }*/

    

    /**
     *
     * @param points
     * @param minKnotAngDiff
     * @param max_knots
     * @param bezier_cost
     * @param line_cost
     * @param bezier_weight
     * @return
     */
    double fit(const Eigen::Ref<const Eigen::MatrixX2d> &points, double minKnotAngDiff=0.0, int max_knots=-1, double bezier_cost=0.0, double line_cost=0.0, double bezier_weight=1.0, int first=0, int last=-1, int ksize=5,
               const Eigen::Ref<const Eigen::Vector2d> &leftTangent = Eigen::Vector2d(0, 0),
               const Eigen::Ref<const Eigen::Vector2d> &rightTangent = Eigen::Vector2d(0, 0), const std::vector<LineConstraint> &edges={}, const std::vector<Eigen::MatrixX2d> &projectedNeighbors={}, double defaultThreshold=0);

    /** legacy fitting method (fit in unconstrained ranges) */
//    double fitConstrained(Eigen::Ref<Eigen::MatrixX2d> points, const std::vector<Eigen::Matrix2d> &edges, const std::vector<Eigen::MatrixX2d> &projectedNeighbors, double threshold, double knotCurvature=0.0, int max_knots=-1, double bezier_cost=0.0, double line_cost=0.0, double bezier_weight=1.0);

    void ransac(const Eigen::Ref<const Eigen::MatrixX2d> &points, double minDeviationRatio, double minAngleDifference, double threshold, std::mt19937 &random);

    int align(double threshold, double angThreshold, const Eigen::Ref<const Eigen::Vector2d> &up, std::vector<double> &groups);
    int align(double threshold, double angThreshold, const Eigen::Ref<const Eigen::Vector2d> &up);

    /**
     * Sample this curve with a parameter from 0 to 1
     */
    Eigen::Vector2d sample(double t) const;
    Eigen::MatrixX2d uniformSample(int maxResolution, int minResolution=1) const;
    size_t size() const;
    const Curve &getCurve(size_t i) const;
    Curve &getCurve(size_t i);
    /** get inclusive interval of point indices to which curve i was fitted */
    std::pair<int, int> getInterval(size_t i) const;
    void addCurve(std::unique_ptr<Curve> curve);
    /** move the left endpoint of curve i by the specified amount */
    void moveVertex(size_t i, const Eigen::Ref<const Eigen::Vector2d> &displacement);
    void combineCurve(CombinedCurve &&other);
    double projectedMinPt(const Eigen::Ref<const Eigen::Vector2d> &direction) const;
    void fixKnots(double maxAngle, double displacementThreshold=std::numeric_limits<double>::max());
    int removeCoplanar(double maxAngle);
    void clear();
    bool exportPlaintext(std::ostream &o) const;
    bool loadPlaintext(std::istream &o, bool constraints=true);
    void exportSVG(std::ostream &o) const;
    void transform(const Eigen::Ref<const Eigen::Vector2d> &d, double ang, double scale);

    /** map of curve indices to line constraints (as 2D line segments) */
    std::vector<std::pair<int, LineConstraint>> constraints_;
private:
    std::vector<std::unique_ptr<Curve>> curves_;

    /** inclusive intervals in the data to which each curve was fit */
    std::vector<std::pair<int, int>> startEndIndices_;

    /** type of knot at the left endpoint of each curve. 0=free, 1=constrained */
    std::vector<int> knotTypes_;

};

/**
 * Use a bezier curve fitted to the 5-point neighborhood to determine the tangent
 * @param points
 * @param index index of row in points to compute the tangent
 * @param looping whether to consider this a closed loop contour
 * @param first start index
 * @param last end index
 * @return normalized tangent vector
 */
Eigen::Vector2d curveTangent(const Eigen::Ref<const Eigen::MatrixX2d> &points, int index, double &angleDiff, double &totalVariation, bool looping, int first=0, int last=0, int ksize=5);

