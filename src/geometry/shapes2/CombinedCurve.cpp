//
// Created by James Noeckel on 9/16/20.
//

#include "CombinedCurve.h"
#include "utils/IntervalTree.h"
#include "find_adjacent_ranges.h"
#include "graphicsgems/FitCurves.h"
#include "utils/sorted_data_structures.hpp"
#include <iostream>
#include "math/nonmax_suppression.h"
#include "utils/printvec.h"
#include "math/RunningAverage.h"
#include "utils/macros.h"

#define PARALLEL_THRESHOLD 0.9
#define NUM_SAMPLES 10
using namespace Eigen;

LineConstraint::LineConstraint(const Eigen::Matrix2d &mat, double thresholdt) {
    edge.first = mat.row(0).transpose();
    edge.second = mat.row(1).transpose();
    threshold = thresholdt;
}

CombinedCurve::CombinedCurve() {

}

CombinedCurve::CombinedCurve(const CombinedCurve &other) {
    for (const auto &curve : other.curves_) {
        curves_.push_back(curve->clone());
    }
    startEndIndices_ = other.startEndIndices_;
    constraints_ = other.constraints_;
    knotTypes_ = other.knotTypes_;
}

CombinedCurve::CombinedCurve(CombinedCurve &&other) noexcept {
    curves_ = std::move(other.curves_);
    startEndIndices_ = std::move(other.startEndIndices_);
    constraints_ = std::move(other.constraints_);
    knotTypes_ = std::move(other.knotTypes_);
}

CombinedCurve &CombinedCurve::operator=(const CombinedCurve &other) {
    if (this != &other) {
        for (const auto &curve : other.curves_) {
            curves_.push_back(curve->clone());
        }
        startEndIndices_ = other.startEndIndices_;
        constraints_ = other.constraints_;
        knotTypes_ = other.knotTypes_;
    }
    return *this;
}

CombinedCurve &CombinedCurve::operator=(CombinedCurve &&other) noexcept {
    if (this != &other) {
        curves_ = std::move(other.curves_);
        startEndIndices_ = std::move(other.startEndIndices_);
        constraints_ = std::move(other.constraints_);
        knotTypes_ = std::move(other.knotTypes_);
    }
    return *this;
}

Vector2d curveTangent(const Ref<const MatrixX2d> &points, int index, double &angleDiff, double &totalVariation, bool looping, int first, int last, int ksize) {
    int start = index - ksize;
    int end = index + ksize;
    if (looping) {
        if (start < 0) {
            start += points.rows();
            index += points.rows();
            end += points.rows();
        }
    } else {
        start = std::max(first, start);
        end = std::min(last, end);
        if (end - start + 1 < 3) {
            return (points.row(end % points.rows()) - points.row(start % points.rows())).transpose().normalized();
        }
    }
    BezierCurve wholeCurve;
    BezierCurve leftCurve(2);
    BezierCurve rightCurve(2);
    leftCurve.fit(points, start, index);
    rightCurve.fit(points, index, end);
    wholeCurve.fit(points, start, end);
    std::vector<double> u = ChordLengthParameterize(points, start, end);
    if (wholeCurve.degree() == 3) {
        u = Reparameterize(points, start, end, u, wholeCurve.points());
    }

    double h1, h2;
    Vector2d cord = (wholeCurve.sample(1) - wholeCurve.sample(0)).normalized();
    Vector2d upNorm(-cord.y(), cord.x());
    double t1 = wholeCurve.projectedMinPt(upNorm, h1);
    double t2 = wholeCurve.projectedMinPt(-upNorm, h2);
    totalVariation = std::abs(h1 + h2);

    //std::cout << "curve: \n" << curve.points() << std::endl;
    //angleDiff = std::abs(curve.curvature(u[index - start]));
    angleDiff = std::acos(leftCurve.tangent(0.5).normalized().dot(rightCurve.tangent(0.5).normalized()));
    return wholeCurve.tangent(u[index - start]).normalized();
}

double CombinedCurve::fit(const Ref<const MatrixX2d> &points, double minKnotAngDiff, int max_knots, double bezier_cost, double line_cost, double bezier_weight, int first, int last, int ksize,
                          const Ref<const Vector2d> &leftTangent,
                          const Ref<const Vector2d> &rightTangent,
                          const std::vector<LineConstraint> &edges,
                          const std::vector<MatrixX2d> &projectedNeighbors,
                          double defaultThreshold) {
//    std::vector<double> newthresholds = thresholds;
//    if (thresholds.size() == 1) {
//        newthresholds.resize(edges.size() + projectedNeighbors.size(), thresholds[0]);
//    } else if (thresholds.size() != edges.size() + projectedNeighbors.size()) {
//        std::cout << "invalid threshold list" << std::endl;
//        return std::numeric_limits<double>::max();
//    }
    std::vector<double> newthresholds(edges.size() + projectedNeighbors.size());
    for (int i=0; i<edges.size(); ++i) {
        newthresholds[i] = edges[i].threshold;
    }
    for (int i=edges.size(); i<newthresholds.size(); ++i) {
        newthresholds[i] = defaultThreshold;
    }
    double minCornerAngleCos = std::cos(minKnotAngDiff);
    int minFreeSpan = ksize * 2;
    IntervalTree<int, int> intervalTree;
    std::vector<Ray2d> edgeRays;
    edgeRays.reserve(edges.size());
    for (int i=0; i<edges.size(); ++i) {
        edgeRays.emplace_back(edges[i].edge);
    }
    if (!edges.empty() || !projectedNeighbors.empty())
    {
        std::vector<Matrix2d> edgesT;
        edgesT.reserve(edges.size());
        for (int i=0; i<edges.size(); ++i) {
            Matrix2d mat;
            mat << edges[i].edge.first.transpose(), edges[i].edge.second.transpose();
            edgesT.push_back(std::move(mat));
        }
        std::vector<std::pair<std::pair<int, int>, int>> index_map = find_adjacent_ranges(points, edgesT,
                                                                                          projectedNeighbors,
                                                                                          newthresholds);
        /*for (const auto &pair : index_map) {
            std::cout << '[' << pair.first.first << ", " << pair.first.second << "] -> " << pair.second << "; ";
        }
        std::cout << std::endl;*/
        std::vector<std::pair<Interval<int>, int>> index_intervals;
        index_intervals.reserve(index_map.size());
        for (auto &pair : index_map) {
            int newstart = pair.first.first + 1;
            int newend = pair.first.second;
            if (newstart < newend) {
                index_intervals.emplace_back(Interval<int>(newstart, newend), pair.second);
            }
        }
        std::cout << "index intervals: " << index_intervals.size() << std::endl;
        intervalTree.build(index_intervals.begin(), index_intervals.end());
    }


    bool looping;
    int N;
    {
        int N_tot = points.rows();
        if (last < 0) {
            last = N_tot;
            looping = true;
        } else {
            looping = false;
        }
        if (looping) {
            N = N_tot;
            first = 0;
            last = N - 1;
        } else {
            N = last - first + 1;
        }
    }
    bezier_cost *= N;
    line_cost *= N;
    std::vector<int> knots;

    MatrixX2d alltangents(N, 2);

    if (max_knots < 0 || max_knots > N) {
        max_knots = N;
    }
    {
        std::vector<double> angDiffs(N);
        //sort points in descending order of curvature
        double minVariation = std::numeric_limits<double>::max();
        int minVarKnot = -1;
        for (int i = first; i <= last; i++) {
            const auto &row = alltangents.row(i-first);
            double totalVariation;
            alltangents.row(i-first) = curveTangent(points, i, angDiffs[i - first], totalVariation, looping, first, last, ksize);
            if (totalVariation < minVariation) {
                minVariation = totalVariation;
                minVarKnot = i;
            }
        }
//        std::cout << "angDiffs: " << angDiffs << std::endl;
        //cyclic nonmax suppression
        std::vector<double> angDiffs2(N + 2);
        std::copy(angDiffs.begin(), angDiffs.end(), angDiffs2.begin() + 1);
        angDiffs2[0] = angDiffs2[N];
        angDiffs2[N+1] = angDiffs2[1];
        int numPeaks = nonmax_suppression<double>(angDiffs2.cbegin(), angDiffs2.cend(), angDiffs2.begin());
        //dilate peaks
        {
            std::vector<double> angDiffs2cpy(angDiffs2);
            for (int i = 0; i < angDiffs2.size(); ++i) {
                if (angDiffs2[i] == 0 && ((i > 0 && angDiffs2[i - 1] > 0) || (i < angDiffs2.size() - 1 && angDiffs2[i + 1] > 0))) {
                    angDiffs2cpy[i] = angDiffs[i];
                    ++numPeaks;
                }
            }
            angDiffs2 = std::move(angDiffs2cpy);
        }
        angDiffs2.erase(angDiffs2.begin());
        angDiffs2.resize(N);
        /*std::cout << "angdiffs2: ";
        for (int i=0; i<angDiffs2.size(); ++i) {
            std::cout << "(" << i << ", " << angDiffs2[i] << "), ";
        }
        std::cout << std::endl;*/
        std::vector<int> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        //sort by locally maximal peaks first
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {return angDiffs2[a] > angDiffs2[b];});
        //sort remaining knots by absolute angle
        std::sort(indices.begin() + numPeaks, indices.end(), [&](int a, int b) {return angDiffs[a] > angDiffs[b];});
        if (looping) {
            for (int i = 0; i < N; i++) {
                //skip knots that are inside a constraint interval, force add interval endpoints as knots
                int query1 = indices[i];
                int query2 = indices[i]+N;
                auto results1 = intervalTree.query(query1);
                bool isInsideLine = false;
                bool isEndpoint = false;
                for (auto it=results1.begin(); it != results1.end(); it++) {
                    if (it->second >= 0) isInsideLine = true;
                    if (it->first.start == query1 || it->first.end == query1+1) {
                        isEndpoint = true;
                    }
                }
                if (!isEndpoint) {
                    auto results2 = intervalTree.query(query2);
                    for (auto it=results2.begin(); it != results2.end(); it++) {
                        if (it->second >= 0) isInsideLine = true;
                        if (it->first.start == query2 || it->first.end == query2+1) {
                            isEndpoint = true;
                        }
                    }
                }
                if (isEndpoint) {
                    knots.push_back(indices[i]);
                } else if (knots.size() < max_knots && !isInsideLine) {
                    knots.push_back(indices[i]);
                }
            }
//            int start_knot = indices[indices.size() - 1];
            int start_knot = minVarKnot;
//            if (knots.size() < N)
                knots.push_back(start_knot);
            std::sort(knots.begin(), knots.end());
            auto it = sorted_find(knots, start_knot);
            int new_start = std::distance(knots.begin(), it);
            std::vector<int> new_knots(knots.size());
            for (int i = 0; i < knots.size(); i++) {
                new_knots[i] =
                        knots[(i + new_start) % knots.size()] + ((i + new_start) / knots.size()) * N; //loop around;
            }
            knots = new_knots;
            knots.push_back(knots[0] + N); //close the loop
        } else {
            for (int i = 0; knots.size() < max_knots - 2 && i < N; i++) {
                if (knots.size() > 1 && angDiffs[indices[i]] < minKnotAngDiff) {
                    break;
                }
                if (indices[i] == first || indices[i] == last) continue;
                knots.push_back(indices[i] + first + 1);
            }
            knots.push_back(first);
            knots.push_back(last);
            std::sort(knots.begin(), knots.end());
        }
    }
    int M = knots.size();
    std::cout << "using " << knots.size() << " knots: " << std::endl;
    for (int ind=0; ind<knots.size(); ++ind) {
        std::cout << '[' << ind << "]: " << knots[ind] % points.rows() << ", ";
    }
    std::cout << std::endl;
    //precompute tangents based on local curve fits
    MatrixX2d tangents(M, 2);
    for (int i = 0; i < M; i++) {
        tangents.row(i) = alltangents.row((knots[i] - first) % N);
    }
    bool leftConstrained = leftTangent.squaredNorm() > 0;
    bool rightConstrained = rightTangent.squaredNorm() > 0;
    if (leftConstrained) {
        tangents.row(0) = leftTangent.transpose();
    }
    if (rightConstrained) {
        tangents.row(M-1) = rightTangent.transpose();
    }
    alltangents.resize(0, 2);
    /** backwards_pointers[j][k] is the beginning of the segment of type j ending at k */
    std::vector<std::vector<int>> backwards_pointers(3, std::vector<int>(M));
    /** backwards_types[j][k] is the type of the segment before the segment of type j ending at k */
    std::vector<std::vector<int>> backwards_types(3, std::vector<int>(M));
    /** curve_costs[j][k] is the lowest cost of fitting a spline of type j ending at k
     * Type 0 = line; type 1 = bezier curve with fixed right tangent, type 2 = bezier curve with free right tangent*/
    std::vector<std::vector<double>> curve_costs(3, std::vector<double>(M));
    /** index of constraint used by segment of type j ending at k (max integer if none) */
    std::vector<std::vector<int>> constraint_table(3, std::vector<int>(M));
    std::vector<std::vector<bool>> free_left_table(3, std::vector<bool>(M));
    //precompute curve costs
    {
        //extra tables for looking up previous curve properties during optimization
        std::vector<std::vector<Eigen::Vector2d>> right_tangent_angles(3, std::vector<Eigen::Vector2d>(M));
//        std::vector<std::vector<double>> right_curvatures(3, std::vector<double>(M));

        std::vector<double> segment_costs = {line_cost, bezier_cost, bezier_cost};
        std::vector<double> curve_weights = {1.0, bezier_weight, bezier_weight};
        std::vector<std::unique_ptr<Curve>> curve_types;
        curve_types.emplace_back(new LineSegment);
        curve_types.emplace_back(new BezierCurve);
        //initial conditions
        for (int j=0; j<3; j++) {
            curve_costs[j][0] = 0;
            backwards_pointers[j][0] = 0;
            backwards_types[j][0] = -1;
        }
        for (int k = 1; k < M; k++) {
            for (int j = 0; j < 3; j++) {
                auto &curve = curve_types[std::min(1, j)];
                bool freeRight = j == 2;

                //tabulated optimal values
                double minCost = std::numeric_limits<double>::max();
                int prev_k = -1;
                int prev_j = -1;
                int final_constraint = std::numeric_limits<int>::max();
                bool final_free_left = false;
                Vector2d final_dir;
//                double final_curvature;
                //search among previous endpoint, type, and end behavior for optimal configuration
                for (int j2=0; j2<3; j2++) {
                    for (int k2 = 0; k2 < k; k2++) {
                        double prevCost = curve_costs[j2][k2];
                        if (prevCost >= minCost || (looping && (k % (M-1) == k2 % (M-1)))) {
                            //line cannot start and end in the same place
                            continue;
                        } else {
                            std::vector<int> constraintInds;
                            {
                                auto interval = Interval<int>(knots[k2] + 1, knots[k]);
                                auto result = intervalTree.query(interval);
                                for (auto && it : result) {
                                    constraintInds.push_back(it.second);
                                }
                            }
                            if (knots[k] >= N) {
                                auto interval = Interval<int>(knots[k2] + 1-N, knots[k]-N);
                                auto result = intervalTree.query(interval);
                                for (auto && it : result) {
                                    constraintInds.push_back(it.second);
                                }
                            }
                            if (knots[k2] < N) {
                                auto interval = Interval<int>(knots[k2] + 1+N, knots[k]+N);
                                auto result = intervalTree.query(interval);
                                for (auto && it : result) {
                                    constraintInds.push_back(it.second);
                                }
                            }
                            //cannot cross more than one constraint interval if they are not colinear
                            if (constraintInds.size() > 1) {
                                bool mismatched = false;
                                for (int constraint=0; constraint<constraintInds.size(); ++constraint) {
                                    if (constraintInds[constraint] < 0) {
                                        mismatched = true;
                                        break;
                                    }
                                }
                                if (mismatched) continue;
                                Vector2d n(edgeRays[constraintInds[0]].d.y(), -edgeRays[constraintInds[0]].d.x());
                                double baseOffset = edgeRays[constraintInds[0]].o.dot(n);
                                for (int constraint=1; constraint<constraintInds.size(); ++constraint) {
                                    int constraintInd = constraintInds[constraint];
                                    if (!edgeRays[constraintInd].d.isApprox(edgeRays[constraintInds[0]].d, 1e-6) ||
                                            std::abs(edgeRays[constraintInd].o.dot(n) - baseOffset) >= std::min(newthresholds[0], newthresholds[constraintInd])/100) {
                                        mismatched = true;
                                        break;
                                    }
                                }
                                if (mismatched) continue;
//                                std::cout << "allowing merge of constraints " << constraintInds << std::endl;
                            }
                            /** minimum cumulative cost of polycurve ending with this candidate */
                            double newCost;
                            /** index of the constraint crossed by this candidate (int max if none, line if positive and in range, curve if negative) */
                            int constraintIndex = std::numeric_limits<int>::max();
                            /** whether the left endpoint of this candidate is free */
                            bool freeLeft = false;
                            /** angle of the right end tangent of this curve candidate */
                            //double tangentAngle;
                            //curve must have at least 4 points if a bezier spline
                            //curve ending in fixed tangent cannot continue to line (?)
                            //curve must be sufficiently long to have free endpoints (except ending)
                            //ending of last curve must have fixed tangent if looping
                            int span = knots[k] - knots[k2] + 1;
                            if (j > 0 && span < 4) /* || (j2 == 1 && j == 0) */ continue;
                            // if last curve in looping sequence, just force right endpoint to be fixed, otherwise base it on length since the span up to the ending is arbitrary
                            if (looping && k == M-1) {
                                if (j == 2) continue;
                            } else {
                                if (j != 1 && span < minFreeSpan) continue;
                            }
                            //unconstrained case
                            if (constraintInds.empty()) {

                                /*//look backwards to determine whether the previous segment is a curve constraint
                                bool prevCurveConstrained = false;
                                if (k2 > 0)
                                {
                                    int prevConstraintInd = constraint_table[j2][k2];
                                    prevCurveConstrained = prevConstraintInd < 0;
                                }
                                //only follow free endpoint constrained curve with non-constrained curve
                                if (prevCurveConstrained && j2 != 2) {
                                    continue;
                                }

                                freeLeft = (!leftConstrained && k2 == 0) || (j2 == 0) || prevCurveConstrained;*/
                                if (j == 0) {
                                    freeLeft = true;
                                } else if (k2 == 0) {
                                    if (looping) freeLeft = false;
                                    else freeLeft = !leftConstrained;
                                } else {
                                    freeLeft = j2 != 1 && span >= minFreeSpan;
                                }
                                /*bool freeLeft = (!looping) && k2 == 0;
                                bool freeRight = (!looping) && k == M-1;*/
                                newCost = prevCost + curve->fit(points, knots[k2], knots[k],
                                                                freeLeft ? Vector2d(0, 0)
                                                                         : tangents.row(
                                                                        k2).transpose(),
                                                                freeRight ? Vector2d(0, 0)
                                                                          : -tangents.row(
                                                                        k).transpose()) *
                                                     curve_weights[j] + segment_costs[j];
                                //check if previous free curve tangent is close to this one's, then go with fixed instead
                                if (k2 > 0 && freeLeft && j2 == 2) {
                                    Vector2d currTangent = curve->tangent(0).normalized();
//                                    double currCurvature = curve->curvature(0);
                                    const Vector2d &prevTangent = right_tangent_angles[j2][k2];
//                                    double prevCurvature = right_curvatures[j2][k2];
//                                    double angsin = prevTangent.x() * currTangent.y() - currTangent.x() * prevTangent.y();
                                    double angcos = prevTangent.dot(currTangent);
                                    //TODO: parameter, and use pre-computed angles from data
                                    if (angcos > minCornerAngleCos) continue;
                                }

                            } else {
                                constraintIndex = constraintInds[0];
                                if (constraintIndex < 0) {
                                    if (j == 0) continue;
                                    bool prevCurveSameConstraint = false;
                                    if (k2 > 0) {
                                        prevCurveSameConstraint = constraint_table[j2][k2] == constraintIndex;
                                    }
                                    //check if previous curve belongs to the same curve constraint interval
                                    //if yes, enforce smooth tangent
                                    //if no, enforce free tangent (line segment or free bezier ending)
                                    if (prevCurveSameConstraint) {
                                        if (j2 != 1) continue;
                                    } else if (j2 == 1) continue;
                                    freeLeft = !prevCurveSameConstraint;
                                    //curve constraint
                                    newCost =
                                            prevCost +
                                            curve->fit(points, knots[k2], knots[k],
                                                       freeLeft ? Vector2d(0, 0) : tangents.row(k2).transpose(),
                                                       freeRight ? Vector2d(0, 0) : -tangents.row(k).transpose()) *
                                            curve_weights[j] +
                                            segment_costs[j];
                                } else {
                                    //line constraint
                                    if (j != 0) continue;
                                    //prioritize recording most "hard" constraint
                                    for (auto cc : constraintInds) {
                                        if (!edges[cc].alignable) {
                                            constraintIndex = cc;
                                            break;
                                        }
                                    }
                                    //TODO: use cost based on projected line so that line doesn't change based on range
                                    //TODO: only extrapolate if endpoint is within threshold distance from infinite line
                                    //TODO: use signed distance assuming counter clockwise orientation, only constrain if outside gap is too large
//                                    double dist1 = std::abs(edgeRays[constraintIndex].orthDist(points.row(knots[k2] % points.rows()))(0));
//                                    double dist2 = std::abs(edgeRays[constraintIndex].orthDist(points.row(knots[k] % points.rows()))(0));
//                                    if (dist1 >= threshold || dist2 >= threshold) continue;
                                    newCost = 0;
                                    RunningAverage meanDist;
                                    for (int p=knots[k2]; p<=knots[k]; ++p) {
                                        double dist = edgeRays[constraintIndex].orthDist(points.row(p%points.rows()))(0);
                                        meanDist.add(dist);
                                    }
                                    double meanDistS = meanDist.getScalar();
                                    for (int p=knots[k2]; p<=knots[k]; ++p) {
                                        double dist = edgeRays[constraintIndex].orthDist(points.row(p%points.rows()))(0) - meanDistS;
                                        newCost += dist*dist;
                                    }
                                    newCost *= curve_weights[j];
                                    newCost += segment_costs[j]; //should be free?
                                    newCost += prevCost;
                                }

                            }
                            // disallow inflections TODO: only major inflection
                            // disallow small deviation curves
                            /*if (j == 1 || j == 2) {
                                double curvature0 = curve->curvature(0);
                                double curvature1 = curve->curvature(1);
                                if ((curvature0 > 0 && curvature1 < 0) || (curvature0 < 0 && curvature1 > 0)) {
                                    continue;
                                }
                            }*/

                            Vector2d endTangent = curve->tangent(1).normalized();
//                            double endCurvature = curve->curvature(1);
                            if (newCost < minCost) {
                                minCost = newCost;
                                prev_k = k2;
                                prev_j = j2;
                                final_constraint = constraintIndex;
                                final_free_left = freeLeft;
                                final_dir = endTangent;
//                                final_curvature = endCurvature;
                            }
                        }
                    }
                }
//                if (minCost == std::numeric_limits<double>::max()) {
//                    std::cout << "no good contours found ending at point " << knots[k] << std::endl;
//                }
                curve_costs[j][k] = minCost;
                backwards_pointers[j][k] = prev_k;
                backwards_types[j][k] = prev_j;
                constraint_table[j][k] = final_constraint;
                free_left_table[j][k] = final_free_left;
                right_tangent_angles[j][k] = final_dir;
//                right_curvatures[j][k] = final_curvature;
            }
        }
    }
//    std::cout << "traversing backwards..." << std::endl;
    int end_j = rightConstrained ? 1 : 2;
    int end_k = M-1;
    if (curve_costs[0][end_k] < curve_costs[end_j][end_k]) {
        end_j = 0;
    }
    double minCost = curve_costs[end_j][end_k];
    std::vector<int> endpoints;
    std::vector<int> types;
    std::vector<bool> freeLefts;
    //debug
    std::vector<std::string> debug_strings;
//    std::vector<std::pair<double, double>> debug_curvatures;
    while (end_k > 0) {
        int prev_j = backwards_types[end_j][end_k];
        int prev_k = backwards_pointers[end_j][end_k];
        bool freeLeft = free_left_table[end_j][end_k];
//        std::cout << "start and end k: " << prev_k << ", " << end_k << "; end_j: " << end_j << std::endl;
        if (0 <= prev_k && prev_k < end_k) {
            if (end_j == 0) {
                curves_.emplace_back(new LineSegment);
            } else {
                curves_.emplace_back(new BezierCurve);
            }
            endpoints.push_back(end_k);
            types.push_back(end_j);
            bool freeRight = end_j == 2;
            //std::cout << "left tangent: " << tangents.row(prev_k) << std::endl;
            curves_.back()->fit(points, knots[prev_k], knots[end_k],
                                freeLeft ? Vector2d(0, 0) : tangents.row(prev_k).transpose(),
                                freeRight ? Vector2d(0, 0) : -tangents.row(end_k).transpose());
            freeLefts.push_back(freeLeft);
            int numPts = knots[end_k] - knots[prev_k] + 1;
            //std::cout << "curve points: " << curves_.back()->points() << std::endl;
            //debug print segment type
//            debug_curvatures.emplace_back(curves_.back()->curvature(0), curves_.back()->curvature(1));
            char typechar = end_j == 0 ? '-' : '~';
            debug_strings.emplace_back();
            debug_strings.back() += (freeLeft ? '[' : '(');
            debug_strings.back() += typechar;
            debug_strings.back() += (constraint_table[end_j][end_k] < 0 ? '~' : (constraint_table[end_j][end_k] < edges.size() ? '-' : '.'));
            debug_strings.back() += '{' + std::to_string(numPts) + '}';
            debug_strings.back() += (freeRight ? ']' : ')');
        } else {
            std::cout << "invalid contour found ending at " << knots[end_k] << "!" << std::endl;
            curves_.clear();
            return std::numeric_limits<double>::max();
        }
        end_j = prev_j;
        end_k = prev_k;
    }
    //debug
    for (int i=debug_strings.size()-1; i>=0; i--) {
        std::cout << debug_strings[i] << ", ";
    }
    std::cout << std::endl;
//    for (int i=debug_curvatures.size()-1; i>=0; i--) {
//        std::cout << '[' << debug_curvatures[i].first << ", " << debug_curvatures[i].second << "], ";
//    }
//    std::cout << std::endl;
    //
    std::reverse(curves_.begin(), curves_.end());
    std::reverse(endpoints.begin(), endpoints.end());
    std::reverse(types.begin(), types.end());
    std::reverse(freeLefts.begin(), freeLefts.end());
    {
        //merge start and end lines if possible
        int firstType = types[0];
        int lastType = types[types.size()-1];
//        int firstConstraint = constraint_table[types[0]][endpoints[0]];
//        int lastConstraint = constraint_table[types[types.size() - 1]][endpoints[endpoints.size() - 1]];
//        bool firstLineConstraint = firstConstraint < edges.size() && firstConstraint >= 0;
//        bool lastLineConstraint = lastConstraint < edges.size() && lastConstraint >= 0;
//        if ((!firstLineConstraint && !lastLineConstraint && types[0] == 0 && types[types.size()-1] == 0) ||
//            (firstLineConstraint && firstConstraint==lastConstraint)) {
        if (firstType == 0 && lastType == 0) {
            Vector2d n1 = curves_[0]->tangent(0).normalized();
            Vector2d n2 = curves_[curves_.size()-1]->tangent(0).normalized();
            if (n1.dot(n2) > PARALLEL_THRESHOLD) {
                curves_[0]->setEndpoints(curves_[curves_.size() - 1]->sample(0), curves_[0]->sample(1));
                curves_.erase(curves_.end() - 1);
                endpoints.erase(endpoints.end() - 1);
                types.erase(types.end() - 1);
                freeLefts[0] = freeLefts[freeLefts.size() - 1];
                freeLefts.erase(freeLefts.end() - 1);
            }
        }
    }
    int C = endpoints.size();
    {
        //populate additional structures
        int startPoint = 0;
        for (int k = 0; k < C; ++k) {
            int endPoint = endpoints[k];
            int prevType = types[k>0?k-1:C-1];
            startEndIndices_.emplace_back(knots[startPoint], knots[endPoint]);
            int constraintInd = constraint_table[types[k]][endpoints[k]];
            if (constraintInd >= 0 && constraintInd < edges.size()) {
                constraints_.emplace_back(k, edges[constraintInd]);
            }
            // if right curve is a curve with a fixed left endpoint, or the left curve has a fixed right endpoint
            if ((types[k] > 0 && !freeLefts[k]) || prevType == 1) {
                knotTypes_.push_back(1);
            } else {
                knotTypes_.push_back(0);
            }
            startPoint = endPoint;
        }
    }
    std::cout << "knot types: " << knotTypes_ << std::endl;
    //project constrained lines and their neighboring curve endpoints
    MatrixX2d newEndpoints(C, 2);
    std::vector<bool> hasNewEndpoint(C);
    for (int i=0; i<C; i++) {
        int constraintInd = constraint_table[types[i]][endpoints[i]];
        int constraintIndNext = constraint_table[types[(i+1)%C]][endpoints[(i+1)%C]];

        if (constraintInd >= 0 && constraintInd < edges.size() && constraintIndNext >= 0 && constraintIndNext < edges.size() && constraintInd != constraintIndNext) {
            const Ray2d &edge1 = edgeRays[constraintInd];
            const Ray2d &edge2 = edgeRays[constraintIndNext];
            if (std::abs(edge1.d.dot(edge2.d)) < PARALLEL_THRESHOLD) {
                double t;
                edge1.intersect(edge2, t);
                newEndpoints.row(i) = edge1.sample(t).transpose();
                hasNewEndpoint[i] = true;
            } else {
                hasNewEndpoint[i] = false;
            }
        } else if ((constraintInd < 0 || constraintInd >= edges.size()) && (constraintIndNext < 0 || constraintIndNext >= edges.size())) {
            hasNewEndpoint[i] = false;
        } else {
            const Ray2d &edge = (constraintInd >= 0 && constraintInd < edges.size()) ? edgeRays[constraintInd] : edgeRays[constraintIndNext];
            double t = edge.project(curves_[i]->sample(1).transpose())(0);
            Vector2d newp = edge.sample(t);
            newEndpoints.row(i) = newp.transpose();
            hasNewEndpoint[i] = true;
        }
    }
    for (int i=0; i<C; i++) {
        int prevI = i > 0 ? i-1 : C-1;
        if (hasNewEndpoint[i] || hasNewEndpoint[prevI]) {
            curves_[i]->setEndpoints(hasNewEndpoint[prevI] ? newEndpoints.row(prevI).transpose() : curves_[i]->sample(0), hasNewEndpoint[i] ? newEndpoints.row(i).transpose() : curves_[i]->sample(1));
        }
    }
    return minCost;
}

/*double CombinedCurve::fitConstrained(Ref<MatrixX2d> points, const std::vector<Matrix2d> &edges_mat, const std::vector<MatrixX2d> &projectedNeighbors, double threshold, double knotCurvature, int max_knots, double bezier_cost, double line_cost, double bezier_weight) {
    //MatrixX2d points = points;
    std::vector<double>thresholds(edges_mat.size()+ projectedNeighbors.size(), threshold);
    std::vector<std::pair<std::pair<int, int>, int>> index_map = find_adjacent_ranges(points, edges_mat, projectedNeighbors, thresholds);
//    std::cout << " index map: ";
//    for (const auto &pair : index_map) {
//        std::cout << pair.first.first << ", " << pair.first.second << ", " << '[' << edges_mat[pair.second].row(0) << "]-[" << edges_mat[pair.second].row(1) << ']' << "; ";
//    }
//    std::cout << std::endl;
    if (index_map.empty()) {
        return fit(points, knotCurvature, max_knots, bezier_cost, line_cost, bezier_weight);
    }
    for (int i=0; i<index_map.size(); i++) {
        int startInd = index_map[i].first.second;
        int endInd = index_map[(i+1) % index_map.size()].first.first;
        while (endInd < startInd) {
            endInd += points.rows();
        }
        int N = endInd - startInd + 1;

        RowVector2d a1;
        RowVector2d b1;
        RowVector2d ab1;
        RowVector2d disp1;
        RowVector2d a2;
        RowVector2d b2;
        RowVector2d ab2;
        RowVector2d disp2;

        if (index_map[i].second >= 0) {
            a1 = edges_mat[index_map[i].second].row(0);
            b1 = edges_mat[index_map[i].second].row(1);
            ab1 = (b1 - a1).normalized();
            disp1 = points.row(startInd % points.rows()) - a1;
            points.row(startInd % points.rows()) = a1 + ab1 * (ab1.dot(disp1));
        }
        if (index_map[(i+1)%index_map.size()].second >= 0) {
            a2 = edges_mat[index_map[(i + 1) % index_map.size()].second].row(0);
            b2 = edges_mat[index_map[(i + 1) % index_map.size()].second].row(1);
            ab2 = (b2 - a2).normalized();
            disp2 = points.row(endInd % points.rows()) - a2;
            points.row(endInd % points.rows()) = a2 + ab2 * (ab2.dot(disp2));
        }
        //project neighboring line constraints to coners
        //TODO: try using a different snapping threshold to complete corners?
        if (N < 2 && index_map[i].second >= 0 && index_map[(i+1)%index_map.size()].second >= 0) {
            //project onto both lines to form corner point
            RowVector2d proj1 = a1 + ab1 * (ab1.dot(disp1));
            points.row(startInd % points.rows()) = a2 + ab2 * (ab2.dot(proj1-a2));
        }

    }
    double totalError = 0.0;
    for (int i=0; i<index_map.size(); i++) {
        int startInd = index_map[i].first.second;
        int endInd = index_map[(i+1) % index_map.size()].first.first;
        while (endInd < startInd) {
            endInd += points.rows();
        }
        int N = endInd - startInd + 1;
        if (index_map[i].second < 0) {
            CombinedCurve clusterCurve;
            totalError += clusterCurve.fit(points, knotCurvature, max_knots, bezier_cost, line_cost,
                                           bezier_weight, index_map[i].first.first, index_map[i].first.second);
            combineCurve(std::move(clusterCurve));
        } else {
            Matrix2d linePoints;
            linePoints << points.row(index_map[i].first.first % points.rows()),
                    points.row(index_map[i].first.second % points.rows());
            LineSegment line(linePoints);
            addCurve(std::make_unique<LineSegment>(linePoints));
        }
        if (N > 1) {
            CombinedCurve subCurve;
            totalError += subCurve.fit(points, knotCurvature, max_knots, bezier_cost, line_cost,
                                       bezier_weight, startInd, endInd);
            combineCurve(std::move(subCurve));
        }

    }
    return totalError;
}*/


Vector2d CombinedCurve::sample(double t) const {
    double scaled_t = t * curves_.size();
    int index = static_cast<int>(std::floor(scaled_t));
    index = std::max(0, std::min(static_cast<int>(curves_.size()-1), index));
    return curves_[index]->sample(scaled_t - index);
}

MatrixX2d CombinedCurve::uniformSample(int maxResolution, int minResolution) const {
    MatrixX2d pts(maxResolution * curves_.size(), 2);
    int offset = 0;
    for (int i=0; i<curves_.size(); i++) {
        MatrixX2d subPts = curves_[i]->uniformSample(maxResolution, minResolution);
        pts.block(offset, 0, subPts.rows(), 2) = subPts;
        offset += subPts.rows();
    }
    return pts.block(0, 0, offset, 2);
}

size_t CombinedCurve::size() const {
    return curves_.size();
}

const Curve &CombinedCurve::getCurve(size_t i) const {
    return *(curves_[i]);
}

std::pair<int, int> CombinedCurve::getInterval(size_t i) const {
    return startEndIndices_[i];
}

Curve &CombinedCurve::getCurve(size_t i) {
    return *(curves_[i]);
}

void CombinedCurve::addCurve(std::unique_ptr<Curve> curve) {
    curves_.push_back(std::move(curve));
    knotTypes_.push_back(0);
}

void CombinedCurve::moveVertex(size_t i, const Ref<const Vector2d> &displacement) {
    Curve &curve = getCurve(i);
    Vector2d pt = curve.sample(0);
    pt += displacement;
    curve.setEndpoints(pt, curve.sample(1));

    Curve &prevcurve = getCurve(i > 0 ? i-1 : curves_.size()-1);
    prevcurve.setEndpoints(prevcurve.sample(0), pt);
}

void CombinedCurve::clear() {
    curves_.clear();
    startEndIndices_.clear();
    constraints_.clear();
    knotTypes_.clear();
}

void CombinedCurve::combineCurve(CombinedCurve &&other) {
    for (size_t c=0; c<other.size(); c++) {
        curves_.push_back(std::move(other.curves_[c]));
        startEndIndices_.push_back(other.startEndIndices_[c]);
//        constraints_.push_back
        knotTypes_.push_back(other.knotTypes_[c]);
    }
    other.clear();
}


double CombinedCurve::projectedMinPt(const Eigen::Ref<const Eigen::Vector2d> &direction) const {
    double minDist = std::numeric_limits<double>::max();
    double t;
    size_t N = size();
    for (size_t i=0; i<N; i++) {
        double dist;
        double newt = curves_[i]->projectedMinPt(direction, dist);
        if (dist < minDist) {
            minDist = dist;
            t = (newt + i)/N;
        }
    }
    return t;
}

int CombinedCurve::removeCoplanar(double maxAngle) {
    double mincos = std::cos(maxAngle);
    int numMerges = 0;
    bool found = true;
    while (found) {
        for (int i=0; i<curves_.size(); ++i) {
            auto &curve = curves_[i];
            int iNext = (i+1)%curves_.size();
            auto &nextCurve = curves_[iNext];

            if (curve->type() == CurveTypes::LINE_SEGMENT && nextCurve->type() == CurveTypes::LINE_SEGMENT) {
                Vector2d tangent1 = curve->tangent(1).normalized();
                Vector2d tangent2 = nextCurve->tangent(0).normalized();
                if (tangent1.dot(tangent2) > mincos) {
                    found = true;
                    curve->setEndpoints(curve->sample(0), nextCurve->sample(1));

                    std::cout << "found coplanarity between " << i << " and " << iNext << "; removing" << std::endl;
                    std::cout << "removing curve ("<<curves_.size()<<")" << std::endl;
                    curves_.erase(curves_.begin() + iNext);

                    std::cout << "reassigning constraints" << std::endl;

                    auto it = sorted_find(constraints_, i);
                    auto itNext = sorted_find(constraints_, iNext);
                    bool hasNextConstraint = false;
                    if (itNext != constraints_.end()) {
                        hasNextConstraint = true;
                        if (it == constraints_.end()) {
                            std::cout << "adding constraint " << iNext << " to " << i << std::endl;
                            sorted_insert(constraints_, i, itNext->second);
                            itNext = sorted_find(constraints_, iNext);//update iterator
                        } else if (it->second.alignable && !itNext->second.alignable) {
                            std::cout << "replacing constraint " << i << " with " << iNext << std::endl;
                            it->second = itNext->second;
                        }
                    }
                    for (auto &pair : constraints_) {
                        if (pair.first > iNext) {
                            pair.first -= 1;
                        }
                    }
                    if (hasNextConstraint) {
                        std::cout << "erasing constraint" << std::endl;
                        constraints_.erase(itNext);
                    }

                    std::cout << "removing knot("<<knotTypes_.size()<<")" << std::endl;
                    if (!knotTypes_.empty()) {
                        knotTypes_.erase(knotTypes_.begin() + iNext);
                    }
                    std::cout << "removing startendindex("<<startEndIndices_.size()<<")" << std::endl;
                    if (!startEndIndices_.empty()) {
                        startEndIndices_.erase(startEndIndices_.begin() + iNext);
                    }
                    ++numMerges;
                    break;
                }
            }
        }
        found = false;
    }
    return numMerges;
}

void CombinedCurve::fixKnots(double maxAngle, double displacementThreshold) {
    double mincos = std::cos(maxAngle);
    std::cout << "mincos: " << mincos << std::endl;
    int N=size();
    std::vector<Vector2d> newTangents(N);
    std::vector<bool> hasNewTangent(N, false);
    for (int i=0; i<N; ++i) {
        auto &nextCurve = curves_[i];
        auto &prevCurve = curves_[i > 0 ? i - 1 : N - 1];
        Vector2d prevTangent = prevCurve->tangent(1);
//        double prevTNorm = prevTangent.norm();
        Vector2d nextTangent = nextCurve->tangent(0);
//        double nextTNorm = nextTangent.norm();
        double dotprod = prevTangent.normalized().dot(nextTangent.normalized());
        if (dotprod > mincos) {
            if (prevCurve->type() == CurveTypes::LINE_SEGMENT && nextCurve->type() == CurveTypes::LINE_SEGMENT) {

            } else if (prevCurve->type() == CurveTypes::BEZIER_CURVE && nextCurve->type() == CurveTypes::BEZIER_CURVE) {
                newTangents[i] = (prevTangent + nextTangent).normalized();
                hasNewTangent[i] = true;
            } else if (prevCurve->type() == CurveTypes::LINE_SEGMENT) {
                newTangents[i] = prevTangent.normalized();
                hasNewTangent[i] = true;
            } else if (nextCurve->type() == CurveTypes::LINE_SEGMENT) {
                newTangents[i] = nextTangent.normalized();
                hasNewTangent[i] = true;
            } else {
                std::cout << "skipping meeting of 2 lines" << std::endl;
            }
        } else {
            std::cout << "angle "<< i <<" too large: cos " << dotprod << std::endl;
        }
        if (hasNewTangent[i]) {
            std::cout << "new tangent: " << newTangents[i].transpose() << std::endl;
        }
    }
    std::cout << "applying new tangents" << std::endl;
    for (int i=0; i<N; ++i) {
        std::cout << "curve " << i << std::endl;
        if (curves_[i]->type() == CurveTypes::BEZIER_CURVE) {
            auto *bezierCurve = (BezierCurve *) curves_[i].get();

            if (hasNewTangent[i]) {
                Vector2d tan = bezierCurve->tangent(0);
                double mag = tan.norm();
                Vector2d newtan = newTangents[i] * mag;
                Vector2d cptDispOld = tan / 3;
                Vector2d cptDispNew = newtan / 3;
                if ((cptDispOld - cptDispNew).norm() < displacementThreshold) {
                    std::cout << "changing left tangent "<< i <<" to " << newtan.transpose() << std::endl;
                    bezierCurve->setLeftTangent(newtan);
                } else {
                    std::cout << "left cpt displacement "<< i <<" too large (" << (cptDispOld-cptDispNew).norm() << " vs " << displacementThreshold << ")" << std::endl;
                }
            }
            int iNext = (i+1)%N;
            if (hasNewTangent[iNext]) {
                Vector2d tan = bezierCurve->tangent(1);
                double mag = tan.norm();
                Vector2d newtan = newTangents[iNext] * mag;
                Vector2d cptDispOld = tan / 3;
                Vector2d cptDispNew = newtan / 3;
                if ((cptDispOld - cptDispNew).norm() < displacementThreshold) {
                    std::cout << "changing right tangent "<< i <<" to " << newtan.transpose() << std::endl;
                    bezierCurve->setRightTangent(newtan);
                } else {
                    std::cout << "right cpt displacement "<< i << " too large (" << (cptDispOld-cptDispNew).norm() << " vs " << displacementThreshold << ")" << std::endl;
                }
            }
        }
    }
}

int CombinedCurve::align(double threshold, double angThreshold, const Ref<const Vector2d> &up) {
    std::vector<double> groups;
    return align(threshold, angThreshold, up, groups);
}

int CombinedCurve::align(double threshold, double angThreshold, const Ref<const Vector2d> &up, std::vector<double> &inoutGroups) {
    int N = size();
    std::vector<std::pair<int, double>> horizYvalues;
    std::vector<std::pair<int, double>> vertXvalues;
//    double upAng = std::atan2(up.y(), up.x());
//    Rotation2D rot(upAng);
//    Rotation2D rotInverse = rot.inverse();
    Vector2d upNorm = up.normalized();
    Matrix2d rot;
    rot << upNorm.x(), upNorm.y(), -upNorm.y(), upNorm.x();
    Matrix2d rotinv;
    rotinv << upNorm.x(), -upNorm.y(), upNorm.y(), upNorm.x();
    Vector2d left(-upNorm.y(), upNorm.x());
    //axis align lines, and make colinear where possible
    //if non-line is flat enough, and axis aligned endpoints, convert to line
    for (int i=0; i<N; ++i) {
        if (curves_[i]->type() != CurveTypes::LINE_SEGMENT) {
//            if (knotTypes_[i] == 1 || knotTypes_[(i+1)%N] == 1) continue;
            double h1, h2;
            double t1 = curves_[i]->projectedMinPt(upNorm, h1);
            double t2 = curves_[i]->projectedMinPt(-upNorm, h2);
            double totalVariation = std::abs(h1 + h2);

            t1 = curves_[i]->projectedMinPt(left, h1);
            t2 = curves_[i]->projectedMinPt(-left, h2);
            totalVariation = std::min(totalVariation, std::abs(h1 + h2));

            if (totalVariation >= threshold) {
                continue;
            }
        }
        auto itCurr = sorted_find(constraints_, i);
        if (itCurr != constraints_.end() && !itCurr->second.alignable) continue;

        Vector2d ptA = curves_[i]->sample(0);
        Vector2d ptB = curves_[i]->sample(1);
        Vector2d ptAorig = ptA;
        Vector2d ptBorig = ptB;

        ptA = rot * ptA;
        ptB = rot * ptB;

        Vector2d disp = ptB - ptA;
        double ang = std::abs(std::abs(std::atan2(disp.y(), disp.x())) - M_PI/2);
        Vector2d midpoint = (ptA + ptB)/2;
        bool aligned = false;
        if (M_PI/2 - ang < angThreshold) {
            //horizontal
            horizYvalues.emplace_back(i, midpoint.y());
            aligned = true;
        } else if (ang < angThreshold) {
            //vertical
            vertXvalues.emplace_back(i, midpoint.x());
            aligned = true;
        }
        if (aligned && curves_[i]->type() != CurveTypes::LINE_SEGMENT) {
            curves_[i] = std::make_unique<LineSegment>((Matrix2d() << ptAorig.transpose(), ptBorig.transpose()).finished());
        }
    }
    std::cout << horizYvalues.size() << " horizontal and " << vertXvalues.size() << " vertical" << std::endl;
    int vert=0;
    for (auto hs : {&horizYvalues, &vertXvalues}) {
        //accumulate alignment groups
        std::vector<std::pair<double, std::vector<int>>> groups;
        for (int i=0; i<hs->size(); ++i) {
            bool found=false;
            for (auto &pair : groups) {
                if (std::abs((*hs)[i].second-pair.first) < threshold) {
                    pair.second.push_back((*hs)[i].first);
                    pair.first = ((pair.second.size()-1) * pair.first + (*hs)[i].second)/pair.second.size();
                    found=true;
                    break;
                }
            }
            if (!found) {
                groups.emplace_back((*hs)[i].second, std::vector<int>(1, (*hs)[i].first));
            }
        }

        //snap groups to input groups
        if (vert == 1) {
            for (auto &pair : groups) {
                bool found = false;
                for (double hIn : inoutGroups) {
                    if (std::abs(pair.first-hIn) < threshold) {
                        std::cout << "snapping group to " << hIn << std::endl;
                        pair.first = hIn;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    inoutGroups.push_back(pair.first);
                }
            }
        }

        //set the horiz/vertical offsets
        for (const auto &pair : groups) {
            std::cout << "group " << pair.first << ": ";
            for (int index : pair.second) {
                std::cout << index << ", ";
                auto it = sorted_find(*hs, index);
                it->second = pair.first;
            }
            std::cout << std::endl;
        }
        ++vert;
    }

    /** new right endpoints for ith curve (in rectified frame)*/
    MatrixX2d newEndpoints(N, 2);
    for (int i=0; i<N; ++i) {
        newEndpoints.row(i) = (rot * curves_[i]->sample(1)).transpose();
    }
    std::vector<bool> hasNewEndpoint(N, false);
    for (int j=0; j<horizYvalues.size() + vertXvalues.size(); ++j) {
        bool vertical = j >= horizYvalues.size();
        auto &pair = vertical ? vertXvalues[j - horizYvalues.size()] : horizYvalues[j];
        auto itPrev = sorted_find(constraints_, pair.first > 0 ? pair.first-1 : N-1);
        auto itNext = sorted_find(constraints_, (pair.first+1) % N);
        Vector2d ptA = newEndpoints.row(pair.first > 0 ? pair.first-1 : N-1).transpose();
        Vector2d ptB = newEndpoints.row(pair.first).transpose();
        hasNewEndpoint[pair.first] = true;
        hasNewEndpoint[pair.first > 0 ? pair.first-1 : N-1] = true;
        if (vertical) {
            ptA.x() = pair.second;
            ptB.x() = pair.second;
        } else {
            ptA.y() = pair.second;
            ptB.y() = pair.second;
        }
        Ray2d ray2(Edge2d(ptA, ptB));
        double t;
        if (itPrev != constraints_.end() && !itPrev->second.alignable) {
            Ray2d ray(Edge2d(rot * itPrev->second.edge.first, rot * itPrev->second.edge.second));
            if (std::abs(ray.d.dot(ray2.d)) < PARALLEL_THRESHOLD) {
                ray.intersect(ray2, t);
                ptA = ray.sample(t);
            }
        }
        if (itNext != constraints_.end() && !itNext->second.alignable) {
            Ray2d ray(Edge2d(rot * itNext->second.edge.first, rot * itNext->second.edge.second));
            if (std::abs(ray.d.dot(ray2.d)) < PARALLEL_THRESHOLD) {
                ray.intersect(ray2, t);
                ptB = ray.sample(t);
            }
        }
        newEndpoints.row(pair.first > 0 ? pair.first-1 : N-1) = ptA.transpose();
        newEndpoints.row(pair.first) = ptB.transpose();
    }
    for (int i=0; i<N; ++i) {
        newEndpoints.row(i) = (rotinv * newEndpoints.row(i).transpose()).transpose();
    }
    for (int i=0; i<N; i++) {
        int prevI = i > 0 ? i-1 : N-1;
        if (hasNewEndpoint[i] || hasNewEndpoint[prevI]) {
            curves_[i]->setEndpoints(hasNewEndpoint[prevI] ? newEndpoints.row(prevI).transpose() : curves_[i]->sample(0), hasNewEndpoint[i] ? newEndpoints.row(i).transpose() : curves_[i]->sample(1));
        }
    }
    return horizYvalues.size() + vertXvalues.size();
}

void CombinedCurve::ransac(const Ref<const MatrixX2d> &points, double minDeviationRatio, double minAngleDifference, double threshold, std::mt19937 &random) {
    size_t N = size();
    /** desired probability that an acceptable model is found (if it exists) */
    double prob = 0.99;
    /** probability of a point being an inlier (also acceptance threshold for a model) */
    double w = 0.75;
    /** optimal number of iterations */
    int k = static_cast<int>(std::ceil(std::log(1-prob)/std::log(1-w*w)));

    std::vector<Ray2d> models(N);
    std::vector<bool> hasModel(N);
    for (size_t i=0; i<N; ++i) {
        int nPts = startEndIndices_[i].second - startEndIndices_[i].first + 1;
        if (nPts * w < 2) continue;
//        curves_[i] = std::make_unique<LineSegment>((Matrix2d() << points.row(startEndIndices_[i].first % points.rows()), points.row(startEndIndices_[i].second % points.rows())).finished());
        double bestError = std::numeric_limits<double>::max();
        Ray2d bestModel;
        for (int iter=0; iter<k; ++iter) {
            int a = std::uniform_int_distribution<int>(startEndIndices_[i].first, startEndIndices_[i].second)(random);
            int b = std::uniform_int_distribution<int>(startEndIndices_[i].first, startEndIndices_[i].second - 1)(random);
            if (b == a) ++b;
            Vector2d lineDir = (points.row(a % points.rows()) - points.row(b % points.rows())).transpose();
            Vector2d n(lineDir.y(), -lineDir.x());
            n.normalize();

            std::vector<int> inliers;
            inliers.reserve(nPts);
            for (int j=startEndIndices_[i].first; j <= startEndIndices_[i].second; ++j) {
                double err = std::abs((points.row(j % points.rows()) - points.row(a % points.rows())).dot(n));
                if (err < threshold) {
                    inliers.push_back(j);
                }
            }
            if (inliers.size() > nPts * w) {
                MatrixX2d data(inliers.size(), 2);
                for (int j=0; j<inliers.size(); ++j) {
                    data.row(j) = points.row(inliers[j] % points.rows());
                }
                RowVector2d mean = data.colwise().mean();
                data.rowwise() -= mean;
                Matrix2d cov = (data.transpose() * data)/(inliers.size()-1);
                SelfAdjointEigenSolver<Eigen::MatrixXd> eigs(cov);
                ArrayXd errs = data * eigs.eigenvectors().col(0);
                double totalErr = (errs * errs).sum();
                if (totalErr < bestError) {
                    ArrayXd ts = data * eigs.eigenvectors().col(1);
                    bestModel = Ray2d(mean.transpose(), eigs.eigenvectors().col(1), ts.minCoeff(), ts.maxCoeff());
                    bestError = totalErr;
                    hasModel[i] = true;
                }
            }
        }
        models[i] = bestModel;
    }
    for (size_t i=0; i<N; ++i) {
        if (hasModel[i]) {
            Vector2d ptA = curves_[i]->sample(0);
            Vector2d ptB = curves_[i]->sample(1);
            Vector2d newA = models[i].sample(models[i].project(ptA.transpose())(0));
            Vector2d newB = models[i].sample(models[i].project(ptB.transpose())(0));
            curves_[i] = std::make_unique<LineSegment>((Matrix2d() << newA.transpose(), newB.transpose()).finished());
            auto &nextCurve = curves_[(i+1)%curves_.size()];
            auto &prevCurve = curves_[i > 0 ? i-1 : curves_.size()-1];
            prevCurve->setEndpoints(prevCurve->sample(0), newA);
            nextCurve->setEndpoints(newB, nextCurve->sample(1));
        }
    }
}

bool CombinedCurve::exportPlaintext(std::ostream &o) const {
    o << curves_.size() << std::endl;
    if (!curves_.empty()) {
        o << curves_[0]->sample(0).transpose() << std::endl;
        std::cout << "exporting curve with " << curves_.size() << " segments" << std::endl;
        for (int i=0; i<curves_.size(); ++i) {
            const auto &curve = curves_[i];
            auto curveType = curve->type();
            std::cout << "writing curve " << i << std::endl;
            if (curve->type() == CurveTypes::LINE_SEGMENT) {
                if (i == 0) {
                    std::cout << "first line segment: " << curves_[i]->sample(0).transpose() << ", " << curves_[i]->sample(1).transpose() << std::endl;
                }
                o << "L " << curve->sample(1).transpose() << " " << knotTypes_[i] << std::endl;
            } else if (curve->type() == CurveTypes::BEZIER_CURVE) {
                const auto *bezierCurve = (const BezierCurve *) curve.get();
                if (bezierCurve->degree() == 3) {
                    o << "C " << bezierCurve->points().row(1) << " " << bezierCurve->points().row(2) << " " << bezierCurve->points().row(3) << " " << knotTypes_[i] << std::endl;
                } else if (bezierCurve->degree() == 2) {
                    o << "Q " << bezierCurve->points().row(1) << " " << bezierCurve->points().row(2) << " " << knotTypes_[i] << std::endl;
                } else {
                    std::cout << "invalid degree " << bezierCurve->degree() << std::endl;
                    return false;
                }
            } else {
                std::cout << "arcs unsupported" << std::endl;
                return false;
                //TODO: arcs
            }
        }
        o << constraints_.size() << std::endl;
        for (int i=0; i<constraints_.size(); ++i) {
            o << constraints_[i].first << " " << constraints_[i].second.uniqueId << " " << constraints_[i].second.alignable << " " << constraints_[i].second.tiltAngCos << " " << constraints_[i].second.threshold << " " << constraints_[i].second.edge.first.transpose() << " " << constraints_[i].second.edge.second.transpose() << std::endl;
        }
    }
    return true;
}

bool CombinedCurve::loadPlaintext(std::istream &o, bool constraints) {
    size_t N;
    std::string line;
    std::getline(o, line);
    {
        std::istringstream is_line(line);
        is_line >> N;
        LINE_FAIL("failed reading number of curves");
    }
    std::cout << "loading curve with " << N << " segments (" << line << ')' << std::endl;
    if (N > 0) {
        RowVector2d currPt;
        std::getline(o, line);
        {
            std::istringstream is_line(line);
            is_line >> currPt.x() >> currPt.y();
            LINE_FAIL("failed reading initial point");
        }
        for (size_t i = 0; i < N; ++i) {
            std::getline(o, line);
            {
                std::istringstream is_line(line);
                char type;
                is_line >> type;
                int knotType;
                if (type == 'L') {
                    Matrix2d pts;
                    pts.row(0) = currPt;
                    is_line >> pts(1, 0) >> pts(1, 1) >> knotType;
                    currPt = pts.row(1);
                    curves_.emplace_back(new LineSegment(pts));
                    knotTypes_.push_back(knotType);
                } else if (type == 'C') {
                    MatrixX2d pts(4, 2);
                    pts.row(0) = currPt;
                    is_line >> pts(1, 0) >> pts(1, 1)
                        >> pts(2, 0) >> pts(2, 1)
                        >> pts(3, 0) >> pts(3, 1)
                        >> knotType;
                    currPt = pts.row(3);
                    curves_.emplace_back(new BezierCurve(pts));
                    knotTypes_.push_back(knotType);
                } else if (type == 'Q') {
                    MatrixX2d pts(3, 2);
                    pts.row(0) = currPt;
                    is_line >> pts(1, 0) >> pts(1, 1)
                            >> pts(2, 0) >> pts(2, 1)
                            >> knotType;
                    currPt = pts.row(2);
                    curves_.emplace_back(new BezierCurve(pts));
                    knotTypes_.push_back(knotType);
                } else {
                    std::cout << "invalid curve type " << type << " (" << line << ')' << std::endl;
                    return false;
                }
                LINE_FAIL("failed to parse node " + std::to_string(i));
            }
        }

    }
    if (constraints) {
        int Nconstraints;
        std::getline(o, line);
        {
            std::istringstream is_line(line);
            is_line >> Nconstraints;
            LINE_FAIL("failed to read number of constraints");
        }
        std::cout << "loading " << Nconstraints << " constraints" << std::endl;
        for (int i = 0; i < Nconstraints; ++i) {
            LineConstraint lc;
            int cid;
            int alignable;
            //            o << constraints_[i].first << " " << constraints_[i].second.uniqueId << " " << constraints_[i].second.alignable << " " << constraints_[i].second.tiltAngCos << " " << constraints_[i].second.threshold << " " << constraints_[i].second.edge.first.transpose() << " " << constraints_[i].second.edge.second.transpose() << std::endl;
            std::getline(o, line);
            {
                std::istringstream is_line(line);
                is_line >> cid >> lc.uniqueId >> alignable >> lc.threshold >> lc.edge.first.x() >> lc.edge.first.y()
                        >> lc.edge.second.x() >> lc.edge.second.y();
                LINE_FAIL("failed to read constraint " + std::to_string(i));
            }
            lc.alignable = alignable;
            constraints_.emplace_back(cid, std::move(lc));
        }
    }
    return true;
}

void CombinedCurve::exportSVG(std::ostream &o) const {
    if (curves_.empty()) return;
    o << "<path ";
    o << "d=\"";
    o << "M " << curves_[0]->sample(0).transpose() << " ";
    for (const auto &curve : curves_) {
        if (curve->type() == CurveTypes::LINE_SEGMENT) {
            o << "L " << curve->sample(1).transpose() << " ";
        } else if (curve->type() == CurveTypes::BEZIER_CURVE) {
            const auto *bezierCurve = (const BezierCurve *) curve.get();
            if (bezierCurve->degree() == 3) {
                o << "C " << bezierCurve->points().row(1) << ", " << bezierCurve->points().row(2) << ", "
                  << bezierCurve->points().row(3) << " ";
            } else if (bezierCurve->degree() == 2) {
                o << "Q " << bezierCurve->points().row(1) << ", " << bezierCurve->points().row(2) << " ";
            } else {
                std::cout << "invalid degree " << bezierCurve->degree() << std::endl;
            }
        } else {
            std::cout << "arcs unsupported" << std::endl;
            //TODO: arcs
        }
    }
    o << "Z\"/>";
}

void CombinedCurve::transform(const Ref<const Vector2d> &d, double ang, double scale) {
    Rotation2D rotation(ang);
    for (auto &curve : curves_) {
        Vector2d endpoint = curve->sample(0);
        Vector2d newTrans = d - endpoint + rotation * endpoint * scale;
        curve->transform(newTrans, ang, scale);
    }
}