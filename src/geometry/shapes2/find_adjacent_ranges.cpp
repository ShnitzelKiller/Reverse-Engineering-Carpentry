//
// Created by James Noeckel on 4/23/20.
//

#include "find_adjacent_ranges.h"
#include "Ray2d.h"
#include <iostream>
//#include "utils/printvec.h"
using namespace Eigen;
/** find the start and end indices (looping) of the smallest range containing all true elements of bitvector */
std::pair<int, int> smallest_looping_range(const std::vector<bool> &bitvector) {
    int N = bitvector.size();
    int max_run_length = 0;
    int max_run_start = -1;
    int curr_run_length = 0;
    int curr_run_start = 0;

    for (int p=0; p<N*2; ++p) {
        if (bitvector[p % N]) {
            if (curr_run_length > max_run_length) {
                max_run_length = curr_run_length;
                max_run_start = curr_run_start;
            }
            curr_run_start = p + 1;
            curr_run_length = 0;
        } else {
            ++curr_run_length;
            if (curr_run_length >= N) {
                return {0, -1};
            }
        }
    }

    int first = (max_run_start + max_run_length) % N;
    int last = first + (N - (max_run_length+1));
    return {first, last};
}

std::vector<std::pair<std::pair<int, int>, int>> find_adjacent_ranges(const Ref<const MatrixX2d> &contour, const std::vector<Matrix2d> &edges, const std::vector<MatrixX2d> &projectedNeighbors, const std::vector<double> &thresholds) {
    //(start, end, edgeIndex)
    std::vector<double> thresholds2(thresholds.size());
    for (size_t i=0; i<thresholds.size(); ++i) {
        thresholds2[i] = thresholds[i] * thresholds[i];
    }
    std::vector<std::pair<std::pair<int, int>, int>> ranges;
//    int curveConstraints = 0;
    for (int i=0; i<projectedNeighbors.size(); i++) {
        const MatrixX2d &neighbor = projectedNeighbors[i];
        //std::cout << "checking neighbor " << i << "(size " << neighbor.rows() << ')' << std::endl;
        std::vector<bool> bitvector(contour.rows(), false);
        for (int p=0; p<contour.rows(); ++p) {
            double dist2;
            for (int p2=0; p2<neighbor.rows(); p2++) {
                dist2 = (neighbor.row(p2) - contour.row(p)).squaredNorm();
                if (dist2 <= thresholds2[i + edges.size()]) {
                    bitvector[p] = true;
                    break;
                }
            }
        }
        auto range = smallest_looping_range(bitvector);
        if (range.first <= range.second) {
//            curveConstraints++;
            ranges.emplace_back(range, -1-i);
        }
    }
    for (int i=0; i<edges.size(); i++) {
        const auto &edge = edges[i];
        RowVector2d ab = (edge.row(1)-edge.row(0)).transpose();
        double lab = ab.norm();
        ab /= lab;
        RowVector2d n(ab.y(), -ab.x()); //outward normal
        std::vector<bool> bitvector(contour.rows());
        for (int p=0; p<contour.rows(); p++) {
            RowVector2d disp = (contour.row(p) - edge.row(0));
            double s = disp.dot(ab);
            double signedDist = disp.dot(n);
//            double dist = std::fabs(signedDist);
//            std::cout << "dist " << p << ": " << dist << "; snorm " << p << ": " << s/lab << std::endl;
            bitvector[p] = s >= 0.0 && s <= lab && (signedDist > -thresholds[i]);
        }
//        std::cout << "bitvector " << i << ": " << bitvector << std::endl;
        auto range = smallest_looping_range(bitvector);
        if (range.first <= range.second) {
            ranges.emplace_back(range, i);
        }
    }
    std::sort(ranges.begin(), ranges.end());
    //fix overlaps
    for (size_t i=1; i<ranges.size()+1; i++) {
        auto &rightRange = ranges[i % ranges.size()];
        auto &leftRange = ranges[i-1];
        int upper_bound = rightRange.first.first + static_cast<int>((i/ranges.size()) * contour.rows());
        if (leftRange.first.second > upper_bound) {
            int overlap = (leftRange.first.second - upper_bound);
            int offset1 = overlap/2;
            int offset2 = overlap - offset1;
            leftRange.first.second -= offset1;
            rightRange.first.first += offset2;
        }
    }
    //DEBUG
    std::cout << "N=" << contour.rows() << std::endl;
    for (const auto &pair : ranges) {
        std::cout << '[' << pair.first.first << ", " << pair.first.second;
        if (pair.first.second > contour.rows()) {
            std::cout << " (" << pair.first.second % contour.rows() << ')';
        }
        std::cout << "] -> " << pair.second << "; ";
    }
    /*std::cout << std::endl;
    for (int i=0; i<ranges.size(); i++) {
        auto &rightRange = ranges[(i+1) % ranges.size()];
        auto &leftRange = ranges[i];
        int upper_bound = rightRange.first.first + ((i+1)/ranges.size()) * contour.rows();
        if (leftRange.first.second > upper_bound) {
            std::cout << "warning: range " << i << " overlaps range " << (i+1) % ranges.size() << std::endl;
        }
    }*/
    /*for (const auto & range : ranges) {
        std::cout << "range: " << range.first.first << ", " << range.first.second << std::endl;
    }*/
    //
    ranges.erase(std::remove_if(ranges.begin(), ranges.end(),
            [&](const std::pair<std::pair<int, int>, int> &a)
            {
        if (a.second < 0) {
            //cluster-based curve fitting must have over 4 points
            return a.first.second <= a.first.first + 5;
        } else {
            return a.first.second <= a.first.first;
        }
            }), ranges.end());
    //std::cout << "used " << curveConstraints << '/' << projectedNeighbors.size() << " curve constraints" << std::endl;
    //"stitch" together connected line constraints
    for (size_t i=1; i<ranges.size()+1; i++) {
        auto &rightRange = ranges[i % ranges.size()];
        auto &leftRange = ranges[i-1];
        if (leftRange.second >= 0 && rightRange.second >= 0) {
            const double &leftThreshold = thresholds[leftRange.second];
            const double &rightThreshold = thresholds[rightRange.second];
            int upper_bound = rightRange.first.first + static_cast<int>((i / ranges.size()) * contour.rows());
            if (leftRange.first.second < upper_bound) {
                auto leftEdge = std::pair<Vector2d, Vector2d>(edges[leftRange.second].row(0).transpose(),
                                                              edges[leftRange.second].row(1).transpose());
                auto rightEdge = std::pair<Vector2d, Vector2d>(edges[rightRange.second].row(0).transpose(),
                                                               edges[rightRange.second].row(1).transpose());
                Ray2d leftRay(leftEdge);
                Ray2d rightRay(rightEdge);
                double distBetween = leftRay.distBetween(rightRay);
                if (distBetween < std::max(leftThreshold, rightThreshold)) {
                    double t;
                    leftRay.intersect(rightRay, t);
                    Vector2d intersectionPoint = leftRay.sample(t);
//                    std::cout << "intersection point: " << intersectionPoint.transpose() << std::endl;
                    if (leftRay.realDist(intersectionPoint.transpose())(0) < leftThreshold && rightRay.realDist(intersectionPoint.transpose())(0) < rightThreshold) {
//                        std::cout << "stitching " << i - 1 << " with " << i % ranges.size() << std::endl;
                        int underlap = upper_bound - leftRange.first.second;
                        int offset1 = underlap / 2;
                        int offset2 = underlap - offset1;
                        leftRange.first.second += offset1;
                        rightRange.first.first -= offset2;
                    }
                }
            }
        }
    }
    return ranges;
}