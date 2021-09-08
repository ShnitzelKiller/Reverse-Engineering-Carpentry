//
// Created by James Noeckel on 10/21/20.
//

#include "SurfaceCompletion.h"
#include <cassert>
#include "maxflow.h"
#include <memory>
#include <iostream>
#include "utils/sorted_data_structures.hpp"
#include <set>
#define CONSTRAINT_ENERGY 1e16f
#define MIN_DIST 1e-13f

using namespace Eigen;

SurfaceCompletion::SurfaceCompletion(const Ref<const Vector3d> &minPt, const Ref<const Vector3d> &maxPt, double grid_spacing, int max_resolution)
    : spacing_(grid_spacing), minPt_(minPt)
{
    Array3d dims = maxPt - minPt;
    if (grid_spacing <= 0) {
        grid_spacing = dims.maxCoeff() / max_resolution;
    }
    res_ = (dims / grid_spacing).ceil().cast<int>() + 1;
    int ind;
    int biggest_res = res_.maxCoeff(&ind);
    if (biggest_res > max_resolution) {
        grid_spacing = dims[ind] / max_resolution;
        res_ = (dims / grid_spacing).ceil().cast<int>() + 1;
    }

    {
        // delta^3 * dPhi / (pi * |e_k|)
        double baseWeight = spacing_ * spacing_ * (4.0 / 26);
        constWeights_.resize(3);
        for (int i=0; i<3; ++i) {
            constWeights_[i] = static_cast<float>(baseWeight / std::sqrt(i+1));
        }
    }
    edgeLengthUpperBounds_.resize(3);
    for (int i=0; i<3; ++i) {
        edgeLengthUpperBounds_[i] = spacing_ * std::sqrt(i + 1) * 1.01;
    }
}

void SurfaceCompletion::setPrimitives(std::vector<BoundedPlane> planes) {
    planes_ = std::move(planes);
}

Eigen::Array3i SurfaceCompletion::resolution() const {
    return res_;
}

//void SurfaceCompletion::addInsideConstraint(int id) {
//    C_in_.push_back(id);
//}
//
//void SurfaceCompletion::addOutsideConstraint(int id) {
//    C_out_.push_back(id);
//}

void SurfaceCompletion::addInsideConstraint(int i, int j, int k) {
    C_in_.push_back(getGridIndex(i, j, k));
}

void SurfaceCompletion::addOutsideConstraint(int i, int j, int k) {
    C_out_.push_back(getGridIndex(i, j, k));
}

const std::vector<int> &SurfaceCompletion::insideConstraints() const {
    return C_in_;
}

const std::vector<int> &SurfaceCompletion::outsideConstraints() const {
    return C_out_;
}


void SurfaceCompletion::constructProblem(double minThickness, double maxThickness) {
    totalEdges_ = 0;
    neighborProperties_.clear();
    C_in_.clear();
    C_out_.clear();
    int N = res_.prod();
    neighborProperties_.resize(N);
    for (auto &neighbor : neighborProperties_) {
        neighbor.reserve(26);
    }
    distfun_.resize(N);
    int gridIndex = 0;
    std::vector<double> allSignedDists(planes_.size());
    for (int i=0; i<res_.x(); ++i) {
        for (int j=0; j<res_.y(); ++j) {
            for (int k=0; k<res_.z(); ++k, ++gridIndex) {
                if (i == 0 || i == res_.x()-1 || j==0 || j==res_.y()-1 || k==0 || k==res_.z()-1) {
                    C_out_.push_back(gridIndex);
                }
                Vector3d pos = minPt_ + spacing_ * Vector3d(i, j, k);
                double minDist = std::numeric_limits<double>::max();
                double largestNegativeDist = std::numeric_limits<double>::lowest();
                for (int p=0; p<planes_.size(); ++p) {
                    const auto &plane = planes_[p];
                    double signedDist = plane.normalDistance(pos.transpose())(0);
                    minDist = std::min(minDist, std::abs(signedDist));
                    allSignedDists[p] = signedDist;
                    if (signedDist < 0) {
                        largestNegativeDist = std::max(largestNegativeDist, signedDist);

                    }
                }
                distfun_[gridIndex] = minDist;
                //visit 26 neighbors
                for (int di=-1; di<=1; ++di) {
                    for (int dj=-1; dj<=1; ++dj) {
                        for (int dk=-1; dk<=1; ++dk) {
                            if (di == 0 && dj == 0 && dk == 0) continue;
                            int pi = i+di;
                            int pj = j+dj;
                            int pk = k+dk;
                            if (pi < 0 || pj < 0 || pk < 0 ||
                                pi >= res_.x() || pj >= res_.y() || pk >= res_.z()) {
                                continue;
                            }
                            //Vector3f pos2 = minPt_ + spacing_ * Vector3f(pi, pj, pk);
                            NeighborEdge neighbor;
                            int neighborType = std::abs(di) + std::abs(dj) + std::abs(dk) - 1;
                            assert(neighborType >= 0 && neighborType < 3);
                            neighbor.type = neighborType;
                            int neighborIndex = getGridIndex(pi, pj, pk);
                            Vector3d dir = Vector3d(di, dj, dk) * spacing_;
                            double maxScore = 0;
                            for (size_t p=0; p<planes_.size(); ++p) {
                                if (-allSignedDists[p] > minThickness && dir.dot(planes_[p].normal()) < 0) {
                                    maxScore = std::max(maxScore, 1.0 - (-allSignedDists[p] - minThickness) / (maxThickness - minThickness));
                                }
                            }

                            bool isOnPrimitive = false;
                            bool isInsideSupport = false;
//                            bool isOutside = false;
                            if (largestNegativeDist > -edgeLengthUpperBounds_[2]) {
                                Vector3d pos2 = minPt_ + spacing_ * Vector3d(pi, pj, pk);
                                for (size_t p = 0; p < planes_.size(); ++p) {
                                    const auto &plane = planes_[p];
                                    double dist1 = allSignedDists[p];
                                    if (dist1 < 0 && dist1 > -edgeLengthUpperBounds_[neighborType]) {
                                        double dist2 = plane.normalDistance(pos2.transpose())(0);
                                        if (dist2 >= 0) {
                                            bool oriented = dir.dot(plane.normal()) > 0;
                                            if (oriented) {
                                                isOnPrimitive = true;
                                                neighbor.intersectingPrimitives.push_back(p);
                                                if (!isInsideSupport) {
                                                    double t;
                                                    plane.intersectRay(pos, dir, t, true);
                                                    Vector3d projected_pt = pos + t * dir;
                                                    if (plane.hasShape() && plane.contains(
                                                            plane.project(projected_pt.transpose()).transpose())) {
                                                        isInsideSupport = true;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            if (isOnPrimitive) {
                                neighbor.cost = 0;
                            } else {
                                neighbor.cost = constWeights_[neighborType] * static_cast<float>(1-maxScore);
                            }
                            if (isInsideSupport) {
                                C_in_.push_back(gridIndex);
                                C_out_.push_back(neighborIndex);
                            }
                            neighborProperties_[gridIndex].emplace_back(neighborIndex, std::move(neighbor));
                            ++totalEdges_;
                        }
                    }
                }
            }
        }
    }
}

void SurfaceCompletion::removeConflictingConstraints() {
    std::sort(C_in_.begin(), C_in_.end());
    auto itEnd = std::unique(C_in_.begin(), C_in_.end());
    C_in_.erase(itEnd, C_in_.end());
    std::sort(C_out_.begin(), C_out_.end());
    itEnd = std::unique(C_out_.begin(), C_out_.end());
    C_out_.erase(itEnd, C_out_.end());
    {
        int i_in=0, i_out=0;
        std::vector<int> intersection;
        while (i_in < C_in_.size() && i_out < C_out_.size()) {
            if (C_in_[i_in] < C_out_[i_out]) {
                ++i_in;
            } else if (C_in_[i_in] > C_out_[i_out]) {
                ++i_out;
            } else {
                intersection.push_back(C_in_[i_in]);
                ++i_in;
                ++i_out;
            }
        }
        std::cout << "removing " << intersection.size() << " conflicting constraints" << std::endl;
        std::sort(intersection.begin(), intersection.end());
        C_in_.erase(std::remove_if(C_in_.begin(), C_in_.end(), [&](int elem) {return sorted_contains(intersection, elem);}), C_in_.end());
        C_out_.erase(std::remove_if(C_out_.begin(), C_out_.end(), [&](int elem) {return sorted_contains(intersection, elem);}), C_out_.end());
    }
}

bool SurfaceCompletion::setSegmentation(const std::vector<bool> &segmentation) {
    if (segmentation.size() != res_.prod()) return false;
    segmentation_ = segmentation;
    return true;
}

const std::vector<bool> &SurfaceCompletion::getSegmentation() const {
    return segmentation_;
}

Vector3d SurfaceCompletion::minPt() const {
    return minPt_;
}

double SurfaceCompletion::spacing() const {
    return spacing_;
}


float SurfaceCompletion::getCurrentCost() const {
    int N = neighborProperties_.size();
    if (N == 0 || N != segmentation_.size()) return std::numeric_limits<float>::max();
    float totalCost = 0;
    for (size_t i=0; i<N; ++i) {
        for (const auto &neighbor : neighborProperties_[i]) {
            if (segmentation_[i]  && !segmentation_[neighbor.first]) {
                totalCost += neighbor.second.cost;
            }
        }
    }
    return totalCost;
}

void error_fn(const char *msg) {
    std::cerr << msg << std::endl;
}

float SurfaceCompletion::maxflow() {
    removeConflictingConstraints();
    std::cout << "building problem" << std::endl;
    int N_grid = res_.prod();
    auto *graph = new maxflow::Graph_FFF(N_grid, totalEdges_, &error_fn);
    graph->add_node(N_grid);
    for (int i : C_in_) {
        graph->add_tweights(i, CONSTRAINT_ENERGY, 0);
    }
    for (int i : C_out_) {
        graph->add_tweights(i, 0, CONSTRAINT_ENERGY);
    }
    for (int gridIndex=0; gridIndex<N_grid; ++gridIndex) {
        for (const auto &neighbor : neighborProperties_[gridIndex]) {
            //only add in increasing order of indices to prevent duplicates (since we add cost and reverse cost)
            if (gridIndex < neighbor.first) {
                float cost = neighbor.second.cost;
                //find reverse neighbor edge
                auto it = sorted_find(neighborProperties_[neighbor.first], gridIndex);
                assert(it != neighborProperties_[neighbor.first].end());
                float reverseCost = it->second.cost;
                graph->add_edge(gridIndex, neighbor.first, cost, reverseCost);
            }
        }
    }
    std::cout << "running flow" << std::endl;
    float flow = graph->maxflow();
    std::cout << "interpreting solution" << std::endl;
    segmentation_.resize(N_grid);
    for (int i=0; i<N_grid; ++i) {
        segmentation_[i] = graph->what_segment(i);
    }
    delete graph;
    return flow;
}

template <typename T>
std::pair<T, T> sorted_pair(T a, T b) {
    std::pair<T, T> orig(a, b);
    if (orig.first > orig.second) std::swap(orig.first, orig.second);
    return orig;
}

void SurfaceCompletion::markViolatingEdges() {
    std::vector<std::pair<int, int>> seeds(planes_.size(), {-1, -1});
    int N_grid = res_.prod();
    for (int i=0; i<N_grid; ++i) {
        if (segmentation_[i]) {
            for (const auto &neighbor : neighborProperties_[i]) {
                if (!segmentation_[neighbor.first]) {
                    if (sorted_contains(C_in_, i) && neighbor.second.intersectingPrimitives.size() == 1) { //TODO: keep track of which primitive's constraint this was instead of requiring there to be just one
                        seeds[neighbor.second.intersectingPrimitives[0]] = sorted_pair(i, neighbor.first);
                    }
                }
            }
        }
    }
    /** planeID -> edge set */
    std::vector<std::set<std::pair<int, int>>> edgeSets(planes_.size());
    for (int p=0; p<planes_.size(); ++p) {
        if (seeds[p].first >= 0) {
            std::set<std::pair<int, int>> visitedEdgeSet;
            std::vector<std::pair<int, int>> frontier;
            frontier.push_back(seeds[p]);
            while (!frontier.empty()) {
                auto edge = frontier.back();
                frontier.pop_back();
                visitedEdgeSet.insert(edge);
                //source's other neighbors
                for (const auto &neighbor : neighborProperties_[edge.first]) {
                    if (neighbor.first != edge.second && segmentation_[neighbor.first] != segmentation_[edge.first]) {
                        auto newEdge = sorted_pair(edge.first, neighbor.first);
                        if (visitedEdgeSet.find(newEdge) == visitedEdgeSet.end()) {
                            frontier.push_back(newEdge);
                        }
                    }
                }
                //target's other neighbors
                for (const auto &neighbor : neighborProperties_[edge.second]) {
                    if (neighbor.first != edge.first && segmentation_[neighbor.first] != segmentation_[edge.second]) {
                        auto newEdge = sorted_pair(edge.second, neighbor.first);
                        if (visitedEdgeSet.find(newEdge) == visitedEdgeSet.end()) {
                            frontier.push_back(newEdge);
                        }
                    }
                }
            }
            edgeSets[p] = visitedEdgeSet;
        }
    }
    // for each cut edge, remove intersections with planes that are not connected to the original support
    int numMarked = 0;
    for (int i=0; i<N_grid; ++i) {
        if (segmentation_[i]) {
            for (auto &neighbor : neighborProperties_[i]) {
                if (!segmentation_[neighbor.first]) {
                    auto &markedPrimitives = neighbor.second.intersectingPrimitives;
                    auto endIter = markedPrimitives.end();
                    for (auto p : markedPrimitives) {
                        auto sortedEdge = sorted_pair(i, neighbor.first);
                        if (edgeSets[p].find(sortedEdge) == edgeSets[p].end()) {
                            endIter = std::remove(markedPrimitives.begin(), endIter, p);
                        }
                    }
                    markedPrimitives.erase(endIter, markedPrimitives.end());
                    if (markedPrimitives.empty()) {
                        neighbor.second.cost = constWeights_[neighbor.second.type];
                        ++numMarked;
                    }
                }
            }
        }
    }
    std::cout << "marked " << numMarked << " invalid edges" << std::endl;
}

Vector3d SurfaceCompletion::getPosition(int i) const {
    return minPt_ + spacing_ * Vector3d(i / (res_.z() * res_.y()), (i / res_.z()) % res_.y(), i % res_.z());
}

int SurfaceCompletion::getGridIndex(int i, int j, int k) const {
    return i * res_.y() * res_.z() + j * res_.z() + k;
}

std::vector<float> SurfaceCompletion::distfun() {
    int N_grid = res_.prod();
    std::vector<float> distfun(N_grid);
    for (int i=0; i<N_grid; ++i) {
        float unsignedDist = std::max(distfun_[i], MIN_DIST);
        distfun[i] = segmentation_[i] ? -unsignedDist : unsignedDist;
    }
    return distfun;
}


