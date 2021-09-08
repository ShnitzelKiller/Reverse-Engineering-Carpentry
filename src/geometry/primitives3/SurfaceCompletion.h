//
// Created by James Noeckel on 10/21/20.
//

#pragma once
#include "geometry/primitives3/BoundedPlane.h"
#include <Eigen/Dense>

struct NeighborEdge {
    float cost;

    /** indices of primitives which intersect with the same orientation as this edge */
    std::vector<int> intersectingPrimitives;

    unsigned char type;

    /** t in [0, 1] for where intersection with each primitive occurs */
    //std::vector<double> intersectionParameters;
};

class SurfaceCompletion {
public:
    SurfaceCompletion(const Eigen::Ref<const Eigen::Vector3d> &minPt, const Eigen::Ref<const Eigen::Vector3d> &maxPt, double grid_spacing, int max_resolution=std::numeric_limits<int>::max());
    void setPrimitives(std::vector<BoundedPlane> planes);
    void constructProblem(double minThickness, double maxThickness);

    /** set inside/outside segentation manually (1=inside) */
    bool setSegmentation(const std::vector<bool> &segmentation);
    const std::vector<bool> &getSegmentation() const;

//    void addInsideConstraint(int id);
//    void addOutsideConstraint(int id);
    void addInsideConstraint(int i, int j, int k);
    void addOutsideConstraint(int i, int j, int k);

    //debug
    const std::vector<int> &insideConstraints() const;
    const std::vector<int> &outsideConstraints() const;

    /** compute the cost of the current segmentation */
    float getCurrentCost() const;
    Eigen::Vector3d minPt() const;
    double spacing() const;
    float maxflow();
    void markViolatingEdges();
    /** get the signed distance after a segmentation is computed via maxflow() */
    std::vector<float> distfun();
    Eigen::Array3i resolution() const;
    Eigen::Vector3d getPosition(int gridIndex) const;
    int getGridIndex(int i, int j, int k) const;
private:
    /** grid index -> neighbor index 0-25 -> (target grid index, edge properties) */
    std::vector<std::vector<std::pair<int, NeighborEdge>>> neighborProperties_;
    /** grid index -> inside? */
    std::vector<bool> segmentation_;
    /** strictly positive distance function */
    std::vector<float> distfun_;
    std::vector<int> C_in_;
    std::vector<int> C_out_;
    std::vector<BoundedPlane> planes_;
    Eigen::Array3i res_;
    Eigen::Vector3d minPt_;
    double spacing_;
    size_t totalEdges_;
    std::vector<float> constWeights_;
    std::vector<float> edgeLengthUpperBounds_;
    void removeConflictingConstraints();
};

