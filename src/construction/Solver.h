//
// Created by James Noeckel on 7/8/20.
//

#pragma once
#include "Construction.h"
#include <random>
#include <utility>
#include "utils/settings.h"
#include "utils/typedefs.hpp"
#include "reconstruction/ReconstructionData.h"

struct SegmentationResult {
    std::vector<bool> grid;
    int width, height;
    Eigen::Vector2d minPt;
    double spacing;
};

class Solver {
public:
    explicit Solver(Settings settings, std::mt19937 &random) : settings_(std::move(settings)), random_(random) {}
    explicit Solver(Settings settings, Construction c, std::mt19937 &random) : settings_(std::move(settings)), construction(std::move(c)), random_(random) {}
    void setDataPoints(PointCloud3::Handle cloud);
    void computeBounds();
    void setReconstructionData(ReconstructionData::Handle rec);

    //BEFORE GRAPH REPRESENTATION

    /** Find candidate part planes and generate all possible candidate parts */
    void initialize();

    /** find views that can see a surface point with the given normal */
    std::vector<std::pair<int32_t, double>> getVisibility(const Eigen::Vector3d &origin, const Eigen::Vector3d &normal);

    /** populate visibility map data structures */
    void computeVisibility();

    /** choose between pairs of parts known to conflict as opposite faces of the same part */
    void pruneOpposing();

    /** select the initial set of part planes to use
     * algorithm: 0=simulated annealing, 1=greedy area, 2=greedy point distance error */
    bool optimizeW(int algorithm=0);

    /** Find the initial set of connections (@return number of connections)*/
    int buildGraph();

    // AGNOSTIC TO GRAPH REPRESENTATION

    /** reduce the number of distinct thicknesses in the shape data */
    int regularizeDepths();

    /** precompute meshes and AABBs for distance computations */
    void recomputeMeshes(bool recomputeBVH=true);


    //REQUIRES GRAPH REPRESENTATION

    /** find new connections */
    int findNewConnections();

    void initializeConnectionTypes();

    void refineConnectionContacts(bool useCurves=false);

    /** Optimize connection types based on image and geometric evidence */
    void optimizeConnections(bool useImages=true);

    /** re-optimize part orientations w.r.t orthogonal connection constraints */
    void realign();

    void recomputeConnectionConstraints();

    /** compute dense contours from joint segmentation of image views, return the number of new parts resulting from splitting */
    int shapeFromImages(bool useConstraints=true);

    /** optimize part shape curves with constraints based on part connections */
    void optimizeShapes(bool useConstraints=true);

    /** globally align vertically (to ensure uprightability) */
    void globalAlignShapes();

    /** remove co-planar knots and fix tangents around low-angle knots */
    void regularizeKnots();

    int removeDisconnectedParts();

    void visualize() const;
    /*bool exportMesh(const std::string &filename) const;
    bool exportModel(const std::string &filename) const;*/
    bool hasNormals() const;
    Construction construction;

    const Construction &getConstruction() const;
    Construction &getConstruction();

    /** map from partIdx -> list of (view, normal dot prod)
     * if idx > numParts, use the backface of part idx - numParts
     * in order of increasing view angle with normal */
    std::vector<std::vector<std::pair<int32_t, double>>> cluster_visibility;
//    std::vector<std::vector<int32_t>> pruned_cluster_visibility;
//    std::unordered_map<int32_t, std::vector<int>> visible_clusters;
private:

    /** Check if this corner connection cuts off part of shape 1
     * return the amount of penetration regardless of whether a shape piece was cut off*/
    double cornerConnectionPenetration(ConnectionEdge &connection, bool &cutoff);

    /** Check if a connection is better served by the cut surfaces if originally a corner connection, otherwise find contacts of an existing cut-cut connection */
    bool sideConnectionValid(ConnectionEdge &connection);

    /**
     * Score the connection of part 1 to part 2 of the specified type
     * @return score, bigger is better
     */
    double connectionScore(const ConnectionEdge &connection);

    /** get list of edge intensities in the image sliding along the specified direction. Return false if no views found */
    bool edgeValues(size_t part1, size_t part2, const Edge3d &edge, const Eigen::Vector3d &faceNormal, const Eigen::Vector3d &connectionNormal, double &outMaxColorDifference, double &outMaxLumDerivative, double &outWeight, int id=0);

    ReconstructionData::Handle reconstruction_;
    double diameter_;
public:
    PointCloud3::Handle cloud_;
    double getDiameter() const;

private:
    std::mt19937 &random_;
    const Settings settings_;
    double floorHeight;

    /** map of segmentation results for each part ID */
//    std::unordered_map<size_t, SegmentationResult> segmentationResults;

    /** solver state */
    bool hasGraph = false;
};

