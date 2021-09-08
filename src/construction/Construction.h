//
// Created by James Noeckel on 7/1/20.
//

#pragma once

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <vector>
#include <Eigen/Dense>
#include "geometry/shapes2/Primitive.h"
#include "utils/typedefs.hpp"
#include <igl/AABB.h>
#include "geometry/primitives3/Ray3d.h"
#include "geometry/primitives3/MultiRay3d.h"
#include "Constraints.h"

/**
 * Part graph node
 */
 struct PartNode {
     size_t partIdx;
 };

/**
 * Information pertaining to the part's role in the construction
 */
class PartData {
public:
    /** ID of part data used by this part (may be shared between instances) */
    size_t shapeIdx;

    std::vector<size_t> opposingPartIds;
    bool bothSidesObserved;
    std::vector<int> pointIndices;
    bool groundPlane = false;

    /** pose: world position is rot * p + pos. Local z coordinate is distance along plane normal*/
    Eigen::Quaterniond rot;
    Eigen::Vector3d pos;
    Eigen::Vector3d normal() const;
    double offset() const;
    Eigen::MatrixX3d unproject(const Eigen::Ref<const Eigen::MatrixX2d> &points) const;
    Eigen::MatrixX2d project(const Eigen::Ref<const Eigen::MatrixX3d> &points) const;
    Eigen::Vector2d projectDir(const Eigen::Ref<const Eigen::Vector3d> &vec) const;
};

/**
 * Geometric data defining a part, before 3D pose is applied
 */
class ShapeData {
public:
    /** ID of the stock used to cut this part */
    size_t stockIdx;

    /** 2D shape */
    std::shared_ptr<Primitive> cutPath;
//    std::shared_ptr<Primitive> bbox;
//    std::shared_ptr<Primitive> pointDensityContour;
    double gridSpacing;

    /** 2D constraint lines assuming counter-cockwise boundary (right-facing normal is outside, left is inside) */
    std::unordered_map<int, std::vector<ShapeConstraint>> shapeConstraints;
    std::unordered_map<int, std::vector<ConnectionConstraint>> connectionConstraints;
    std::unordered_map<int, std::vector<Guide>> guides;
    /** true if the shape has been optimized with respect to an outdated set of constraints */
    bool dirty = true;
};


class StockData {
public:
    double thickness;
};

enum LoadMode {
    SELECTION,
    ASSEMBLY,
    SEGMENTATION,
    BASIC,
    BASICPLUS
};

/**
 * Edge metadata, specifying the connection type and constituent parts
 */
class ConnectionEdge {
public:
    enum ConnectionType {
        CET_CUT_TO_FACE, //part 1 cut is in contact with face of part 2
        CET_CORNER, //same, but part 2 cut is coplanar with main plane of part 1
        CET_CUT_TO_CUT,
        CET_FACE_TO_FACE,
        CET_UNKNOWN
    };
    ConnectionType type = CET_UNKNOWN;
    /** backface2: whether part 2's back face makes contact in the connection
     * backface1: whether part 1's back face makes contact in connection (if in face-face connection) or if it is aligned with the outer cut of part 2 (if in corner connection)
     * */
    bool backface1, backface2;
    /** line of contact of innermost planes (with respect to connection) */
    MultiRay3d innerEdge;

    /** line of contact, but ignoring "incident" part shape (length of entire "cutting" part) */
//    MultiRay3d bigEdge;

    /** node IDs within the graph */
    size_t part1;
    size_t part2;

    /** ray representing the interface, with a point on the interface and a normal vector
     * Normal points outwards from part 1 into part 2*/
    std::vector<Ray3d> interfaces;

    /** 0 = none, 1 = connection optimized, 2 = contact optimized */
    int optimizeStage = 0;
};

class Construction {
public:
    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, PartNode, ConnectionEdge> Graph;
    typedef boost::graph_traits<Graph>::vertex_iterator vertex_iter;
    typedef boost::graph_traits<Graph>::edge_iterator edge_iter;
    typedef boost::graph_traits<Graph>::out_edge_iterator out_edge_iter;
    typedef boost::graph_traits<Graph>::adjacency_iterator adjacency_iter;
    typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
    typedef boost::graph_traits<Graph>::edge_descriptor Edge;
    typedef boost::property_map<Graph, boost::vertex_index_t>::type IndexMap;
    Construction() = default;

    /** set the part selection vector */
    void setW(const std::vector<bool> &w);

    /** visualize the construction in igl */
    void visualize(double displacement, const std::string& connectorsMesh="", double spacing=1, double scale=1) const;

    void saveShapes() const;

    bool exportMesh(const std::string &filename, const std::string &connectorFilename="", double connectorSpacing=1, double scale=1) const;

    /** export in part-based XML format */
    bool exportModel(const std::string &filename) const;

    bool exportPart(size_t partIdx, std::ofstream &of, LoadMode mode) const;

        /** export in easily parseable plaintext format */
    bool exportPlaintext(const std::string &filename, LoadMode mode) const;

    /** load the format saved by exportPlaintext() */
    bool loadPlaintext(const std::string &filename, LoadMode mode);

    /** merge thicknesses based on a threshold */
    int regularizeDepths(double threshold);

    /** populate the graph with vertices and connections according to the current part data and w vector */
    int buildGraph(double distanceThreshold);

    int findNewConnections(double distanceThreshold);

    MultiRay3d connectionContact(const ConnectionEdge &connection, double margin);

    void recomputeConnectionContacts(double margin, bool useCurves);

//    void realignConnectionEdges();

    void recomputeConnectionConstraints(double margin, double max_abs_dot_product);

    void pruneShapeConstraints();

    //void enforceConnections();
    /**
     * Compute connectivity metric (requires W to be set)
     * @param a constant for probability distribution
     * @return
     */
    double connectivityEnergy(double a);
    /**
     * Compute total squared distance of points (requires W to be set)
     * @param points
     * @return
     */
    double pointDistance(const PointCloud3::Handle &points);
    /**
     * Compute squared distance of every point to every candidate part
     * @param points
     * @return N x K matrix, where N is the size of the point cloud and K is the number of candidate parts
     */
    Eigen::MatrixXd allDistances(const PointCloud3::Handle &points, int stride=1);

    /**
     * Minimum symmetric distance between 2 parts
     * @param idx1 id of part 1
     * @param idx2 id of part 2
     * @param point1 point on part 1
     * @param point2 point on part 2
     * @return
     */
    double distanceBetweenParts(size_t idx1, size_t idx2, Eigen::Ref<Eigen::RowVector3d> point1, Eigen::Ref<Eigen::RowVector3d> point2) const;

    /** Check if an edge to face connection with the specified parts is allowed (ignore input connection type for this stage) */
    bool connectionValid(ConnectionEdge &connection, double margin);

    /**
     * Minimum symmetric distance between 2 parts
     * @param idx1 id of part 1
     * @param idx2 id of part 2
     * @return
     */
    double distanceBetweenParts(size_t idx1, size_t idx2) const;

    std::pair<Vertex, bool> partIdxToVertex(size_t partIdx) const;

    bool contains(size_t partIdx, const Eigen::Ref<const Eigen::RowVector3d> &pt);

    /** precompute individual part meshes */
    void computeMeshes(bool recomputeBVH=true);

    /** compile a single mesh out of current choice of parts
     *  tuple of V, F, I
     *  where I is a vector of part indices for each element in F */
    void computeTotalMesh(bool recomputeBVH=true);

    std::pair<Eigen::MatrixX3d, Eigen::MatrixX3i> connectorMesh(const std::string &connectorOBJ, double spacing, double scale) const;

    /** left-justify all 2D shapes (each part must have a unique shape ID for this to work) */
    void recenter();

    /** find a "nice" basis for all the parts based on their connections */
    void rotateParts();

    /** scale entire model */
    void scale(double scale);

    void rotate(const Eigen::Quaterniond &rot);

    StockData &getStock(size_t partIdx);
    const StockData &getStock(size_t partIdx) const;
    ShapeData &getShape(size_t partIdx);
    const ShapeData &getShape(size_t partIdx) const;

    Graph g;
    std::vector<PartData> partData;
    std::vector<ShapeData> shapeData;
    std::vector<StockData> stockData;
    std::vector<std::pair<Eigen::MatrixX3d, Eigen::MatrixX3i>> partMeshes;
    /** vertices, faces, face part labels */
    std::tuple<Eigen::MatrixX3d, Eigen::MatrixXi, Eigen::VectorXi> mesh;
    /** aabbs corresponding to partMeshes */
    std::vector<igl::AABB<Eigen::MatrixX3d, 3>> aabbs;
    /** aabb corresponding to mesh */
    igl::AABB<Eigen::MatrixX3d, 3> aabb;
    bool hasTotalMeshInfo = false;
    std::vector<bool> w_;

private:

    /**
     * Smallest distance on part 1 to any vertex of part 2
     * @param idx1 id of part 1
     * @param idx2 id of part 2
     * @param closestPoint closest point on part 1
     * @param index index of vertex in part 2 closest to 1
     * @return
     */
    double distanceBetweenPartsAsymm(size_t idx1, size_t idx2, Eigen::Ref<Eigen::RowVector3d> closestPoint, int &index) const;

    /**
     * Find the 3D contact edge between the specified parts
     * @param partIdx1
     * @param partIdx2
     * @param backface1 whether part1's back face meets the inner corner
     * @param backface2 whether part2's back face meets the inner corner
     */
    MultiRay3d contactEdge(size_t partIdx1, size_t partIdx2, double offset1, double offset2, double margin, bool usePart1Shape=true, bool usePart2Shape=true) const;

    /**
     * Precompute various vertex representations of the part given its curves
     * @param c
     * @param curves2D
     * @param curves2DBack
     * @param curves3D
     * @param curves3DBack
     * @param holes
     */
    void samplePartCurves(size_t c, Eigen::MatrixX2d &curves2D, Eigen::MatrixX2d &curves2DBack, Eigen::MatrixX3d &curves3D, Eigen::MatrixX3d &curves3DBack, std::vector<std::pair<Eigen::MatrixX2d, Eigen::MatrixX3d>> &holes) const;

    CombinedCurve getBackCurve(size_t partIdx) const;
};