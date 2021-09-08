//
// Created by James Noeckel on 7/1/20.
//

#include "Construction.h"
#include <igl/triangle/triangulate.h>
#include "geometry/shapes2/Primitive.h"
//#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <boost/graph/one_bit_color_map.hpp>
#include <boost/graph/stoer_wagner_min_cut.hpp>
//#include "geometry/geom.h"
#include "geometry/primitives3/intersect_planes.h"
#include "geometry/primitives3/BoundedPlane.h"
#include "utils/mesh_valid.h"
#include "geometry/primitives3/Multimesh.h"


using namespace Eigen;
using namespace boost;

Vector3d PartData::normal() const {
    return rot * Vector3d(0, 0, 1);
}

double PartData::offset() const {
    return -normal().dot(pos);
}

MatrixX3d PartData::unproject(const Ref<const MatrixX2d> &points) const {
    MatrixX3d points3d(points.rows(), 3);
    for (size_t r=0; r<points.rows(); ++r) {
        points3d.row(r) = (rot * Vector3d(points(r, 0), points(r, 1), 0) + pos).transpose();
    }
    return points3d;
}

MatrixX2d PartData::project(const Ref<const MatrixX3d> &points) const {
    MatrixX2d points2d(points.rows(), 2);
    Quaterniond rotinv = rot.conjugate();
    for (size_t r=0; r<points.rows(); ++r) {
        points2d.row(r) = (rotinv * (points.row(r).transpose() - pos)).transpose().head(2);
    }
    return points2d;
}

Vector2d PartData::projectDir(const Ref<const Vector3d> &vec) const {
    Quaterniond rotinv = rot.conjugate();
    return (rotinv * vec).head(2);
}

void Construction::setW(const std::vector<bool> &w) {
    w_ = w;
//    computeTotalMesh();
}

double Construction::connectivityEnergy(double a) {
    std::vector<std::tuple<size_t, size_t, double>> weightedEdges;
    std::vector<size_t> indices;
    for (size_t i=0; i<partData.size(); ++i) {
        if (w_[i]) {
            indices.push_back(i);
        }
    }
    if (indices.size() < 2) {
        return 0;
    }
    std::vector<size_t> inv_indices(partData.size(), -1);
    for (size_t i=0; i<indices.size(); ++i) {
        inv_indices[indices[i]] = i;
    }
    //TODO: don't create edges below a threshold probability
    for (auto it=indices.begin(); it != indices.end(); ++it) {
        for (auto it2 = it+1; it2 != indices.end(); ++it2) {
            double distance = distanceBetweenParts(*it, *it2);
            /** probability of non-connection */
            double Qij = std::max(1.0 - exp(-a * distance), 1e-8);
            double weight = -log(Qij);
            weightedEdges.emplace_back(inv_indices[*it], inv_indices[*it2], weight);
        }
    }

    typedef boost::property<boost::edge_weight_t, double> EdgeWeightProperty;
    typedef boost::adjacency_list<boost::listS, boost::vecS,boost::undirectedS,boost::no_property,EdgeWeightProperty> DenseGraph;
    typedef graph_traits < DenseGraph >::edge_descriptor DEdge;
    typedef graph_traits < DenseGraph >::vertex_descriptor DVertex;
    property_map<DenseGraph, vertex_index_t>::type dIndex;
    DenseGraph dg(indices.size());
    property_map<DenseGraph, edge_weight_t>::type weightmap = get(edge_weight, dg);
    for (const auto &edge : weightedEdges) {
        DEdge e;
        bool inserted;
        boost::tie(e, inserted) = add_edge(std::get<0>(edge), std::get<1>(edge), dg);
        weightmap[e] = std::get<2>(edge);
        //std::cout << "edge " << std::get<0>(edge) << "-" << std::get<1>(edge) << ": " << weightmap[e] << std::endl;
    }
    BOOST_AUTO(parities, boost::make_one_bit_color_map(num_vertices(dg), get(boost::vertex_index, dg)));
    double w = boost::stoer_wagner_min_cut(dg, get(boost::edge_weight, dg), boost::parity_map(parities));
    /** max probability of disjoint subsets of parts being disconnected */
    double prodQij = exp(-w);
    return -log(1.0 - prodQij);
}

double Construction::distanceBetweenPartsAsymm(size_t idx1, size_t idx2, Ref<RowVector3d> closestPoint, int &index) const {
    VectorXd sqrD;
    VectorXi II;
    MatrixXd CC;
    const auto &mesh1 = partMeshes[idx1];
    const auto &mesh2 = partMeshes[idx2];
    aabbs[idx1].squared_distance(mesh1.first, mesh1.second, mesh2.first, sqrD, II, CC);
    double d = sqrD.minCoeff(&index);
    closestPoint = CC.row(index);
    return std::sqrt(d);
}

double Construction::distanceBetweenParts(size_t idx1, size_t idx2, Ref<RowVector3d> point1, Ref<RowVector3d> point2) const {
    RowVector3d part1point, part2point;
    int part1index, part2index;
    double minDist1 = distanceBetweenPartsAsymm(idx1, idx2, part1point, part2index);
    double minDist2 = distanceBetweenPartsAsymm(idx2, idx1, part2point, part1index);
    if (minDist1 < minDist2) {
        point1 = part1point;
        point2 = partMeshes[idx2].first.row(part2index);
        return minDist1;
    } else {
        point1 = partMeshes[idx1].first.row(part1index);
        point2 = part2point;
        return minDist2;
    }
}

double Construction::distanceBetweenParts(size_t idx1, size_t idx2) const {
    RowVector3d point1;
    RowVector3d point2;
    return distanceBetweenParts(idx1, idx2, point1, point2);
}

bool Construction::contains(size_t partIdx, const Ref<const RowVector3d> &pt) {
    const auto &pd = partData[partIdx];
    Vector3d localPt = pd.rot.conjugate() * (pt.transpose() - pd.pos);
    if (localPt.z() > 0 || localPt.z() < -getStock(partIdx).thickness) {
        return false;
    }
    return getShape(partIdx).cutPath->contains(localPt.head(2));
}

StockData &Construction::getStock(size_t partIdx) {
    return stockData[shapeData[partData[partIdx].shapeIdx].stockIdx];
}

ShapeData &Construction::getShape(size_t partIdx) {
    return shapeData[partData[partIdx].shapeIdx];
}

const StockData &Construction::getStock(size_t partIdx) const {
    return stockData[shapeData[partData[partIdx].shapeIdx].stockIdx];
}

const ShapeData &Construction::getShape(size_t partIdx) const {
    return shapeData[partData[partIdx].shapeIdx];
}


int Construction::regularizeDepths(double threshold) {
    //TODO: threshold-based merging

    std::vector<double> thicknesses;
    std::vector<std::vector<size_t>> indices;
    for (size_t c=0; c<partData.size(); ++c) {
        if (w_[c]) {
            double thickness = getStock(c).thickness;
            bool found = false;
            for (size_t j=0; j<thicknesses.size(); ++j) {
                double t = thicknesses[j];
                if (std::abs(t - thickness) < threshold) {
                    found = true;
                    indices[j].push_back(c);
                    break;
                }
            }
            if (!found) {
                thicknesses.push_back(thickness);
                indices.push_back(std::vector<size_t>(1, c));
            }
        }
    }
    size_t newStartIndex = stockData.size();
    for (auto thickness : thicknesses) {
        StockData sd;
        sd.thickness = thickness;
        stockData.push_back(sd);
    }

    for (size_t tInd=0; tInd<thicknesses.size(); ++tInd) {
        for (auto c : indices[tInd]) {
//            double thicknessChange = getStock(c).thickness - thicknesses[tInd];
//            Vector3d n = partData[c].normal();
//            partData[c].pos += n * (thicknessChange / 2);
            getShape(c).stockIdx = tInd + newStartIndex;
        }
    }
    return thicknesses.size();
}

MatrixXd Construction::allDistances(const PointCloud3::Handle &points, int stride) {
    size_t nPts = points->P.rows()/stride;
    MatrixX3d pointsReduced;
    if (stride > 1) {
        pointsReduced = MatrixX3d(nPts, 3);
        for (size_t i = 0; i < pointsReduced.rows(); ++i) {
            pointsReduced.row(i) = points->P.row(i * stride);
        }
    }
    MatrixXd allDistances(nPts, partData.size());
    size_t N = partData.size();
#pragma omp parallel for default(none) shared(N, allDistances, aabbs, points, stride, pointsReduced)
    for (size_t k=0; k<N; ++k) {
        const auto &mesh = partMeshes[k];
        VectorXd sqrD;
        VectorXi II;
        MatrixXd CC;
        if (stride > 1) {
            aabbs[k].squared_distance(mesh.first, mesh.second, pointsReduced, sqrD, II, CC);
        } else {
            aabbs[k].squared_distance(mesh.first, mesh.second, points->P, sqrD, II, CC);
        }
        allDistances.col(k) = sqrD;
    }
    return allDistances;
}


double Construction::pointDistance(const PointCloud3::Handle &points) {
    size_t nV = 0;
    if (std::accumulate(w_.begin(), w_.end(), 0) == 0) return std::numeric_limits<double>::max();
    /*{
        igl::opengl::glfw::Viewer viewer;
        viewer.data().set_mesh(std::get<0>(mesh), std::get<1>(mesh));
        viewer.data().set_colors(std::get<2>(mesh));
        viewer.launch();
    }*/
    VectorXd sqrD;
    VectorXi II;
    MatrixXd CC;
    aabb.squared_distance(std::get<0>(mesh), std::get<1>(mesh), points->P, sqrD, II, CC);
    //igl::point_mesh_squared_distance(points->P,V,I,sqrD,II,CC);
    return sqrD.sum();
}

void Construction::computeTotalMesh(bool recomputeBVH) {
    Multimesh multi;
    for (int i=0; i<partData.size(); ++i) {
        if (w_[i]) {
            multi.AddMesh(partMeshes[i]);
        }
    }
    auto totalMesh = multi.GetTotalMesh();

    VectorXi I(totalMesh.second.rows());

    size_t offsetF = 0;
    //size_t realIndex = 0;
    std::cout << "populating vertices, faces" << std::endl;
    for (size_t i=0; i<partMeshes.size(); ++i) {
        if (w_[i]) {
            const auto &Fi = partMeshes[i].second;
            //C.block(offsetF, 0, Fi.rows(), 3).rowwise() = colorAtIndex(i, partData.size());
            for (size_t j=0; j<Fi.rows(); ++j) {
                I(offsetF + j) = i;
            }
            offsetF += partMeshes[i].second.rows();
            //++realIndex;
        }
    }
    mesh = {totalMesh.first, totalMesh.second, I};
    int invalidID = mesh_valid(totalMesh.first, totalMesh.second);
    if (invalidID >= 0) {
        std::cout << "total mesh invalid at face " << invalidID << std::endl;
        hasTotalMeshInfo = false;
        return;
    }
    aabb = igl::AABB<Eigen::MatrixX3d, 3>();
    if (recomputeBVH) {
        std::cout << "recomputing total AABB" << std::endl;
        aabb.init(std::get<0>(mesh), std::get<1>(mesh));
    }
    hasTotalMeshInfo = true;
}

CombinedCurve Construction::getBackCurve(size_t partIdx) const {
    const auto &shape = getShape(partIdx);
    double thickness = getStock(partIdx).thickness;
    CombinedCurve backCurve = shape.cutPath->curves();
    size_t N = backCurve.size();
    for (const auto &pair : backCurve.constraints_) {
        if (pair.second.tiltAngCos != 0) {
            std::cout << "tilting " << partIdx << " curve " << pair.first << " by " << pair.second.tiltAngCos << std::endl;
            //offset in the back plane to account for tilt
            double offset = thickness * pair.second.tiltAngCos / std::sqrt(1 - pair.second.tiltAngCos * pair.second.tiltAngCos);
            Vector2d ab = (pair.second.edge.second - pair.second.edge.first).normalized();
            Vector2d outNormal(ab.y(), -ab.x());
            Vector2d displacement = outNormal * offset;
            backCurve.moveVertex(pair.first, displacement);
            size_t nextIndex = (pair.first + 1) % N;
            backCurve.moveVertex(nextIndex, displacement);
        }
    }

    return backCurve;
}

void Construction::samplePartCurves(size_t c, MatrixX2d &curves2D, MatrixX2d &curves2DBack, MatrixX3d &curves3D, MatrixX3d &curves3DBack, std::vector<std::pair<MatrixX2d, MatrixX3d>> &holes) const {
    const auto &shape = getShape(c);
    double thickness = getStock(c).thickness;
    curves2D = shape.cutPath->points();
    curves3D = MatrixX3d(curves2D.rows(), 3);
    for (int r=0; r<curves2D.rows(); ++r) {
        curves3D.row(r) = (partData[c].rot * Vector3d(curves2D(r, 0), curves2D(r, 1), 0) + partData[c].pos).transpose();
    }

    {
        //back curve
        CombinedCurve backCurve = getBackCurve(c);
        PolyCurveWithHoles newPrimitive(std::move(backCurve));
        curves2DBack = newPrimitive.points();
        curves3DBack = MatrixX3d(curves2D.rows(), 3);
        for (int r=0; r<curves2D.rows(); ++r) {
            curves3DBack.row(r) = (partData[c].rot * Vector3d(curves2DBack(r, 0), curves2DBack(r, 1), -thickness) + partData[c].pos).transpose();
        }
    }

    holes.reserve(shape.cutPath->children().size());
    const auto children = shape.cutPath->children();
    for (const auto & i : children) {
        MatrixX2d childCurve = i->points();
        MatrixX3d childCurve3d(childCurve.rows(), 3);
        for (int r=0; r<childCurve.rows(); ++r) {
            childCurve3d.row(r) = (partData[c].rot * Vector3d(childCurve(r, 0), childCurve(r, 1), 0) + partData[c].pos).transpose();
        }
        holes.emplace_back(std::move(childCurve), std::move(childCurve3d));
    }
}

void Construction::computeMeshes(bool recomputeBVH) {
    partMeshes.clear();
    partMeshes.reserve(partData.size());
    /** part ID, 2d curve, 3d curve, holes */
    std::vector<std::tuple<size_t, MatrixX2d, MatrixX3d, std::vector<std::pair<MatrixX2d, MatrixX3d>>>> sampledCurves_;

    for (size_t c=0; c<partData.size(); ++c) {
        MatrixX2d curves2D;
        MatrixX2d curves2DBack;
        MatrixX3d curves3D;
        MatrixX3d curves3DBack;
        std::vector<std::pair<MatrixX2d, MatrixX3d>> holes;
        samplePartCurves(c, curves2D, curves2DBack, curves3D, curves3DBack, holes);
        if (curves2D.rows() == 0) continue;
        if (!holes.empty()) {
            std::cout << "generating mesh part " << c << " with " << holes.size() << " holes" << std::endl;
        }
        size_t numHolePts = 0;
        for (const auto & hole : holes) {
            numHolePts += hole.first.rows();
        }
        size_t numHalfTotalPoints = curves2D.rows() + numHolePts;
        MatrixX3d V(numHalfTotalPoints * 2, 3);
        V.block(0, 0, curves2D.rows(), 3) = curves3D;
        V.block(numHalfTotalPoints, 0, curves2D.rows(), 3) = curves3DBack;
        double thickness = getStock(c).thickness;
        std::cout << "part " << c << " thickness: " << thickness << " (shapeID " << partData[c].shapeIdx << ", stockID" << getShape(c).stockIdx << std::endl;
        RowVector3d normalShift = ((partData[c].normal()) * thickness).transpose();
        size_t holeOffset = curves2D.rows();
        for (const auto & hole : holes) {
            V.block(holeOffset, 0, hole.first.rows(), 3) = hole.second;
            V.block(holeOffset + numHalfTotalPoints, 0, hole.first.rows(), 3) = hole.second.rowwise() - normalShift;
            holeOffset += hole.first.rows();
        }
        MatrixX3i I(numHalfTotalPoints * 2, 3);
        for (int r = 0; r < curves2D.rows(); ++r) {
            int i0 = r;
            int i1 = (r + 1) % curves2D.rows();
            int i2 = numHalfTotalPoints + r;
            int i3 = numHalfTotalPoints + (r + 1) % curves2D.rows();
            I.row(r) = RowVector3i(i0, i1, i3);
            I.row(numHalfTotalPoints + r) = RowVector3i(i2, i0, i3);
        }
        holeOffset = curves2D.rows();
        for (const auto & hole : holes) {
            for (int r=0; r<hole.first.rows(); ++r) {
                int i0 = r;
                int i1 = (r + 1) % hole.first.rows();
                int i2 = numHalfTotalPoints + r;
                int i3 = numHalfTotalPoints + (r + 1) % hole.first.rows();
                I.row(holeOffset + r) = RowVector3i(i0, i1, i3).array() + holeOffset;
                I.row(holeOffset + numHalfTotalPoints + r) = RowVector3i(i2, i0, i3).array() + holeOffset;
            }
            holeOffset += hole.first.rows();
        }
        //triangulation
        Eigen::MatrixXd boundary2D(numHalfTotalPoints, 2);
        MatrixXi I2d(numHalfTotalPoints, 2);
        boundary2D.block(0, 0, curves2D.rows(), 2) = curves2D;
        for (size_t j=0; j<curves2D.rows(); ++j) {
            I2d(j, 0) = j;
            I2d(j, 1) = (j + 1) % curves2D.rows();
        }
        holeOffset = curves2D.rows();
        //std::cout << "curves2D.rows()=" << curves2D.rows() << "; boundary2d.rows()=" << boundary2D.rows() << std::endl;
        for (const auto &hole : holes) {
            boundary2D.block(holeOffset, 0, hole.first.rows(), 2) = hole.first;
            for (size_t j=0; j<hole.first.rows(); ++j) {
                I2d.row(holeOffset + j) = RowVector2i(
                        holeOffset + j,
                        holeOffset + (j+1) % hole.first.rows()
                        );
            }
            holeOffset += hole.first.rows();
        }

//        std::cout << "num rows: " << curves2D.rows() << std::endl << std::endl;
//        std::cout << I2d << std::endl << std::endl;
//        std::cout << curves2D << std::endl;
        //TODO: use guaranteed interior point instead of bounding box center
        MatrixXd H(holes.size(), 2);
        for (size_t i=0; i<holes.size(); ++i) {
            std::cout << "triangulating hole " << i << " to find interior point" << std::endl;
            MatrixXd HH(0, 2);
            MatrixXd VHole;
            MatrixXi FHole;
            MatrixXi I2dHole(holes[i].first.rows(), 2);
            for (size_t j=0; j<I2dHole.rows(); ++j) {
                I2dHole(j, 0) = j;
                I2dHole(j, 1) = (j+1) % I2dHole.rows();
            }
            igl::triangle::triangulate(holes[i].first, I2dHole, HH, "Qp", VHole, FHole);
            H.row(i) = (VHole.row(FHole(0, 0)) + VHole.row(FHole(0, 1)) + VHole.row(FHole(0, 2)))/3.0;
        }
        std::cout << "triangulating full shape" << std::endl;
        MatrixXd V2;
        MatrixXi F2;
        igl::triangle::triangulate(boundary2D, I2d, H, "p", V2, F2);
        std::cout << "triangulation finished" << std::endl;
        size_t Ioffset = I.rows();
        MatrixX3i Inew(Ioffset + F2.rows() * 2, 3);
        Inew.block(0, 0, Ioffset, 3) = I;
        Inew.block(Ioffset, 0, F2.rows(), 3) = F2;
        //back faces
        MatrixX3i F3 = F2.array() + numHalfTotalPoints;
        Inew.block(Ioffset + F2.rows(), 0, F2.rows(), 1) = F3.col(0);
        Inew.block(Ioffset + F2.rows(), 1, F2.rows(), 1) = F3.col(2);
        Inew.block(Ioffset + F2.rows(), 2, F2.rows(), 1) = F3.col(1);
        partMeshes.emplace_back(std::move(V), std::move(Inew));
    }
    aabbs.clear();
    if (recomputeBVH) {
        std::cout << "recomputing aabbs" << std::endl;
        aabbs.resize(partMeshes.size());
        for (size_t i = 0; i < aabbs.size(); ++i) {
            int invalidID = mesh_valid(partMeshes[i].first, partMeshes[i].second);
            if (invalidID >= 0) {
                std::cout << "Invalid mesh " << i << " at face " << invalidID << "!" << std::endl;
            }
            aabbs[i].init(partMeshes[i].first, partMeshes[i].second);
        }
    }
    std::cout << "recomputing total mesh" << std::endl;
    computeTotalMesh(recomputeBVH);
}

int Construction::buildGraph(double distanceThreshold) {
    size_t numParts = 0;
    std::vector<size_t> partIdxs;
    for (size_t i=0; i<w_.size(); ++i) {
        if (w_[i]) {
            ++numParts;
            partIdxs.push_back(i);
        }
    }
    g = Graph(numParts);
    size_t i=0;
    vertex_iter vi, vend;
    for (boost::tie(vi, vend) = vertices(g); vi != vend; ++vi, ++i) {
        Vertex v1 = *vi;
        g[v1].partIdx = partIdxs[i];
    }
    return findNewConnections(distanceThreshold);
}


int Construction::findNewConnections(double distanceThreshold) {
    size_t numEdges = 0;
    vertex_iter vi, vend;
    IndexMap index = get(vertex_index, g);
    for (boost::tie(vi, vend) = vertices(g); vi != vend; ++vi) {
        Vertex v1 = *vi;
        const PartData &pd1 = partData[g[v1].partIdx];
        for (vertex_iter vi2 = vi+1; vi2 != vend; ++vi2) {
            Vertex v2 = *vi2;
            if (!edge(v1, v2, g).second) {
                const PartData &pd2 = partData[g[v2].partIdx];
                double distance = distanceBetweenParts(g[v1].partIdx, g[v2].partIdx);
//                std::cout << "part distance: " << distance << std::endl;
                if (distance < distanceThreshold) {
                    std::cout << "adding " << g[v1].partIdx << '-' << g[v2].partIdx << std::endl;
                    numEdges++;
                    Edge e;
                    bool inserted;
                    boost::tie(e, inserted) = add_edge(v1, v2, g);
                    if (inserted) {
                        g[e].part1 = index[v1];
                        g[e].part2 = index[v2];
                    } else {
                        std::cout << "failed to insert edge " << index[v1] << '-' << index[v2] << '!' << std::endl;
                    }
                }
            }
        }
    }
    return numEdges;
}

MultiRay3d Construction::contactEdge(size_t partIdx1, size_t partIdx2, double offset1, double offset2, double margin, bool usePart1Shape, bool usePart2Shape) const {
    const PartData &pd1 = partData[partIdx1];
    const PartData &pd2 = partData[partIdx2];
    offset1 += pd1.offset();
    offset2 += pd2.offset();
    BoundedPlane plane1(getShape(partIdx1).cutPath, pd1.rot.conjugate().matrix(), offset1);
    BoundedPlane plane2(getShape(partIdx2).cutPath, pd2.rot.conjugate().matrix(), offset2);
    if (!usePart1Shape) {
        plane1.setCurrentShape(-1);
    }
    if (!usePart2Shape) {
        plane2.setCurrentShape(-1);
    }
    MultiRay3d intersection;
    std::cout << "intersecting " << partIdx1 << " and " << partIdx2 << " with margin " << margin << "(thicknesses " << getStock(partIdx1).thickness << ", " << getStock(partIdx2).thickness << ")" << std::endl;
    plane1.intersect(plane2, intersection, margin);
    std::cout << "intersected" << std::endl;
    return intersection;
}

bool Construction::connectionValid(ConnectionEdge &connection, double margin) {
    Vertex v1 = vertex(connection.part1, g);
    Vertex v2 = vertex(connection.part2, g);
    size_t part1 = g[v1].partIdx;
    size_t part2 = g[v2].partIdx;
    std::cout << "validating connection [" << part1 << '-' << part2 << ']' << std::endl;
    const auto &pd1 = partData[part1];
    const auto &pd2 = partData[part2];
    Vector3d n1 = pd1.normal();
    Vector3d n2 = pd2.normal();
//    Vector3d tangent = n1.cross(n2);
    double offset1 = pd1.offset();
    double offset2 = pd2.offset();
    double depth1 = getStock(part1).thickness;
    double depth2 = getStock(part2).thickness;

//    MatrixX2d points2 = getShape(part1).cutPath->points();
//    double minOffset = std::numeric_limits<double>::max();
//    double maxOffset = std::numeric_limits<double>::min();
//    for (size_t r = 0; r < points2.rows(); ++r) {
//        Vector3d pt3d = pd2.unproject(points2.row(r)).row(0).transpose();
//        double offset = pt3d.dot(tangent);
//        minOffset = std::min(minOffset, offset);
//        maxOffset = std::max(maxOffset, offset);
//    }

    MatrixX2d points1 = getShape(part1).cutPath->points();
    double minDepth1 = std::numeric_limits<double>::max();
    double maxDepth1 = std::numeric_limits<double>::min();
    //detect whether protrusion of part 1 is not too great
    for (size_t r = 0; r < points1.rows(); ++r) {
        Vector3d pt3d = pd1.unproject(points1.row(r)).row(0).transpose();
//        double offset = pt3d.dot(tangent);
//        if (offset >= minOffset + margin && offset <= maxOffset - margin) {
            double dist = pt3d.dot(n2) + offset2;
            minDepth1 = std::min(minDepth1, dist);
            maxDepth1 = std::max(maxDepth1, dist);
//        }
    }
    if (minDepth1 > maxDepth1) return false;

    connection.type = ConnectionEdge::CET_CUT_TO_FACE;
    if (-(depth2 + minDepth1) > maxDepth1) {
        connection.backface2 = true;
    } else {
        connection.backface2 = false;
    }
    return  std::min(maxDepth1, -(depth2 + minDepth1)) <= depth2;
}

MultiRay3d Construction::connectionContact(const ConnectionEdge &connection, double margin) {
    if (connection.type == ConnectionEdge::CET_UNKNOWN) return MultiRay3d();
    Vertex v1 = vertex(connection.part1, g);
    Vertex v2 = vertex(connection.part2, g);
    size_t partIdx1 = g[v1].partIdx;
    size_t partIdx2 = g[v2].partIdx;
    //set the offsets to interior planes of the corner connection
    double offset1 = !connection.backface1 ? getStock(partIdx1).thickness : 0;
    double offset2 = connection.backface2 ? getStock(partIdx2).thickness : 0;
//        Vector3d n1 = partData[partIdx1].normal();
    Vector3d n2 = partData[partIdx2].normal();
    double offset2Offset = 0;
    if (connection.type == ConnectionEdge::CET_CUT_TO_FACE || (connection.optimizeStage > 0 && connection.type == ConnectionEdge::CET_CORNER)) {
        offset2Offset = connection.backface2 ? margin*2 : -margin*2;
        std::cout << "offsetting connection [" << partIdx1 << '-' << partIdx2 << "] by " << offset2Offset << std::endl;
    }
    offset2 += offset2Offset;
    std::cout << "final offset of [" << partIdx1 << '-' << partIdx2 << "] is " << offset2 << std::endl;
    MultiRay3d mEdge = contactEdge(partIdx1, partIdx2, offset1, offset2, margin);
    if (offset2Offset != 0) {
        MultiRay3d mEdgeOrig = contactEdge(partIdx1, partIdx2, offset1, offset2-offset2Offset, 0);
        double oOrig = mEdgeOrig.o.dot(mEdge.d);
        double o = mEdge.o.dot(mEdge.d);
        mEdge.o = mEdgeOrig.o + (o - oOrig) * mEdge.d;
//        mEdge.o += n2 * offset2Offset;
    }
    return mEdge;
}

void Construction::recomputeConnectionContacts(double margin, bool useCurves) {
    if (useCurves) {
        margin /= 10;
    }
    edge_iter ei, eb;
    for (boost::tie(ei, eb) = edges(g); ei != eb; ++ei) {
        Edge e = *ei;
        g[e].innerEdge = connectionContact(g[e], margin);
        if (g[e].innerEdge.size() == 0) {
            Vertex v1 = vertex(g[e].part1, g);
            Vertex v2 = vertex(g[e].part2, g);
            size_t partIdx1 = g[v1].partIdx;
            size_t partIdx2 = g[v2].partIdx;
            std::cout << "warning: " << partIdx1 << '-' << partIdx2 << " has no contact" << std::endl;
        }
//        g[e].optimizeStage = 2;
    }
}

/*void Construction::realignConnectionEdges() {
    edge_iter ei, eb;
    for (boost::tie(ei, eb) = edges(g); ei != eb; ++ei) {
        Edge e = *ei;
        Vertex v1 = vertex(g[e].part1, g);
        Vertex v2 = vertex(g[e].part2, g);
        size_t partIdx1 = g[v1].partIdx;
        size_t partIdx2 = g[v2].partIdx;
        auto multiray = contactEdge(partIdx1, partIdx2, !g[e].backface1, g[e].backface2, 0);
        g[e].innerEdge.d = multiray.d;
    }
}*/

Edge2d getConstraintEdge(const MultiRay3d &contactEdge, size_t edgeIndex, bool reverseDir, const Vector2d &edgeNormal, double margin, double constraintOffset, const PartData &pd1, const PartData& pd2, bool &valid) {
    valid = true;
    Edge2d edge2d;
    {
        Edge3d edge3d = contactEdge.getEdge(edgeIndex);
        edge2d = Edge2d(pd1.project(edge3d.first.transpose()).transpose(), pd1.project(edge3d.second.transpose()).transpose());
        if (reverseDir) std::swap(edge2d.first, edge2d.second);
    }
    Vector2d edgeDir = edge2d.second - edge2d.first;
    double edgeLength = edgeDir.norm();
    edgeDir /= edgeLength;
//                        double minGap = std::numeric_limits<double>::max();
    double offset = edgeNormal.dot(edge2d.first);
//                        for (int r = 0; r < points.rows(); ++r) {
//                            double dist = std::abs(points.row(r).dot(edgeNormal) - offset);
//                            minGap = std::min(dist, minGap);
//                        }
//                    if (g[e].type == ConnectionEdge::CET_CORNER && reversed) {
//                        minGap += getStock(partIdx2).thickness;
//                    }

//SHRINK
//if (edgeLength <= margin * 2) {
//        valid = false;
//        return {{0,0},{0,0}};
//    }
//    edge2d.first += edgeDir * margin;
//    edge2d.second -= edgeDir * margin;
//    std::cout << "adding " << constraintOffset * edgeNormal.transpose() << std::endl;
    edge2d.first += constraintOffset * edgeNormal;
    edge2d.second += constraintOffset * edgeNormal;
    return edge2d;
}

/** find guide edge for connection constraints 1 by intersecting with connection constraints 2
 * ray1 start and end should be set to finite values based on constraint edges, and will be extended to make way for intersection */
bool guideEdge(Ray2d &ray1, const Ray2d &ray2) {
    double t1;
    ray1.intersect(ray2, t1);
    bool changed = false;
    if (t1 < ray1.start) {
        ray1.start = t1;
        changed = true;
    }
    if (t1 > ray1.end) {
        ray1.end = t1;
        changed = true;
    }
    return changed;
}

void Construction::recomputeConnectionConstraints(double margin, double max_abs_dot_product) {
    IndexMap index = get(vertex_index, g);
//    vertex_iter vi, vend;
    std::cout << "adding constraints from general exclusion" << std::endl;
    size_t N = num_vertices(g);
    for (size_t i=0; i<N; ++i) {
        Vertex v = vertex(i, g);
        size_t partIdx1 = g[v].partIdx;
        std::cout << "processing part " << partIdx1 << std::endl;
        PartData &pd1 = partData[partIdx1];
        MatrixX2d points = getShape(partIdx1).cutPath->points();
        //collect shape constraints from neighboring parts
        out_edge_iter ei, eb;
        getShape(partIdx1).connectionConstraints.clear();

        for (size_t j=0; j<N; ++j) {
            if (i == j) continue;
            Vertex v2 = vertex(j, g);
            auto pair = edge(v, v2, g);
            size_t partIdx2 = g[v2].partIdx;
            const auto &pd2 = partData[partIdx2];
            Vector3d surfaceNormal = pd2.normal();
            if (std::abs(pd1.normal().dot(surfaceNormal)) > max_abs_dot_product) continue;
            //no connection, or cut to face connection, or corner connection where part 1 is truncated
            bool noConnection = !pair.second || g[pair.first].type == ConnectionEdge::CET_UNKNOWN;
            if (noConnection || ((g[pair.first].type == ConnectionEdge::CET_CUT_TO_FACE || g[pair.first].type == ConnectionEdge::CET_CORNER) && g[pair.first].part1 == index[v])) {
                double offset1, offset2;
                std::cout << "adding bigEdge connection constraint from part " << partIdx2 << " to " << partIdx1 << std::endl;
                if (noConnection) {
                    //use heuristics to decide what side the edge is on
                    ConnectionEdge connection12;
                    connection12.part1 = index[v];
                    connection12.part2 = index[v2];
                    ConnectionEdge connection21 = connection12;
                    std::swap(connection21.part1, connection21.part2);
                    bool valid12 = connectionValid(connection12, margin);
                    bool valid21 = connectionValid(connection21, margin);
                    if (!valid12) continue;
                    if (!connection12.backface2) {
                        surfaceNormal = -surfaceNormal;
                    }
                    //"left" tangent of outward surfaceNormal should point ccw along boundary

                    offset1 = connection21.backface2 ? getStock(partIdx1).thickness : 0;
                    offset2 = connection12.backface2 ? getStock(partIdx2).thickness : 0;
                } else {
                    //use known connection
                    {
                        Vertex cv1 = vertex(g[pair.first].part1, g);
                        Vertex cv2 = vertex(g[pair.first].part2, g);
                        std::cout << "using known connection " << g[cv1].partIdx <<'-' << g[cv2].partIdx << std::endl;
                        std::cout << "backface1: " << g[pair.first].backface1 << ", backface2: " << g[pair.first].backface2 << std::endl;
                    }
                    if (!g[pair.first].backface2) {
                        surfaceNormal = -surfaceNormal;
                    }
                    offset1 = !g[pair.first].backface1 ? getStock(partIdx1).thickness : 0;
                    offset2 = g[pair.first].backface2 ? getStock(partIdx2).thickness : 0;
                    std::cout << "offset1: " << offset1 << ", offset2: " << offset2 << std::endl;
                }
                auto bigEdge = contactEdge(partIdx1, partIdx2, offset1, offset2, margin, false, true);

                Vector2d projectedNormal = pd1.projectDir(surfaceNormal).normalized();
                Vector2d tangent(-projectedNormal.y(), projectedNormal.x());
                Vector2d edgeDir2d = pd1.projectDir(bigEdge.d).normalized();
                Vector2d edgeNormal(edgeDir2d.y(), -edgeDir2d.x());
                bool reverseDir = false;
                if (pd1.projectDir(bigEdge.d).dot(tangent) < 0) {
                    edgeNormal = -edgeNormal;
                    reverseDir = true;
                }

                for (size_t edgeIndex = 0; edgeIndex < bigEdge.ranges.size(); ++edgeIndex) {
                    bool valid;
                    Edge2d edge2d = getConstraintEdge(bigEdge, edgeIndex, reverseDir, edgeNormal, margin/3,
                                                      0, pd1, pd2, valid);
                    if (!valid) continue;
                    std::cout << "added bigEdge constraint" << std::endl;
                    ConnectionConstraint cc;
                    cc.edge = std::move(edge2d);
                    cc.inside = false;
                    cc.useInCurveFit = false;
//                    cc.startGuideExtention = margin;
//                    cc.endGuideExtension = margin;
//                    cc.connection = e;
//                    cc.contactID = -1;
                    getShape(partIdx1).connectionConstraints[partIdx2].push_back(std::move(cc));
                }
            }
        }

        std::cout << "adding constraints from connections" << std::endl;
        for (boost::tie(ei, eb) = out_edges(v, g); ei != eb; ++ei) {
            Edge e = *ei;
            bool reversed = g[e].part2 == index[v];
            Vertex v2 = reversed ? vertex(g[e].part1, g) : vertex(
                    g[e].part2, g);
            size_t partIdx2 = g[v2].partIdx;
            PartData &pd2 = partData[partIdx2];
            if (g[e].type == ConnectionEdge::CET_CUT_TO_FACE ||
                g[e].type == ConnectionEdge::CET_CORNER) {
                std::cout << "adding constraints from " << partIdx1 <<'-' << partIdx2 << std::endl;
                const MultiRay3d &innerEdge = g[e].innerEdge;
                //if this part's cut is attached to the other part's face
                if (g[e].part1 == index[v] || g[e].type == ConnectionEdge::CET_CORNER) {
                    /** distance to shift the constraint edge outwards */
                    double constraintOffset = 0;
                    /** normal of the constraint plane (outwards from part 1) */
                    Vector3d surfaceNormal = pd2.normal();
                    if (reversed) {
                        if (g[e].backface1) {
                            surfaceNormal = -surfaceNormal;
                        }
                        constraintOffset = getStock(partIdx2).thickness;
                        std::cout << "shifting " << partIdx1 << " constraint from " << partIdx2 << " by " << constraintOffset << std::endl;
                    } else if (!g[e].backface2) {
                        surfaceNormal = -surfaceNormal;
                    }
                    //"left" tangent of outward surfaceNormal should point ccw along boundary
                    Vector2d projectedNormal = pd1.projectDir(surfaceNormal).normalized();
                    Vector2d tangent(-projectedNormal.y(), projectedNormal.x());
                    Vector2d edgeDir2d = pd1.projectDir(innerEdge.d).normalized();
                    Vector2d edgeNormal(edgeDir2d.y(), -edgeDir2d.x());
                    bool reverseDir = false;
                    if (pd1.projectDir(innerEdge.d).dot(tangent) < 0) {
                        edgeNormal = -edgeNormal;
                        reverseDir = true;
                    }
                    std::cout << "checking " << innerEdge.ranges.size() << " edges" << std::endl;
                    for (size_t edgeIndex=0; edgeIndex < innerEdge.ranges.size(); ++edgeIndex) {
                        bool valid;
                        Edge2d edge2d = getConstraintEdge(innerEdge, edgeIndex, reverseDir, edgeNormal, margin, constraintOffset, pd1, pd2, valid);
                        if (!valid) continue;
                        std::cout << "added regular connection constraint" << std::endl;
                        ConnectionConstraint cc;
                        cc.edge = std::move(edge2d);
                        cc.outside = false;
//                        cc.startGuideExtention = margin;
//                        cc.endGuideExtension = margin;
//                        cc.connection = e;
//                        cc.contactID = edgeIndex;
                        getShape(partIdx1).connectionConstraints[partIdx2].push_back(std::move(cc));
                    }
                }
            } else if (g[e].type == ConnectionEdge::CET_CUT_TO_CUT) {
                std::cout << "adding cut cut constraints between parts " << partIdx1 << " and " << partIdx2 << std::endl;
                ConnectionConstraint cc;
                cc.margin = margin;
                for (const auto &interface : g[e].interfaces) {
                    {
                        ConnectionConstraint cc1 = cc;
                        Vector2d interfaceDirProj = -pd1.projectDir(interface.d); //points outwards from part 1
                        if (reversed) interfaceDirProj = -interfaceDirProj;
                        Vector2d ptA = pd1.project(interface.o.transpose()).transpose();
                        Vector2d n2proj = pd1.projectDir(pd2.normal());
                        Vector2d ptB = ptA - n2proj * getStock(partIdx2).thickness;
                        Vector2d dir1 = ptB - ptA;
                        Vector2d n(dir1.y(), -dir1.x()); //right facing normal should be outwards according to interfaceDir
                        cc1.edge = Edge2d(ptA, ptB);
                        if (n.dot(interfaceDirProj) < 0) {
                            std::swap(cc1.edge.first, cc1.edge.second);
                        }
                        getShape(partIdx1).connectionConstraints[partIdx2].push_back(std::move(cc1));
                    }
                    {
                        ConnectionConstraint cc2 = cc;
                        Vector2d interfaceDirProj = pd2.projectDir(interface.d); //points outwards from part 1
                        if (reversed) interfaceDirProj = -interfaceDirProj;
                        Vector2d ptA = pd2.project(interface.o.transpose()).transpose();
                        Vector2d n1proj = pd2.projectDir(pd1.normal());
                        Vector2d ptB = ptA - n1proj * getStock(partIdx1).thickness;
                        Vector2d dir2 = ptB - ptA;
                        Vector2d n(dir2.y(), -dir2.x()); //right facing normal should be outwards according to interfaceDir
                        cc2.edge = Edge2d(ptA, ptB);
                        if (n.dot(interfaceDirProj) < 0) {
                            std::swap(cc2.edge.first, cc2.edge.second);
                        }
                        getShape(partIdx2).connectionConstraints[partIdx1].push_back(std::move(cc2));
                    }
                }
            }
        }
        std::cout << "adding guides from shape constraints" << std::endl;
        //add guides from shape constraints
        for (const auto &pair : getShape(partIdx1).shapeConstraints) {
            Vector2d n = (pair.second[0].edge.second - pair.second[0].edge.first).normalized();
            for (auto &ci : pair.second) {
                Guide guide;
                guide.edge = Edge2d(ci.edge.first - n * margin, ci.edge.second + n * margin);
                getShape(partIdx1).guides[pair.first].push_back(std::move(guide));
            }
        }
        //Stitch together guides for constraints from parts that are connected
        std::cout << "stitching together guides" << std::endl;
        for (auto &pair1 : getShape(partIdx1).connectionConstraints) {
            std::cout << "checking constraint from " << pair1.first << std::endl;
            auto v1p = partIdxToVertex(pair1.first);
            if (!v1p.second) {
                std::cout << "warning: invalid part ID for construction " << pair1.first << std::endl;
                continue;
            }
            if (v1p.second && !pair1.second.empty()) {
                //add guides from connection constraints and intersect with transitive connections to extend corners
                Ray2d ray1(Edge2d(pair1.second[0].edge.first, pair1.second[0].edge.second));
                double t1min = std::numeric_limits<double>::max();
                double t1max = std::numeric_limits<double>::lowest();
                for (auto & ci : pair1.second) {
                    t1min = std::min(t1min, ray1.project(ci.edge.first.transpose())(0));
                    t1max = std::max(t1max, ray1.project(ci.edge.second.transpose())(0));
                }
                ray1.start = t1min - margin;
                ray1.end = t1max + margin;

                std::cout << "checking for intersecting constraints from parts connected to " << pair1.first << std::endl;
                for (const auto &pair2 : getShape(partIdx1).connectionConstraints) {
                    if (pair1.first != pair2.first) {
                        auto v2p = partIdxToVertex(pair2.first);
                        std::cout << "checked for existing node: " << v2p.second << std::endl;
                        if (v2p.second && !pair2.second.empty()) {
                            auto edgepair = edge(v1p.first, v2p.first, g);
                            std::cout << "checked for edge: " << edgepair.second << std::endl;
                            if (edgepair.second && (g[edgepair.first].type == ConnectionEdge::CET_CORNER ||
                                                    g[edgepair.first].type == ConnectionEdge::CET_CUT_TO_FACE)) {
                                Ray2d ray2(Edge2d(pair2.second[0].edge.first, pair2.second[0].edge.second));
                                std::cout << "intersecting rays" << std::endl;
                                bool changed = guideEdge(ray1, ray2);
                                std::cout << "intersected" << std::endl;
                                if (changed) {
                                    std::cout << "Extending constraint from " << pair1.first << " to constraint "
                                              << pair2.first << " on part " << partIdx1 << std::endl;
                                }
                            }
                        }
                    }
                }
                std::cout << "ray1.start: " << ray1.start << ", ray1.end: " << ray1.end << std::endl;
                Guide guide;
                guide.edge = Edge2d(ray1.sample(ray1.start), ray1.sample(ray1.end));
                std::cout << "setting guide endpoint to " << guide.edge.first.transpose() << ", " << guide.edge.second.transpose() << std::endl;
                std::cout << "adding guide to index " << pair1.first << std::endl;
                getShape(partIdx1).guides[pair1.first].push_back(guide);
                std::cout << "pushed back" << std::endl;
            } else {
                std::cout << "warning: invalid part ID or empty constraint set for part " << pair1.first << " on part " << partIdx1 << std::endl;
            }
        }
    }
}

void Construction::pruneShapeConstraints() {
    size_t N = num_vertices(g);
    for (size_t i=0; i<N; ++i) {
        Vertex v = vertex(i, g);
        size_t partIdx1 = g[v].partIdx;
        const auto &connectionConstraints = getShape(partIdx1).connectionConstraints;
        auto &shapeConstraints = getShape(partIdx1).shapeConstraints;
        std::vector<size_t> removedIndices;
        for (const auto &sc : shapeConstraints) {
            if (connectionConstraints.find(sc.first) != connectionConstraints.end()) removedIndices.push_back(sc.first);
            else if (sc.first >= 0 && sc.first < partData.size()) {
                const auto &otherConnectionConstraints = getShape(sc.first).connectionConstraints;
                if (otherConnectionConstraints.find(partIdx1) != otherConnectionConstraints.end()) removedIndices.push_back(sc.first);
            }
        }
        for (auto ind : removedIndices) {
            shapeConstraints.erase(ind);
        }
    }
}

void Construction::recenter() {
    for (size_t c=0; c<partData.size(); ++c) {
        auto &pd = partData[c];
        auto &sd = getShape(c);
        Vector2d newCenter = getShape(c).cutPath->points().colwise().minCoeff().transpose();
        Vector3d shift = pd.rot * Vector3d(newCenter.x(), newCenter.y(), 0);
        pd.pos += shift;
        if (sd.cutPath)
            sd.cutPath->transform(-newCenter, 0, 1);
//        if (sd.bbox)
//            sd.bbox->transform(-newCenter, 0, 1);
//        if (sd.pointDensityContour)
//            sd.pointDensityContour->transform(-newCenter, 0, 1);
    }
}

void Construction::rotateParts() {
    vertex_iter vi, vend;
//    out_edge_iter ei, eb;

    for (boost::tie(vi, vend) = vertices(g); vi != vend; ++vi) {
        Vertex v = *vi;
        auto partIdx = g[v].partIdx;
        auto &shape = getShape(partIdx);
        double ang = 0;
        for (const auto &pair : shape.connectionConstraints) {
            //TODO: use smallest bbox one
            if (pair.second.empty()) continue;
            Vector2d dir = pair.second.front().edge.second - pair.second.front().edge.first;
            ang = std::atan2(dir.y(), dir.x());
            break;
        }
        if (ang != 0) {
            //transform all local coordinate objects
            shape.cutPath->transform(Vector2d(0, 0), -ang, 1);
            Rotation2D rot(-ang);
            for (auto &pair : shape.connectionConstraints) {
                for (auto &cc : pair.second) {
                    cc.edge.first = rot * cc.edge.first;
                    cc.edge.second = rot * cc.edge.second;
                }
            }
            for (auto &pair : shape.shapeConstraints) {
                for (auto &cc : pair.second) {
                    cc.edge.first = rot * cc.edge.first;
                    cc.edge.second = rot * cc.edge.second;
                }
            }
            for (auto &pair : shape.guides) {
                for (auto &cc : pair.second) {
                    cc.edge.first = rot * cc.edge.first;
                    cc.edge.second = rot * cc.edge.second;
                }
            }

            //transform frame
            Matrix2d mat2 = Rotation2D(ang).matrix();
            Matrix3d mat3 = Matrix3d::Identity();
            mat3.block<2, 2>(0, 0) = mat2;
            Quaterniond rotLocal(mat3);
            partData[partIdx].rot = partData[partIdx].rot * rotLocal;
        }
    }
}

void Construction::scale(double scale) {
    for (auto &pd : partData) {
        pd.pos *= scale;
    }
    for (auto &sd : shapeData) {
        if (sd.cutPath)
            sd.cutPath->transform(Vector2d(0, 0), 0, scale);
//        if (sd.bbox)
//            sd.bbox->transform(Vector2d(0, 0), 0, scale);
//        if (sd.pointDensityContour)
//            sd.pointDensityContour->transform(Vector2d(0, 0), 0, scale);
    }
    for (auto &stock : stockData) {
        stock.thickness *= scale;
    }
}

void Construction::rotate(const Eigen::Quaterniond &rot) {
    for (auto &pd : partData) {
        pd.pos = rot * pd.pos;
        pd.rot = rot * pd.rot;
    }
}

std::pair<Construction::Vertex, bool> Construction::partIdxToVertex(size_t partIdx) const {
    vertex_iter vi, vend;
    for (boost::tie(vi, vend) = vertices(g); vi != vend; ++vi) {
        Vertex v = *vi;
        if (g[v].partIdx == partIdx) {
            return {v, true};
        }
    }
    return {Construction::Vertex(), false};
}
