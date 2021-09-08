//
// Created by James Noeckel on 1/23/21.
//

#include "Construction.h"
#include <igl/opengl/glfw/Viewer.h>
#include <imgui/imgui.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
//#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include "utils/colorAtIndex.h"
//#include "utils/color_conversion.hpp"
#include "utils/macros.h"
#include "test/testUtils/curveFitUtils.h"
#include <igl/readOBJ.h>
#include "geometry/primitives3/Multimesh.h"

using namespace Eigen;
using namespace boost;

void Construction::visualize(double displacement, const std::string &connector_mesh, double spacing, double scale) const {
    igl::opengl::glfw::Viewer viewer;
    viewer.data().point_size = 5;
    viewer.data().show_overlay = true;
    viewer.data().label_color.head(3) = Vector3f(1, 1, 1);

    Multimesh cMesh;
    cMesh.AddMesh(std::make_pair(std::get<0>(mesh), std::get<1>(mesh)));
    if (!connector_mesh.empty()) {
        cMesh.AddMesh(connectorMesh(connector_mesh, spacing, scale));
    }
    auto cMeshOut = cMesh.GetTotalMesh();

    MatrixX3d C = MatrixX3d::Ones(cMeshOut.second.rows(), 3) * 0.5;
    for (size_t i=0; i<std::get<1>(mesh).rows(); ++i) {
        C.row(i) = colorAtIndex(std::get<2>(mesh)(i), partData.size());
    }
    viewer.data().set_mesh(cMeshOut.first, cMeshOut.second);
    viewer.data().set_colors(C);
    std::cout << "labeling parts" << std::endl;
    for (size_t c=0; c<partData.size(); ++c) {
        if (w_[c]) {
            MatrixX2d points = getShape(c).cutPath->points();
            MatrixX3d points3d(points.rows(), 3);
            for (size_t r=0; r<points.rows(); ++r) {
                points3d.row(r) = (partData[c].rot * Vector3d(points(r, 0), points(r, 1), 0) + partData[c].pos).transpose();
            }
            RowVector3d centroid = 0.5 * (points3d.colwise().maxCoeff() + points3d.colwise().minCoeff());
            viewer.data().add_label(centroid, "PART " + std::to_string(c));
            viewer.data().add_points(centroid, RowVector3d(0, 0, 1));
            viewer.data().add_edges(centroid, centroid + partData[c].normal().transpose() * getStock(c).thickness, RowVector3d(0, 0, 1));
            std::cout << "part " << c << " guides: ";
            for (const auto &pair : getShape(c).guides) {
                for (const auto &guide : pair.second) {
                    std::cout << pair.first << ", ";
                    RowVector3d ptA = partData[c].unproject(guide.edge.first.transpose());
                    RowVector3d ptB = partData[c].unproject(guide.edge.second.transpose());
                    viewer.data().add_edges(ptA, ptB, RowVector3d(0.5, 0.5, 0.5));
                }
            }
            std::cout << std::endl;
            std::cout << "part " << c << " connection constraint edges: ";
            for (const auto &pair : getShape(c).connectionConstraints) {
                for (const auto &constraint : pair.second) {
                    std::cout << pair.first << " (";
                    if (constraint.inside) std::cout << "I";
                    if (constraint.outside) std::cout << "O";
                    std::cout << "), ";
                    RowVector3d ptA = partData[c].unproject(constraint.edge.first.transpose());
                    RowVector3d ptB = partData[c].unproject(constraint.edge.second.transpose());
                    ptA += partData[c].normal() * displacement;
                    ptB += partData[c].normal() * displacement;
                    viewer.data().add_edges(ptA, ptB, colorAtIndex(c, partData.size()) * 0.5);
                    viewer.data().add_points(ptA, colorAtIndex(c, partData.size()) * 0.5);
                }
            }
            std::cout << std::endl;

            std::cout << "part " << c << " shape constraint edges: ";
            for (const auto &sc : getShape(c).shapeConstraints) {
                for (const auto &constraint : sc.second) {
                    std::cout << sc.first << " (";
                    std::cout << (constraint.convex ? 'x' : 'v');
                    if (constraint.inside) std::cout << "I";
                    if (constraint.outside) std::cout << "O";
                    std::cout << "), ";
                    RowVector3d ptA = partData[c].unproject(constraint.edge.first.transpose());
                    RowVector3d ptB = partData[c].unproject(constraint.edge.second.transpose());
                    if (constraint.opposing) {
                        ptA -= partData[c].normal() * getStock(c).thickness;
                        ptB -= partData[c].normal() * getStock(c).thickness;
                        ptA -= partData[c].normal() * displacement;
                        ptB -= partData[c].normal() * displacement;
                    } else {
                        ptA += partData[c].normal() * displacement;
                        ptB += partData[c].normal() * displacement;
                    }
                    viewer.data().add_edges(ptA, ptB, colorAtIndex(c, partData.size()));
                }
            }
            std::cout << std::endl;
        }
    }
    edge_iter ei, eb;
    std::cout << "connections: " << std::endl;
    for (boost::tie(ei, eb) = edges(g); ei != eb; ++ei) {
        Edge e = *ei;
        Vertex v1 = vertex(g[e].part1, g);
        Vertex v2 = vertex(g[e].part2, g);
        size_t partIdx1 = g[v1].partIdx;
        size_t partIdx2 = g[v2].partIdx;
        const auto &pd1 = partData[partIdx1];
        const auto &pd2 = partData[partIdx2];
        Vector3d n1 = pd1.normal();
        Vector3d n2 = pd2.normal();
        RowVector3d offset = (n1 + n2).transpose() * displacement;
        //visualize closeness
        /*RowVector3d point1 = partMeshes[partIdx1].first.colwise().mean();
        RowVector3d point2 = partMeshes[partIdx2].first.colwise().mean();*/
        RowVector3d point1, point2;
//        double dist = distanceBetweenParts(partIdx1, partIdx2, point1, point2);
        viewer.data().add_edges(point1, point2, colorAtIndex(partIdx1, partData.size()));
        viewer.data().add_edges(point1 + offset, point2 + offset, colorAtIndex(partIdx2, partData.size()));
        viewer.data().add_label(0.5 * (point1 + point2), "(n.n=" + std::to_string(n1.dot(n2)) + ")");
        //visualize connections
        if (g[e].type == ConnectionEdge::CET_UNKNOWN) {
            continue;
        }
        std::cout << "connection [" << partIdx1 << '-' << partIdx2 << "] type " << g[e].type  << " backface1: " << g[e].backface1 << "; backface2: " << g[e].backface2 << std::endl;
        if (g[e].type == ConnectionEdge::CET_CUT_TO_CUT) {
//            std::cout << "building interface plane meshes" << std::endl;
            for (const auto &interface : g[e].interfaces) {
                Eigen::Vector3d up(0, 0, 0);
                int ind = 0;
                double mincomp = std::abs(interface.d[0]);
                for (int i = 1; i < 3; i++) {
                    double abscomp = std::abs(interface.d[i]);
                    if (abscomp < mincomp) {
                        ind = i;
                        mincomp = abscomp;
                    }
                }
                up[ind] = 1.0f;
                Eigen::Matrix<double, 2, 3> basis;
                basis.row(1) = interface.d.cross(up).normalized();
                basis.row(0) = basis.row(1).transpose().cross(interface.d).normalized();


                MatrixX3d V1(4, 3);
                MatrixX3d V2(4, 3);

                V1.row(0) = interface.o.transpose() - basis.row(0) * displacement - basis.row(1) * displacement;
                V1.row(1) = interface.o.transpose() - basis.row(0) * displacement + basis.row(1) * displacement;
                V1.row(2) = interface.o.transpose() + basis.row(0) * displacement + basis.row(1) * displacement;
                V1.row(3) = interface.o.transpose() + basis.row(0) * displacement - basis.row(1) * displacement;
                V2.block(0, 0, 3, 3) = V1.block(1, 0, 3, 3);
                V2.row(3) = V1.row(0);
                viewer.data().add_edges(V1, V2, RowVector3d(1.0, 0.5, 0.0));
            }
        } else {
            for (size_t edgeIndex=0; edgeIndex < g[e].innerEdge.ranges.size(); ++edgeIndex) {
                Edge3d edge3d = g[e].innerEdge.getEdge(edgeIndex);
                viewer.data().add_edges(edge3d.first.transpose() + offset, edge3d.second.transpose() + offset, RowVector3d(1, 0.5, 1));
                Edge3d edge1 = edge3d;
                Edge3d edge2 = edge3d;
                Vector3d surfaceNormal1 = n1;
                Vector3d surfaceNormal2 = n2;
                if (g[e].backface1) {
                    surfaceNormal1 = -surfaceNormal1;
                }
                if (!g[e].backface2) {
                    surfaceNormal2 = -surfaceNormal2;
                }
                edge2.first += surfaceNormal1 * getStock(partIdx1).thickness;
                edge2.second += surfaceNormal1 * getStock(partIdx1).thickness;
                if (g[e].type == ConnectionEdge::CET_CORNER) {
                    edge1.first -= surfaceNormal1 * getStock(partIdx1).thickness;
                    edge1.second -= surfaceNormal1 * getStock(partIdx1).thickness;
                }
                std::cout << "building plane meshes" << std::endl;
                MatrixX3d V1(5, 3);
                MatrixX3d V2(5, 3);
                V1.row(0) = edge1.first;
                V1.row(1) = edge1.second;
                V1.row(2) = edge2.second;
                V1.row(3) = edge2.first;
                V2.block(0, 0, 3, 3) = V1.block(1, 0, 3, 3);
                V2.row(3) = V1.row(0);
                V1.row(4) = edge1.first;
                V2.row(4) = edge2.second;
                viewer.data().add_edges(V1, V2, RowVector3d(0.0, 0.5, 1.0));
            }
        }
    }
    std::cout << "displaying" << std::endl;
    viewer.core().background_color = {1, 1, 1, 0};
//    if (openWindow) {
        igl::opengl::glfw::imgui::ImGuiMenu menu;
        menu.callback_draw_viewer_window = []() {};
        viewer.plugins.push_back(&menu);
        viewer.launch();
//    }
}

bool Construction::exportMesh(const std::string &filename, const std::string &connectorFilename, double connectorSpacing, double connectorScale) const {
    std::cout << "exporting mesh to " << filename << std::endl;
    Multimesh cmesh;
    cmesh.AddMesh(std::make_pair(std::get<0>(mesh), std::get<1>(mesh)));
    if (!connectorFilename.empty()) {
        cmesh.AddMesh(connectorMesh(connectorFilename, connectorSpacing, connectorScale));
    }
    auto cmesho = cmesh.GetTotalMesh();
    return igl::writeOBJ(filename, cmesho.first, cmesho.second);
}

void writeCurve(std::ostream &file, const CombinedCurve &curves) {
    for (size_t j=0; j<curves.size(); ++j) {
        const auto &curve = curves.getCurve(j);
        file << "<segment type=\"" << static_cast<std::underlying_type<CurveTypes>::type>(curve.type()) << "\">" << std::endl;
        MatrixX2d points = curve.uniformSample(20);
        for (int k=0; k < points.rows(); k++) {
            file << "<point2 value=\"" << points.row(k) << "\"/>" << std::endl;
        }
        file << "<point2 value=\"" << curve.sample(1).transpose() << "\"/>" << std::endl;
        file << "</segment>" << std::endl;
    }
}

bool Construction::exportPart(size_t partIdx, std::ofstream &of, LoadMode mode) const {
    IndexMap index = get(vertex_index, g);
    const auto &pd = partData[partIdx];
    if (mode == SELECTION) {
        of << (w_[partIdx] ? 1 : 0) << std::endl;
    }
    std::cout << "saving thickness " << getStock(partIdx).thickness << std::endl;
    of << getStock(partIdx).thickness;
    if (mode == SEGMENTATION) {
        of << " " << getShape(partIdx).gridSpacing;
    }
    of << std::endl;
    of << pd.rot.w() << " " << pd.rot.x() << " " << pd.rot.y() << " " << pd.rot.z() << std::endl;
    of << pd.pos.transpose() << std::endl;
    const auto &shape = getShape(partIdx).cutPath;
    if (!shape->curves().exportPlaintext(of)) {
        std::cout << "failed exporting curve for part " << partIdx << std::endl;
        return false;
    }
    const auto &children = shape->children();
    of << children.size() << " children" << std::endl;
    for (const auto &child : children) {
        if (!child->curves().exportPlaintext(of)) {
            std::cout << "failed exporting child of part " << partIdx << std::endl;
            return false;
        }
    }
    const auto &shapeConstraints=getShape(partIdx).shapeConstraints;
    size_t numSC = 0;
    for (const auto &pair : shapeConstraints) {
        numSC += pair.second.size();
    }
    if (mode != BASIC && mode != BASICPLUS) {
        of << numSC << " shape constraints" << std::endl;
        for (const auto &pair : shapeConstraints) {
            int id;
            if (mode == SELECTION) {
                id = pair.first;
            } else {
                auto v = partIdxToVertex(pair.first);
                id = v.second ? (int) index[v.first] : -1;
            }
            for (const auto &sc : pair.second) {
                of << id << " " << sc.convex << " " << sc.edge.first.transpose() << " " << sc.edge.second.transpose()
                   << " " << sc.inside << " " << sc.opposing << " " << sc.otherOpposing << std::endl;
            }
        }
    }

    return true;
}

bool Construction::exportPlaintext(const std::string &filename, LoadMode mode) const {
    std::cout << "saved checkpoint file at " << filename << std::endl;
    std::ofstream of(filename);
    if (of) {
        of << std::setprecision(20);
        if (mode == SELECTION) {
            size_t N = partData.size();
            of << N << std::endl;
            for (size_t i=0; i<N; ++i) {
                if (!exportPart(i, of, mode)) return false;
            }
        } else {
            size_t N = num_vertices(g);
            of << N << " parts" << std::endl;
            for (size_t i = 0; i < N; ++i) {
                Vertex v = vertex(i, g);
                size_t partIdx = g[v].partIdx;
                if (!exportPart(partIdx, of, mode)) return false;
            }

            size_t N_c = 0;
            edge_iter ei, eb;
            for (boost::tie(ei, eb) = edges(g); ei != eb; ++ei) {
                Edge e = *ei;
                if (g[e].type != ConnectionEdge::CET_UNKNOWN) {
                    ++N_c;
                }
            }
            of << N_c << " connections" << std::endl;
            for (boost::tie(ei, eb) = edges(g); ei != eb; ++ei) {
                Edge e = *ei;
                if (g[e].type != ConnectionEdge::CET_UNKNOWN) {
                    of << g[e].part1 << " " << g[e].part2 << " " << g[e].type << " " << g[e].backface1 << " "
                       << g[e].backface2 << std::endl;
                    const auto &innerEdge = g[e].innerEdge;
                    of << innerEdge.o.transpose() << " " << innerEdge.d.transpose() << std::endl;
                    of << innerEdge.size() << std::endl;
                    for (int i = 0; i < innerEdge.size(); ++i) {
                        of << innerEdge.ranges[i].first << " " << innerEdge.ranges[i].second << std::endl;
                    }
                }
            }
        }
    } else {
        std::cout << "could not export plaintext to " << filename << std::endl;
        return false;
    }
    std::cout << "saved checkpoint file at " << filename << std::endl;
    return true;
}

bool Construction::loadPlaintext(const std::string &filename, LoadMode mode) {
    bool curveConstraints = (mode != BASIC);
    std::cout << "loading from file " << filename << std::endl;
    std::ifstream ifs(filename);
    if (ifs) {
        size_t N;
        std::string line;
        std::getline(ifs, line);
        {
            std::istringstream is_line(line);
            is_line >> N;
        }
        std::cout << "loading " << N << " parts" << std::endl;
        partData.reserve(N);
        shapeData.reserve(N);
        stockData.reserve(N);
        w_ = std::vector<bool>(N, true);
        if (mode != SELECTION) {
            g = Graph(N);
        }
        for (int i=0; i<N; ++i) {
            std::cout << "loading part " << i << std::endl;
            if (mode != SELECTION) {
                Vertex v = vertex(i, g);
                g[v].partIdx = i;
            } else {
                int isUsed;
                std::getline(ifs, line);
                {
                    std::istringstream is_line(line);
                    is_line >> isUsed;
                    LINE_FAIL("failed to read used state");
                }
                w_[i] = isUsed;
            }
            PartData pd;
            pd.shapeIdx = i;
            if (i== N-1) pd.groundPlane = true;
            ShapeData sd;
            sd.stockIdx = i;
            StockData stock;
            //get thickness
            std::getline(ifs, line);
            {
                std::istringstream is_line(line);
                is_line >> stock.thickness;
                LINE_FAIL("failed to parse thickness");
                std::cout << "thickness: " << stock.thickness << std::endl;
                if (mode == SEGMENTATION) {
                    is_line >> sd.gridSpacing;
//                    LINE_FAIL("failed to parse grid spacing");
                }
            }
            //get rotation
            std::getline(ifs, line);
            {
                std::istringstream is_line(line);
                is_line >> pd.rot.w() >> pd.rot.x() >> pd.rot.y() >> pd.rot.z();
            }
            //get translation
            std::getline(ifs, line);
            {
                std::istringstream is_line(line);
                is_line >> pd.pos.x() >> pd.pos.y() >> pd.pos.z();
            }
            CombinedCurve outerCurve;
            if (!outerCurve.loadPlaintext(ifs, curveConstraints)) {
                std::cout << "failed loading curve for part " << i << std::endl;
                return false;
            }
            size_t numChildren;
            std::getline(ifs, line);
            {
                std::istringstream is_line(line);
                is_line >> numChildren;
            }
            std::cout << numChildren << " child curves for part " << i << std::endl;
            std::vector<std::shared_ptr<Primitive>> holes;
            for (size_t j=0; j<numChildren; ++j) {
                CombinedCurve childCurve;
                if (!childCurve.loadPlaintext(ifs, curveConstraints)) {
                    std::cout << "failed loading curve for part " << i << " child " << j << std::endl;
                    return false;
                }
                holes.emplace_back(new PolyCurveWithHoles(childCurve));
            }
            sd.cutPath = std::make_shared<PolyCurveWithHoles>(outerCurve, std::move(holes));

            if (mode != BASIC && mode != BASICPLUS) {
                size_t numShapeConstraints = 0;
                std::getline(ifs, line);
                {
                    std::istringstream is_line(line);
                    is_line >> numShapeConstraints;
                }
                std::cout << "loading " << numShapeConstraints << " shape constraints" << std::endl;
                for (size_t j=0; j<numShapeConstraints; ++j) {
                    ShapeConstraint sc;
                    int id;
                    std::getline(ifs, line);
                    {
                        std::istringstream is_line(line);
                        int convex, inside, opposing, otherOpposing;
                        is_line >> id >> convex >> sc.edge.first.x() >> sc.edge.first.y() >> sc.edge.second.x() >> sc.edge.second.y() >> inside >> opposing >> otherOpposing;
                        sc.convex = convex;
                        sc.inside = inside;
                        sc.opposing = opposing;
                        sc.otherOpposing = otherOpposing;
                        LINE_FAIL("failed reading shape constraint");
                    }
                    if (id < 0) {
                        std::cout << "shape constraint " << j << " is " << id << std::endl;
                    }
                    sd.shapeConstraints[id].push_back(std::move(sc));
                }
            }

            partData.push_back(std::move(pd));
            shapeData.push_back(std::move(sd));
            stockData.push_back(std::move(stock));
        }

        if (mode != SELECTION) {
            size_t N_c;
            std::getline(ifs, line);
            {
                std::istringstream is_line(line);
                is_line >> N_c;
            }
            std::cout << N_c << " connections" << std::endl;
            for (size_t i = 0; i < N_c; ++i) {
                int id1, id2, type;
                int backface1, backface2;
                std::getline(ifs, line);
                {
                    std::istringstream is_line(line);
                    is_line >> id1 >> id2 >> type >> backface1 >> backface2;
                    LINE_FAIL("failed to parse connection " + std::to_string(i));
                }
                std::cout << "connection " << i << ": " << id1 << '-' << id2 << " type " << type;
                std::cout << '(' << line << ')' << std::endl;
                std::cout << "backface1: " << backface1 << "; backface2: " << backface2 << std::endl;
                MultiRay3d innerEdge;
                std::getline(ifs, line);
                {
                    std::istringstream is_line(line);
                    is_line >> innerEdge.o.x() >> innerEdge.o.y() >> innerEdge.o.z() >> innerEdge.d.x()
                            >> innerEdge.d.y() >> innerEdge.d.z();
                }
                size_t numEdges;
                std::getline(ifs, line);
                {
                    std::istringstream is_line(line);
                    is_line >> numEdges;
                }
                std::cout << "num edges: " << numEdges << std::endl;
                for (size_t j = 0; j < numEdges; ++j) {
                    std::pair<double, double> range;
                    std::getline(ifs, line);
                    {
                        std::istringstream is_line(line);
                        is_line >> range.first >> range.second;
                    }
                    innerEdge.ranges.push_back(range);
                }
                Edge e;
                bool inserted;
                Vertex v1 = vertex(id1, g);
                Vertex v2 = vertex(id2, g);
                boost::tie(e, inserted) = add_edge(v1, v2, g);
                if (!inserted) {
                    std::cout << "failed to insert edge" << std::endl;
                }
                g[e].innerEdge = std::move(innerEdge);
                g[e].part1 = id1;
                g[e].part2 = id2;
                g[e].backface1 = backface1;
                g[e].backface2 = backface2;
                g[e].optimizeStage = 1;
                g[e].type = static_cast<ConnectionEdge::ConnectionType>(type);
            }
        }
    } else {
        std::cout << "failed to open " << filename << std::endl;
        return false;
    }
    return true;
}

bool Construction::exportModel(const std::string &filename) const {
    std::cout << "exporting model to " << filename << std::endl;
    std::ofstream file(filename);
    if (file) {
        file << "<solution>" << std::endl;
        size_t N = num_vertices(g);
        for (size_t i=0; i<N; ++i) {
            Vertex v = vertex(i, g);
            size_t partIdx = g[v].partIdx;
            const auto &pd = partData[partIdx];
            file << "<part id=\"" << partIdx
                 << "\" depth=\"" << getStock(partIdx).thickness
                 << "\" rotation=\"" << pd.rot.w() << " " << pd.rot.x() << " " << pd.rot.y() << " " << pd.rot.z()
                 << "\" translation=\"" << pd.pos.x() << " " << pd.pos.y() << " " << pd.pos.z() << "\">" << std::endl;
            file << "<shape>" << std::endl;
            {
                const auto &curves = getShape(partIdx).cutPath->curves();
                CombinedCurve backCurves = getBackCurve(partIdx);
                std::ofstream svgFile(filename + "-part" + std::to_string(partIdx) + ".svg");
                svgFile << R"(<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">)" << std::endl;
                std::cout << "exporting SVG for part " << partIdx << std::endl;
                curves.exportSVG(svgFile);
                curves.exportSVG(file);
                backCurves.exportSVG(svgFile);

                const auto &children = getShape(partIdx).cutPath->children();
                for (const auto &child : children) {
                    file << "<hole>" << std::endl;
                    child->curves().exportSVG(svgFile);
                    child->curves().exportSVG(file);
//                writeCurve(file, child->curves());
                    file << "</hole>" << std::endl;
                }
                svgFile << "</svg>" << std::endl;
            }

            file << "</shape>";
//            writeCurve(file, curves);
            file << "</part>" << std::endl;

        }

        edge_iter ei, eb;
        for (boost::tie(ei, eb) = edges(g); ei != eb; ++ei) {
            Edge e = *ei;
            if (g[e].type != ConnectionEdge::CET_UNKNOWN) {
                Vertex v1 = vertex(g[e].part1, g);
                Vertex v2 = vertex(g[e].part2, g);
                file << "<connection id1=\"" << g[v1].partIdx << "\" id2=\"" << g[v2].partIdx << "\" type=\""
                     << g[e].type << "\"/>" << std::endl;
            }
        }

        file << "</solution>" << std::endl;
    } else {
        std::cout << "could not open file " << filename << std::endl;
        return false;
    }
    return true;
}

void Construction::saveShapes() const {
    size_t N = num_vertices(g);
    for (size_t i=0; i<N; ++i) {
        Vertex v = vertex(i, g);
        size_t partIdx = g[v].partIdx;

        CombinedCurve curve = getShape(partIdx).cutPath->curves();
        Vector2d minPt;
        double scale = computeScale(curve, minPt);
        std::string name = "part_" + std::to_string(i) + "_curve.png";
        cv::Mat img = display_curve(name, curve, scale, minPt);
        cv::imwrite(name, img);
    }
}

std::pair<Eigen::MatrixX3d, Eigen::MatrixX3i> Construction::connectorMesh(const std::string &connectorOBJ, double spacingd, double scale) const {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    Multimesh cmesh;

    if (!igl::readOBJ(connectorOBJ, V, F)) {
        std::cout << "failed to load connector OBJ file " << connectorOBJ << std::endl;
        return {V, F};
    }

    V.array() *= scale;

    edge_iter ei, eb;
    for (boost::tie(ei, eb) = edges(g); ei != eb; ++ei) {
        Edge e = *ei;
        Vertex v1 = vertex(g[e].part1, g);
        Vertex v2 = vertex(g[e].part2, g);
        size_t partIdx1 = g[v1].partIdx;
        size_t partIdx2 = g[v2].partIdx;
        const auto &pd1 = partData[partIdx1];
        const auto &pd2 = partData[partIdx2];
        if (pd2.groundPlane) continue;
        Vector3d n1 = pd1.normal();
        Vector3d n2 = pd2.normal();

        double thickness = getStock(partIdx1).thickness;
        double thickness2 = getStock(partIdx2).thickness;

        Vector3d cutDir = (n1 - n1.dot(n2) * n2).normalized();
        Vector3d nailDir = (n2 - n2.dot(n1) * n1).normalized();

        if (g[e].backface2) nailDir = -nailDir;

        const auto &innerEdge = g[e].innerEdge;

        double angSin = n1.cross(n2).norm();
        double midCut = 0.5 * thickness / angSin;
        double nailDepth = thickness2 / angSin;

        Eigen::Quaterniond rot = Eigen::Quaterniond::FromTwoVectors(Vector3d(0, 0, 1), nailDir);

        for (int i=0; i<innerEdge.ranges.size(); ++i) {
            double len = innerEdge.ranges[i].second - innerEdge.ranges[i].first;
            if (len < spacingd*2) {
                continue;
            }
            for (int j=0; j<2; ++j) {
                double startOffset;
                if (j == 0) {
                    startOffset = spacingd;
                } else {
                    startOffset = len - spacingd;
                }
                Vector3d currPos = innerEdge.o + innerEdge.d * (innerEdge.ranges[i].first + startOffset) - cutDir * midCut;
                MatrixXd nailV = V;
                nailV.col(2).array() -= nailDepth;
                for (int r=0; r<V.rows(); ++r) {
                    Vector3d p = nailV.row(r).transpose();
                    nailV.row(r) = (rot * p).transpose();
                }
                nailV.rowwise() += currPos.transpose();

                cmesh.AddMesh(std::make_pair(nailV, F));
            }
        }
    }

    return cmesh.GetTotalMesh();
}