//
// Created by James Noeckel on 7/9/20.
//

#include "Solver.h"
#include <iostream>
#include <numeric>
#include "FeatureExtractor.h"
#include "utils/timingMacros.h"
#include "test/testUtils/displaySurfaceCompletion.h"
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include "utils/color_conversion.hpp"
#include "utils/cyclicPermuteRows.h"
#include "geometry/primitives3/SurfaceCompletion.h"
#include "imgproc/dda_foreach.h"
#include "utils/colorAtIndex.h"
/*#include <igl/fast_winding_number.h>
#include <igl/octree.h>
#include <algorithm>
#include <geometry/VoxelGrid.hpp>
#include <igl/copyleft/cgal/point_areas.h>
#include <igl/knn.h>
#include <igl/copyleft/marching_cubes.h>*/

//#define USE_CURVES
#define MAX_GEOCUT_ITERS 3

using namespace Eigen;

//void Solver::initialize(double epsilon, double voxel_width, double norm_adjacency_threshold, double support, double probability, bool cylinders, double cluster_epsilon, float normal_threshold, uint seed) {
void Solver::initialize() {
    if (!cloud_) {
        std::cout << "warning: nullpointer point cloud" << std::endl;
    }
    std::cout << "initializing feature extractor... ";
    ReconstructionData::Handle reconstruction(new ReconstructionData);
    FeatureExtractor features(cloud_, reconstruction, random_);
    std::cout << "done" << std::endl;
    if (!settings_.globfit_import.empty()) {
        std::cout << "importing globfit" << std::endl;
        if (!features.import_globfit(settings_.globfit_import, settings_.support/100, diameter_)) {
            std::cout << "warning: Failed to load globfit file " << settings_.globfit_import << std::endl;
        }
        std::cout << "loaded" << std::endl;
    }


    //parameter initialization
    RowVector3d bboxMin = cloud_->P.colwise().minCoeff();
    RowVector3d bboxMax = cloud_->P.colwise().maxCoeff();
    RowVector3d bbox = bboxMax-bboxMin;
    double threshold = diameter_/settings_.master_resolution;
    double cluster_epsilon = settings_.clustering_factor < 0 ? settings_.clustering_factor : threshold * settings_.clustering_factor;
    double voxel_width = diameter_ / settings_.voxel_resolution;
    double adjacency_threshold = voxel_width * settings_.adjacency_factor;
    double bezier_cost = settings_.curve_cost * diameter_ * diameter_ * settings_.voxel_resolution;
    double line_cost = settings_.line_cost * diameter_ * diameter_ * settings_.voxel_resolution;
    double thickness_spacing = diameter_ / settings_.thickness_resolution * 1.01;
    std::cout << "bounding box: min " << bboxMin << ", max: " << bboxMax << " (diameter " << diameter_ << ')' << std::endl;
    std::cout << "ransac epsilon: " << threshold << std::endl;
    std::cout << "ransac cluster epsilon: " << cluster_epsilon << std::endl;
    std::cout << "voxel_width: " << voxel_width << std::endl;

    /*for (size_t i=0; i<cloud->size(); i++) {
        std::cout << "normal " << i << ": " << (*cloud)[i].getNormalVector3fMap().transpose() << std::endl;
    }*/

    if (settings_.globfit_import.empty()) {
        std::cout << "detecting primitives..." << std::endl;
        features.detect_primitives(threshold, settings_.support / 100, settings_.probability, settings_.use_cylinders,
                                   cluster_epsilon, settings_.normal_threshold);
        std::cout << "found " << features.planes.size() << " planes and " << features.cylinders.size() << " cylinders"
                  << std::endl;
    }
    if (settings_.visualize) {
        igl::opengl::glfw::Viewer viewer;
        int nPts = cloud_->P.rows() / settings_.visualization_stride;
        MatrixX3d colors(nPts, 3);
        MatrixX3d visPoints(nPts, 3);
        for (int i=0; i<nPts; ++i) {
            int p = i * settings_.visualization_stride;
            int c = features.point_labels[p];
            colors.row(i) = c >= 0 ? colorAtIndex(c, features.planes.size()) : RowVector3d(1, 1, 1);
//            colors.row(i) = cloud_->N.row(i).array()*0.5+0.5;
            visPoints.row(i) = cloud_->P.row(p);
        }
        viewer.data().add_points(visPoints, colors);
        viewer.data().point_size = 2;
        viewer.core().align_camera_center(visPoints);
        viewer.core().background_color = {1, 1, 1, 1};
        viewer.launch();
    }

    if (!settings_.globfit_export.empty() && settings_.globfit_import.empty()) {
        features.export_globfit(settings_.globfit_export, settings_.visualization_stride, 1.0/diameter_);
    }

    std::cout << "finding oriented bounding boxes" << std::endl;
    features.detect_bboxes();
    if (settings_.use_winding_number) {
        std::cout << "precomputing winding number field" << std::endl;
        features.compute_winding_number(settings_.winding_number_stride, settings_.k_n);
        std::cout << "detecting contours with winding number" << std::endl;
        features.detect_contours(voxel_width, settings_.contour_threshold, settings_.max_contour_hole_ratio, settings_.use_winding_number);
    }
    std::cout << "detecting point density contour" << std::endl;
    features.detect_contours(voxel_width, settings_.contour_threshold, settings_.max_contour_hole_ratio, false, settings_.use_geocuts ? 2 : 0, settings_.debug_visualization);
    if (settings_.use_geocuts) {
        std::cout << "generating minimal surface" << std::endl;
        features.setCurrentShape(CONTOUR_ERODED_ID);
        SurfaceCompletion surf(bboxMin.transpose().array() - 3 * voxel_width,
                               bboxMax.transpose().array() + 3 * voxel_width, voxel_width, 200);
        surf.setPrimitives(features.planes);
        //TODO: add support for cylinders
        std::cout << "constraining lines of sight" << std::endl;
        for (const auto &pair : reconstruction_->images) {
            Vector3d start = pair.second.origin();
            start -= surf.minPt();
            start /= surf.spacing();
            std::cout << "adding " << pair.second.point3D_ids_.size() << " rays from camera " << pair.first << std::endl;
            for (int64_t pid : pair.second.point3D_ids_) {
                Vector3d endpoint = reconstruction_->points[pid].xyz_;
                endpoint -= (endpoint-start).normalized() * (voxel_width * 2);
                endpoint -= surf.minPt();
                endpoint /= surf.spacing();
                auto lambda = [&](int iCurr, int jCurr, int kCurr) {
                    if (iCurr >= 0 && iCurr < surf.resolution().x() && jCurr >= 0 && jCurr < surf.resolution().y() && kCurr >= 0 && kCurr < surf.resolution().z()) {
                        surf.addOutsideConstraint(iCurr, jCurr, kCurr);
                    }
                };
                dda_foreach(lambda, start.x(), start.y(), start.z(), endpoint.x(), endpoint.y(), endpoint.z());
                /*int i=(int)std::round(endpoint.x()), j=(int)std::round(endpoint.y()), k=(int)std::round(endpoint.z());
                if (i >= 0 && i < surf.resolution().x() && j >= 0 && j < surf.resolution().y() && k >= 0 && k < surf.resolution().z()) {
                    surf.addOutsideConstraint(i, j, k);
                }*/
            }
            //DEBUG:
//            break;
        }
        {
            //visualize constraints
            Visualizer visualizer;
            visualizer.visualize_primitives(features.planes, std::vector<Cylinder>());
            displaySegmentation(surf, visualizer, VIS_OUTSIDE);
            visualizer.launch();
        }
        std::cout << "initializing problem with resolution [" << surf.resolution().transpose() << ']' << std::endl;
        DECLARE_TIMING(surfInit);
        START_TIMING(surfInit);
        surf.constructProblem(5 * voxel_width, 30 * voxel_width);
        STOP_TIMING(surfInit);
        PRINT_TIMING(surfInit);

//        std::cout << "inside constraints: " << surf.insideConstraints().size() << std::endl;
//        std::cout << "outside constraints: " << surf.outsideConstraints().size() << std::endl;
        for (int iter=0; iter<MAX_GEOCUT_ITERS; ++iter) {
            std::cout << "running max flow" << std::endl;
            DECLARE_TIMING(maxflow);
            START_TIMING(maxflow);
            float cost = surf.maxflow();
            STOP_TIMING(maxflow);
            PRINT_TIMING(maxflow);
            std::cout << "max flow: " << cost << std::endl;
            {
                Visualizer visualizer;
                visualizer.visualize_primitives(features.planes, std::vector<Cylinder>());
                displaySegmentation(surf, visualizer);
                visualizer.launch();
            }
            if (iter < MAX_GEOCUT_ITERS-1) {
                std::cout << "marking violating edges" << std::endl;
                DECLARE_TIMING(mark);
                START_TIMING(mark);
                surf.markViolatingEdges();
                STOP_TIMING(mark);
                PRINT_TIMING(mark);
            }
        }
    }

    int contour_name = settings_.use_winding_number ? CONTOUR_WN_ID : CONTOUR_DENSITY_ID;

    std::cout << "finding adjacencies... ";
    features.setCurrentShape(contour_name); // use marching squares contour to find adjacency
    features.detect_adjacencies(adjacency_threshold, settings_.norm_adjacency_threshold);
    std::cout << "adjacencies:" << std::endl;
    for (size_t c = 0; c < features.adjacency.size(); c++) {
        std::cout << "cluster " << c << "(" << features.adjacency[c].size() << "): ";
        for (const auto &pair : features.adjacency[c]) {
            unsigned long adj = pair.first;
            std::cout << adj;
            if (!pair.second.convex) {
                std::cout << " (concave)";
            }
            std::cout << ", ";
        }
        std::cout << std::endl;
    }

#ifdef USE_CURVES
    {
        auto start_t = clock();
        std::cout << "detecting shapes... ";
        int numSuccess = features.detect_curves(voxel_width, settings_.min_knot_angle, settings_.max_knots, bezier_cost, line_cost,
                                                settings_.curve_weight);
        if (numSuccess < features.planes.size()) {
            std::cout << "Warning: " << numSuccess << '/' << features.planes.size() << " shapes found"  << std::endl;
            //return 1;
        }
        auto total_t = clock() - start_t;
        float time_sec = static_cast<float>(total_t) / CLOCKS_PER_SEC;
        std::cout << "finished in " << time_sec << " seconds" << std::endl;
    }
#endif
    {
        features.setCurrentShape(contour_name);
        features.detect_parallel(voxel_width, settings_.norm_parallel_threshold);
        std::cout << "parallel:" << std::endl;
        for (size_t c=0; c<features.parallel.size(); c++) {
            if (!features.parallel[c].empty()) {
                std::cout << "cluster " << c << ": ";
                for (const auto &pair : features.parallel[c]) {
                    std::cout << pair.first << " at " << pair.second << ", ";
                }
                std::cout << std::endl;
            }
        }
    }

    features.setCurrentShape(contour_name);
    std::cout << "finding thicknesses..." << std::endl;
    //TODO: proper margin for thickness snapping
    features.compute_depths(thickness_spacing, settings_.thickness_resolution, voxel_width * 2, settings_.edge_detection_threshold, std::log(settings_.thickness_spatial_discount_factor) / diameter_);
    std::cout << "otherOpposing: " << std::endl;
    for (size_t c=0; c<features.opposing_planes.size(); c++) {
        if (!features.opposing_planes[c].empty()) {
            std::cout << "cluster " << c << ": ";
            for (size_t index : features.opposing_planes[c]) {
                std::cout << index << ", ";
            }
            std::cout << std::endl;
        }
    }

    //TODO: just make sure there is always at least one depth and get rid of this
    std::cout << "filling in missing depths" << std::endl;
    bool filtered = features.filter_depths(settings_.thickness_clusters, diameter_ * 1e-6, false, voxel_width * 2);
    if (!filtered) std::cout << "failed!" << std::endl;
    MatrixX3d colors(features.planes.size(), 3);
    for (size_t i = 0; i < features.planes.size(); i++) {
        colors.row(i) = colorAtIndex(i, features.planes.size());
    }
    if (settings_.visualize) {
//        std::cout << "visualizing intersection edges" << std::endl;
//        features.setCurrentShape(CONTOUR_DENSITY_ID);
//        {
//            igl::opengl::glfw::Viewer viewer;
//            viewer.data().show_overlay = true;
//            viewer.data().label_color.head(3) = Vector3f(1, 1, 1);
//            viewer.core().background_color = {0, 0, 0, 1};
//            viewer.core().align_camera_center(cloud_->P);
//            for (size_t c = 0; c < features.planes.size(); c++) {
//                for (const auto &pair : features.adjacency[c]) {
//                    for (size_t i = 0; i < pair.second.intersectionEdges.size(); ++i) {
//                        Edge3d edge3d = pair.second.intersectionEdges.getEdge(i);
//                        viewer.data().add_edges(edge3d.first.transpose(),
//                                                edge3d.second.transpose(), RowVector3d(1, 0, 0));
//                    }
//                }
//            }
//            viewer.launch();
//        }
        {
            igl::opengl::glfw::Viewer viewer;
            viewer.data().show_overlay = true;
            viewer.data().label_color.head(3) = Vector3f(1, 1, 1);
            viewer.core().background_color = {0, 0, 0, 1};

            int nPts = cloud_->P.rows() / settings_.visualization_stride;
            MatrixX3d colors(nPts, 3);
            MatrixX3d visPoints(nPts, 3);
            for (int i=0; i<nPts; ++i) {
                int p = i * settings_.visualization_stride;
                int c = features.point_labels[p];
                colors.row(i) = c >= 0 ? colorAtIndex(c, features.planes.size()) : RowVector3d(1, 1, 1);
//            colors.row(i) = cloud_->N.row(i).array()*0.5+0.5;
                visPoints.row(i) = cloud_->P.row(p);
            }
            viewer.data().add_points(visPoints, colors);
            viewer.data().point_size = 2;

            viewer.core().align_camera_center(cloud_->P);
            std::cout << "visualizing contours" << std::endl;
            for (size_t c = 0; c < features.planes.size(); ++c) {
                if (features.depths[c] < 0) {
                    std::cout << "error: empty depth for " << c << "!" << std::endl;
                    continue;
                }
                {
                    MatrixX3d contour_eig = features.planes[c].points3D(features.planes[c].getShape(BBOX_ID).points());
                    MatrixX3d contour_eig2 = cyclicPermuteRows(contour_eig, 1);
                    viewer.data().add_edges(contour_eig, contour_eig2, RowVector3d(0, 0, 0));
                }

                if (features.planes[c].hasShape()) {
                    MatrixX3d contour_eig = features.planes[c].points3D(
                            features.planes[c].getShape(features.planes[c].getCurrentShape()).points());
                    MatrixX3d contour_eig2 = cyclicPermuteRows(contour_eig, 1);
                    viewer.data().add_edges(contour_eig, contour_eig2, colors.row(c));
                    contour_eig.rowwise() -= features.planes[c].basis().row(2) * features.depths[c];
                    contour_eig2.rowwise() -= features.planes[c].basis().row(2) * features.depths[c];
                    viewer.data().add_edges(contour_eig, contour_eig2, colors.row(c));
                    if (!features.planes[c].getShape(features.planes[c].getCurrentShape()).children().empty()) {
                        for (const auto &child : features.planes[c].getShape(contour_name).children()) {
                            MatrixX3d contour_eig = features.planes[c].points3D(child->points());
                            MatrixX3d contour_eig2 = cyclicPermuteRows(contour_eig, 1);
                            viewer.data().add_edges(contour_eig, contour_eig2, colors.row(c));
                            contour_eig.rowwise() -= features.planes[c].basis().row(2) * features.depths[c];
                            contour_eig2.rowwise() -= features.planes[c].basis().row(2) * features.depths[c];
                            viewer.data().add_edges(contour_eig, contour_eig2, colors.row(c));
                        }
                    }

                    RowVector3d centroid = 0.5 * (contour_eig.colwise().maxCoeff() + contour_eig.colwise().minCoeff());
                    viewer.data().add_label(centroid, "PART " + std::to_string(c));
                    viewer.data().add_points(centroid, colors.row(c));
                } else {
                    std::cout << "no shape for part " << c << "!" << std::endl;
                }
            }

            igl::opengl::glfw::imgui::ImGuiMenu menu;
            menu.callback_draw_viewer_window = []() {};
            viewer.plugins.push_back(&menu);
            viewer.launch();
        }
    }

    /*{
        //debug: visualize winding number isosurface
        double gridSpacing = bbox.maxCoeff() / settings_.voxel_resolution;
        RowVector3i gridRes = (bbox.array() / gridSpacing).ceil().cast<int>();
        size_t numGridPoints = gridRes.prod();
        MatrixX3d gridPoints(numGridPoints, 3);
        for (size_t i = 0; i < numGridPoints; ++i) {
            gridPoints(i, 0) = static_cast<double>(i % gridRes.x()) / gridRes.x() * bbox.x() + bboxMin.x();
            gridPoints(i, 1) =
                    static_cast<double>((i / gridRes.x()) % gridRes.y()) / gridRes.y() * bbox.y() + bboxMin.y();
            gridPoints(i, 2) =
                    static_cast<double>((i / gridRes.x()) / gridRes.y()) / gridRes.z() * bbox.z() + bboxMin.z();
        }
        VectorXd windingNumber = features.query_winding_number(gridPoints);
        MatrixX3d V(0, 3);
        MatrixX3i F(0, 3);
        igl::copyleft::marching_cubes(windingNumber, gridPoints, gridRes.x(), gridRes.y(), gridRes.z(),
                                      settings_.contour_threshold, V, F);
        igl::opengl::glfw::Viewer viewer;
        viewer.data().set_mesh(V, F.rowwise().reverse());
        viewer.launch();
        //end debug
    }
    {
        std::vector<MatrixX3d> vis_contours(features.planes.size());
        for (size_t c=0; c<features.planes.size(); ++c) {
            auto planeBbox = dynamic_cast<Bbox&>(features.planes[c].getShape(0));
            Vector2d bboxDims2d = planeBbox.bb() - planeBbox.aa();
            Vector2i bboxRes2d = (bboxDims2d.array() / voxel_width).ceil().cast<int>();
            std::vector<double> grid(bboxRes2d.prod(), 0);
            int depthRes = features.depths[c].empty() ? 10 : static_cast<int>(std::ceil(features.depths[c][0]/thickness_spacing));
            MatrixX3d Q(bboxRes2d.prod() * depthRes, 3);
            std::cout << "winding numbers for plane " << c << " (" << Q.rows() << ')' << std::endl;
            for (size_t x=0; x<bboxRes2d.x(); ++x) {
                for (size_t y=0; y<bboxRes2d.y(); ++y) {
                    RowVector2d pos2D = (RowVector2d(x, y) * voxel_width) + planeBbox.aa().transpose();
                    for (size_t d=0; d<depthRes; ++d) {
                        Q.row(x + y * bboxRes2d.x() + d * bboxRes2d.prod()) = features.planes[c].points3D(pos2D) - features.planes[c].basis().row(2) * (d * thickness_spacing);
                    }
                }
            }
            VectorXd WN = features.query_winding_number(Q);
            std::cout << "max wn: " << WN.maxCoeff() << std::endl;
            double maxAvgWn = 0.0;
            for (size_t x=0; x<bboxRes2d.x(); ++x) {
                for (size_t y=0; y<bboxRes2d.y(); ++y) {
                    int weight = 0;
                    auto &val = grid[x + y * bboxRes2d.x()];
                    for (size_t d=0; d<depthRes; ++d) {
                        int newweight = weight + 1;
                        val = (weight * val + WN(x + y * bboxRes2d.x() + d * bboxRes2d.prod()))/newweight;
                        weight = newweight;
                    }
                    maxAvgWn = std::max(maxAvgWn, val);;
                }
            }
            std::cout << "max average wn: " << maxAvgWn << std::endl;
            VoxelGrid2D voxelGrid(grid, bboxRes2d.x(), bboxRes2d.y(), planeBbox.aa().x(), planeBbox.aa().y(), voxel_width);
            std::vector<std::vector<int>> hierarchy;
            std::vector<std::vector<Vector2d>> marching_squares_contours = voxelGrid.marching_squares(hierarchy, settings_.contour_threshold);
            std::cout << "found " << marching_squares_contours.size() << " marching squares contours for plane " << c << std::endl;
            std::cout << std::endl << hierarchy.back().size() << " outer contours: " << hierarchy.back().size();
            if (!hierarchy.back().empty()) {
                int max_outer_contour_index = *std::max_element(hierarchy.back().begin(), hierarchy.back().end(),
                                                                [&](int a, int b) {
                                                                    return marching_squares_contours[a].size() <
                                                                           marching_squares_contours[b].size();
                                                                });
                std::cout << "max outer contour size: " << marching_squares_contours[max_outer_contour_index].size() << std::endl;
                vis_contours[c].resize(marching_squares_contours[max_outer_contour_index].size(), 3);
                for (size_t p=0; p<marching_squares_contours[max_outer_contour_index].size(); ++p) {
                    vis_contours[c].row(p) = features.planes[c].points3D(marching_squares_contours[max_outer_contour_index][p].transpose());
                }
            } else {
                std::cout << "warning: no contours found for plane " << c << std::endl;
            }
        }

        igl::opengl::glfw::Viewer viewer;
        for (size_t c=0; c<features.planes.size(); ++c) {
            if (vis_contours[c].rows() > 0) {
                MatrixX3d contour_eig2(vis_contours[c].rows(), 3);
                contour_eig2.block(1, 0, vis_contours[c].rows() - 1, 3) = vis_contours[c].block(0, 0,
                                                                                                vis_contours[c].rows() -
                                                                                                1, 3);
                contour_eig2.row(0) = vis_contours[c].row(vis_contours[c].rows() - 1);
                viewer.data().add_edges(vis_contours[c], contour_eig2, colors.row(c));
            }
        }
        viewer.launch();
    }*/

    /*std::cout << "greedy part generation" << std::endl;
    std::vector<std::vector<int>> newClusters = features.clusters;
    std::vector<size_t> clusterIds(features.planes.size());
    std::iota(clusterIds.begin(), clusterIds.end(), 0);
    std::sort(clusterIds.begin(), clusterIds.end(), [&](size_t a, size_t b) {return newClusters[a].size() > newClusters[b].size();});

    std::vector<std::pair<size_t, CombinedCurve>> newCurves;
    for (size_t k = 0; k < clusterIds.size(); k++) {
        std::sort(clusterIds.begin() + k, clusterIds.end(), [&](size_t a, size_t b) {return newClusters[a].size() > newClusters[b].size();});
        size_t c = clusterIds[k];
        BoundedPlane &extruded_primitive = features.planes[c];
        if (newClusters[c].size() < features.clusters[c].size()/5 || features.depths[c].empty()) continue;
        //recompute shape
        MatrixX2d cloud2d_eig(newClusters[c].size(), 2);
        for (int i=0; i<newClusters[c].size(); i++) {
            int index = newClusters[c][i];
            cloud2d_eig.row(i) = features.planes[c].project(cloud_->row(index).head(3));
        }
        VoxelGrid2D voxels(cloud2d_eig, voxel_width, 1000);
        std::vector<std::vector<int>> hierarchy;
        std::vector<std::vector<Vector2d>> marching_squares_contours = voxels.marching_squares(hierarchy, 0.5);
        if (!hierarchy.back().empty()) {
            std::cout << "part " << c << ": ";
            int max_outer_contour_index = *std::max_element(hierarchy.back().begin(), hierarchy.back().end(),
                                                            [&](int a, int b) {
                                                                return marching_squares_contours[a].size() <
                                                                       marching_squares_contours[b].size();
                                                            });
            const std::vector<Vector2d> &max_outer_contour = marching_squares_contours[max_outer_contour_index];
            MatrixX2d contour_eig(max_outer_contour.size(), 2);
            for (int p = 0; p < max_outer_contour.size(); p++) {
                contour_eig.row(p) = max_outer_contour[p].transpose();
            }
            CombinedCurve outerCurve;
            double L2 = outerCurve.fit(contour_eig, settings_.min_knot_angle, settings_.max_knots, bezier_cost, line_cost, settings_.curve_weight);
            std::cout << "found curve with " << outerCurve.size() << " segments for plane " << c << std::endl;
            newCurves.emplace_back(c, outerCurve);
        } else {
            continue;
        }
        for (auto &pair : features.parallel[c]) {
            if (features.planes[pair.first].basis().row(2).dot(extruded_primitive.basis().row(2)) < 0) {
                newClusters[pair.first].clear();
            }
        }
        //update neighbors
        for (auto &pair : features.adjacency[c]) {
            size_t n = pair.first;
            if (n < features.planes.size()) {
                BoundedPlane &neighbor = features.planes[n];
                Vector2d pt_a = neighbor.project(pair.second.first.transpose()).transpose();
                size_t N = newClusters[n].size();
                MatrixX3d neighbor_points(N, 3);

                Vector2d dir = -neighbor.project(extruded_primitive.basis().row(2)).transpose().normalized();
                double z_start = dir.dot(pt_a);
                for (size_t p = 0; p < N; p++) {
                    neighbor_points.row(
                            p) = cloud_->row(newClusters[n][p]).head(3);
                }
                MatrixX2d neighbor_points_2D = neighbor.project(neighbor_points);
                neighbor_points.resize(0, 3);
                std::vector<int> newCluster;
                for (size_t p = 0; p < N; p++) {
                    if ((neighbor_points_2D.row(p).dot(dir) - z_start) > features.depths[c][0]) {
                        newCluster.push_back(newClusters[n][p]);
                    }
                }
                newClusters[n] = newCluster;
            }
        }
    }

    std::cout << "found " << newCurves.size() << " parts" << std::endl;

    size_t shapeIdx=0;
    for (vp = vertices(construction.g); vp.first != vp.second; ++vp.first, ++shapeIdx) {
        Construction::Vertex v = *vp.first;
        PartData &pd = construction.partData[construction.g[v].partIdx];
        pd.shapeIdx = shapeIdx;
        //TODO: calculate pose and establish convention
        const auto &plane = features.planes[newCurves[shapeIdx].first];
        //std::cout << "basis " << shapeIdx << ": \n" << plane.basis().transpose() << std::endl;
        pd.rot = plane.basis();
        pd.rot = pd.rot.conjugate();
        //std::cout << "basis as quaternion: \n" << pd.rot.matrix() << std::endl;
        pd.pos = -plane.basis().row(2) * plane.offset();
        ShapeData data;
        data.cutPath = newCurves[shapeIdx].second;
        data.stockIdx = shapeIdx;
        construction.shapeData.push_back(data);
        StockData stock;
        stock.thickness = features.depths[newCurves[shapeIdx].first][0];
        construction.stockData.push_back(stock);
    }*/

    const double intersectionEdgeMargin = voxel_width * 3;
    //update construction
    construction.partData.reserve(features.planes.size());
    construction.shapeData.reserve(features.planes.size());
    construction.stockData.reserve(features.planes.size());
    std::cout << "candidate planes: " << features.planes.size() << std::endl;
    int N = features.planes.size();
    for (size_t c=0; c<N; ++c) {
        std::cout << "converting plane " << c << " to part" << std::endl;
        const auto &plane = features.planes[c];
        construction.partData.emplace_back();
        PartData &pd = construction.partData.back();
        pd.shapeIdx = c;
        pd.rot = Quaterniond(plane.basis()).conjugate();
        pd.pos = -plane.basis().row(2) * plane.offset();
        pd.opposingPartIds = features.opposing_planes[c];
        pd.bothSidesObserved = !pd.opposingPartIds.empty();
        pd.pointIndices = std::move(features.clusters[c]);
        construction.shapeData.emplace_back();
        ShapeData &sd = construction.shapeData.back();
#ifdef USE_CURVES
        sd.cutPath = plane.getShape(3).clone();
#else
        sd.cutPath = plane.getShape(contour_name).clone();
#endif
//        sd.pointDensityContour = plane.getShape(CONTOUR_DENSITY_ID).clone();
        sd.gridSpacing = voxel_width;
//        sd.bbox = plane.getShape(BBOX_ID).clone();
        //add 3D adjacency edges as (oriented) 2D constraint lines
        for (const auto &pair : features.adjacency[c]) {
            for (size_t i=0; i<pair.second.intersectionEdges.size(); ++i) {
                Edge3d edge3d = pair.second.intersectionEdges.getEdge(i);
                Edge2d edge2d(pd.project(edge3d.first.transpose()).transpose(),
                              pd.project(edge3d.second.transpose()).transpose());
                Vector2d dir(edge2d.second - edge2d.first);
                double len = dir.norm();
                if (len < 2 * intersectionEdgeMargin) continue;
                dir /= len;
                edge2d.second -= dir * intersectionEdgeMargin;
                edge2d.first += dir * intersectionEdgeMargin;
                Vector2d n = Vector2d(dir.y(), -dir.x());
                Vector2d planeNormal = pd.projectDir(
                        features.planes[pair.first].normal()); //outside direction
                if (n.dot(planeNormal) < 0) {
                    std::swap(edge2d.first, edge2d.second);
                }
                if (!pair.second.convex) {
                    std::swap(edge2d.first, edge2d.second);
                }
                ShapeConstraint sc;
                sc.edge = std::move(edge2d);
                sc.convex = pair.second.convex;
                sc.inside = true;
                sc.outside = sc.convex;
                sd.shapeConstraints[pair.first].push_back(std::move(sc));
            }
        }

        /*{
            //re-center
            Vector2d centroid = sd.bbox->points().colwise().mean().transpose();
            Vector3d shift = pd.rot * Vector3d(centroid.x(), centroid.y(), 0);
            pd.pos += shift;
            sd.bbox->translate(-centroid);
            sd.cutPath->translate(-centroid);
            sd.pointDensityContour->translate(-centroid);
        }*/
        sd.stockIdx = c;
        construction.stockData.emplace_back();
        StockData &stock = construction.stockData.back();
        stock.thickness = features.depths[c];
    }

    //add ground plane
    {
        PartData pd;
        pd.shapeIdx = N;
        pd.groundPlane = true;
        pd.rot = Eigen::Quaterniond::Identity();
        pd.pos = Eigen::Vector3d(0, 0, bboxMin.z());
        construction.partData.push_back(std::move(pd));

        ShapeData sd;
        std::shared_ptr<Primitive> groundShape(new Bbox((bboxMin.head(2) - bbox.head(2) * 0.5).transpose(), (bboxMax.head(2) + bbox.head(2) * 0.5).transpose()));
        sd.cutPath = std::move(groundShape);
        sd.gridSpacing = voxel_width;
        sd.stockIdx = N;
        construction.shapeData.push_back(std::move(sd));

        StockData stock;
        stock.thickness = voxel_width * 10;
        construction.stockData.push_back(std::move(stock));
    }

    construction.setW(std::vector<bool>(construction.partData.size(), true));
    construction.computeMeshes();
}