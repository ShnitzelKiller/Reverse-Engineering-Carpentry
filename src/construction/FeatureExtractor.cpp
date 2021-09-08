//
// Created by James Noeckel on 3/16/20.
//

#include "geometry/cgal/geom.h"
#include "geometry/primitives3/BoundedPlane.h"
#include "geometry/primitives3/Cylinder.h"
#include "geometry/shapes2/VoxelGrid.hpp"
#include "utils/top_k_indices.hpp"
#include "math/integration.h"
#include "math/robust_derivative.hpp"
#include "math/nonmax_suppression.h"
#include "math/GaussianMixture.h"
#include "FeatureExtractor.h"
#include <numeric>
#include <unordered_set>
#include <random>
#include "geometry/fast_winding_number.h"
#include "geometry/shapes2/Curve.h"
#include <igl/octree.h>
#include "math/fields/PointDensityField.h"
#include "math/fields/WindingNumberField3D.h"
#include "math/fields/FieldSlice.h"
#include "geometry/shapes2/Primitive.h"
#include "geometry/shapes2/polygon_thickness.h"
#include "geometry/shapes2/convert_contour.h"
#include <chrono>
#include "utils/printvec.h"

#define OFFSET_STEPS 0

using namespace Eigen;

FeatureExtractor::FeatureExtractor(PointCloud3::Handle cloud, ReconstructionData::Handle reconstruction, std::mt19937 &random) : cloud_(std::move(cloud)), reconstruction_(std::move(reconstruction)), random_(random) {
    if (cloud_->P.rows() > 0) {
        compute_bounds();
    }
}

FeatureExtractor::FeatureExtractor(ReconstructionData::Handle reconstruction, std::mt19937 &random) : reconstruction_(std::move(reconstruction)), random_(random) {
}

void FeatureExtractor::compute_bounds() {
    minPt_ = cloud_->P.colwise().minCoeff();
    maxPt_ = cloud_->P.colwise().maxCoeff();
}

void FeatureExtractor::compute_point_labels() {
    point_labels.clear();
    point_labels.resize(cloud_->P.rows(), -1);
    for (int i=0; i<clusters.size(); i++) {
        for (auto ind : clusters[i]) {
            point_labels[ind] = i;
        }
    }
}

size_t FeatureExtractor::recompute_normals() {
    size_t num_flipped = 0;
    for (size_t c=0; c<planes.size(); c++) {
        double sum_dot_product = 0.0;
        for (int ind : clusters[c]) {
            sum_dot_product += planes[c].basis().row(2).dot(cloud_->N.row(ind));
        }
        if (sum_dot_product < 0) {
            planes[c].flip();
            num_flipped++;
        }
    }
    return num_flipped;
}

size_t FeatureExtractor::split_clusters(int min_support) {
    size_t num_extra = 0;
    size_t N = planes.size();
    for (size_t c=0; c<N; c++) {
        std::vector<int> positive;
        std::vector<int> negative;
        for (int ind : clusters[c]) {
            if (planes[c].basis().row(2).dot(cloud_->N.row(ind)) >= 0) {
                positive.push_back(ind);
            } else {
                negative.push_back(ind);
            }
        }
        std::cout << c << ": positive: " << positive.size() << ", negative: " << negative.size() << std::endl;
        if (negative.size() > min_support) {
            clusters.insert(clusters.begin() + N + num_extra, std::move(negative));
            planes.push_back(planes[c]);
            planes.back().flip();
            num_extra++;
            std::cout << "split cluster " << c << std::endl;
        }
        clusters[c] = std::move(positive);
    }
    return num_extra;
}

void FeatureExtractor::compute_winding_number(int winding_number_stride, int k_n) {
    precompute_fast_winding_number(*cloud_, k_n, *windingNumberData_, winding_number_stride);
    windingNumberDirty_ = false;
}

VectorXd FeatureExtractor::query_winding_number(const Ref<const MatrixX3d> &Q) const {
    if (windingNumberDirty_) {
        std::cerr << "no winding number data" << std::endl;
        return VectorXd(0);
    }
    return fast_winding_number(*windingNumberData_, Q);
}

void FeatureExtractor::detect_primitives(double threshold, double support, double probability, bool use_cylinders,
                                         double cluster_epsilon, double normal_threshold) {
    if (!cloud_) {
        std::cerr << "no point cloud set, aborting" << std::endl;
    }
    //segmentStock(cloud, normals, threshold, 10, k_r, clusters);
    std::vector<PlaneParam> plane_params;
    std::vector<CylinderParam> cylinder_params;
    auto min_points = static_cast<size_t>(support * cloud_->P.rows());
    efficient_ransac(cloud_, plane_params, cylinder_params, clusters, threshold, min_points, probability, use_cylinders, cluster_epsilon, normal_threshold);

    for (auto & plane_param : plane_params) {
        planes.emplace_back(plane_param.first, plane_param.second);
    }

    //filter out points based on depth
    /*for (size_t c=0; c<planes.size(); c++) {
        clusters[c].erase(std::remove_if(clusters[c].begin(), clusters[c].end(), [&](int ind) {
            double dist = planes[c].basis().row(2) * (*cloud_)[ind].getVector3fMap().cast<double>() + planes[c].offset();
            return std::fabs(dist) > threshold;
        }), clusters[c].end());
    }*/


    for (size_t c=0; c<cylinder_params.size(); c++) {
        double start = std::numeric_limits<double>::max();
        double end = std::numeric_limits<double>::lowest();
        for (int index : clusters[plane_params.size() + c]) {
            Vector3d point3d = cloud_->P.row(index).transpose();
            double projection = cylinder_params[c].first.second.dot(point3d - cylinder_params[c].first.first);
            start = std::min(start, projection);
            end = std::max(end, projection);
        }
        cylinders.emplace_back(cylinder_params[c].first.first, cylinder_params[c].first.second, cylinder_params[c].second, start, end);
    }

    size_t num_flipped = recompute_normals();
    size_t num_split = split_clusters(min_points);
    std::cout << "flipped " << num_flipped << " normals and split " << num_split << " clusters" << std::endl;
    compute_point_labels();
}

void FeatureExtractor::detect_bboxes() {
    size_t N = planes.size();
//#pragma omp parallel for default(none) shared(N, planes)
    for (size_t c=0; c<N; ++c) {
        MatrixX2d cloud2d(clusters[c].size(), 2);
        for (int i=0; i<clusters[c].size(); i++) {
            int index = clusters[c][i];
            cloud2d.row(i) = planes[c].project(cloud_->P.row(index));
        }

        //add bounding boxes
        Matrix<double, 4, 2> bbox;
        min_bbox(cloud2d, bbox);

        //realign basis to min bounding box axes for efficient representation
        RowVector2d e01 = bbox.row(1) - bbox.row(0);
        double l01 = e01.norm();
        RowVector2d e03 = bbox.row(3) - bbox.row(0);
        double l03 = e03.norm();
        Matrix<double, 2, 2> rot;
        rot << e01/l01, e03/l03;
        Vector2d aa = rot * bbox.row(0).transpose();
        Vector2d bb = rot * bbox.row(2).transpose();
        planes[c].changeBasis(rot);
        planes[c].addShape(BBOX_ID, std::make_shared<Bbox>(aa, bb));
        //add convex hulls
        /*MatrixX2d hull;
        convex_hull(cloud2d, hull);
        shapeAdd.lock();
        planes[c].addShape(CONVEX_HULL_ID, std::make_shared<Polygon>(std::move(hull)));
        shapeAdd.unlock();*/
    }
}

#pragma omp declare reduction(ListJoin: std::vector<int>: \
	omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
//	initializer (omp_priv=Custom(100))

void FeatureExtractor::detect_contours(double voxel_width, double threshold, double min_contour_hole_ratio, bool use_winding_number, int erosion, bool debug_vis) {
    int name = use_winding_number ? CONTOUR_WN_ID : CONTOUR_DENSITY_ID;
    size_t N = planes.size();
    std::vector<int> invalidShapes;
    auto t_start = clock();
    auto t_start_w = std::chrono::high_resolution_clock::now();
//#pragma omp parallel for default(none) shared(N, voxel_width, threshold, use_winding_number, name, planes, vis_shapes_raw, std::cout, debug_vis, erosion, min_contour_hole_ratio) reduction(ListJoin:invalidShapes)
    for (size_t c=0; c<N; ++c) {
        //std::cout << "shape " << c << ": " << std::endl;
        PointCloud2::Handle cloud2d(new PointCloud2);
        cloud2d->P.resize(clusters[c].size(), 2);
        for (int i=0; i<clusters[c].size(); i++) {
            int index = clusters[c][i];
            cloud2d->P.row(i) = planes[c].project(cloud_->P.row(index));
        }
        //more general contours via marching squares
        double stdev = voxel_width/4.0;
        RowVector2d minPt = cloud2d->P.colwise().minCoeff();
        RowVector2d maxPt = cloud2d->P.colwise().maxCoeff();
        RowVector2d min2d = minPt.array() - (3*stdev);
        RowVector2d max2d = maxPt.array() + (3*stdev);
        if (min2d.isApprox(max2d, 0)) continue;
        ScalarField<2>::Handle field;
        if (use_winding_number) {
            //winding number field
            ScalarField<3>::Handle wnField(
                    new WindingNumberField3D(windingNumberData_));
            field = ScalarField<2>::Handle(new FieldSlice(std::move(wnField), Quaterniond(planes[c].basis()).conjugate(), planes[c].offset(), voxel_width * 3.0, 5));
        } else {
            //point density field
            field = ScalarField<2>::Handle(
                    new PointDensityField(std::move(cloud2d), stdev));
        }
        VoxelGrid2D voxels(std::move(field), min2d.x(), min2d.y(), max2d.x(), max2d.y(), voxel_width, 1000);
        std::vector<std::vector<int>> hierarchy;
        std::vector<std::vector<Vector2d>> marching_squares_contours = voxels.marching_squares(hierarchy, threshold);
        /*std::cout << "resolution: " << voxels.resolution().transpose() << std::endl;
        std::cout << "found " << marching_squares_contours.size() << " contours" << std::endl;
        std::cout << "sizes: ";
        size_t totaledges = 0;
        for (int i=0; i<marching_squares_contours.size() ;i++) {
            const auto& cont = marching_squares_contours[i];
            totaledges += cont.size();
            std::cout << i << ":" << cont.size() << ", ";
        }
        std::cout << std::endl << hierarchy.back().size() << " outer contours: ";
        for (auto ind : hierarchy.back()) {
            std::cout << ind << ", ";
        }
        std::cout << std::endl << "total edges in all contours: " << totaledges << std::endl;*/
        std::shared_ptr<Primitive> shape = convertContour(marching_squares_contours, hierarchy, min_contour_hole_ratio);
        if (shape) {
            planes[c].addShape(name, std::move(shape));
            if (erosion > 0) {
                // also add eroded version of this shape
                voxels.discretize(threshold);
                //voxels.dilate(0.5, false);
                //voxels.dilate(0.5, true);
                voxels.dilate(0.5, true, MatrixXi::Ones(erosion*2+1, erosion*2+1), erosion, erosion);
                marching_squares_contours = voxels.marching_squares(hierarchy, 0.5);
                std::shared_ptr<Primitive> erodedShape = convertContour(marching_squares_contours, hierarchy, 5);
                if (erodedShape) planes[c].addShape(CONTOUR_ERODED_ID, std::move(erodedShape));
            }
        } else {
#pragma omp critical(printEmpty)
            std::cout << "warning: empty contour for part " << c << std::endl;
            invalidShapes.push_back(c);
        }
    }
    auto total_t = clock() - t_start;
    auto total_t_w = std::chrono::high_resolution_clock::now() - t_start_w;
    float time_sec = static_cast<float>(total_t) / CLOCKS_PER_SEC;
    if (!invalidShapes.empty()) {
        std::cout << invalidShapes.size() << " invalid contours: " << invalidShapes << std::endl;
    }
    std::cout << "contour detection finished in " << time_sec << " CPU seconds (" << std::chrono::duration<double>(total_t_w).count() << "s wall clock time" << std::endl;
}

//int FeatureExtractor::detect_curves(double voxel_width, double knot_curvature, int max_knots, double bezier_cost, double line_cost, double curve_weight) {
//    //PROJECT 3D POINTS ONTO PLANES AND COMPUTE CANDIDATE SHAPES
//    vis_shapes.resize(planes.size());
//    int numSuccess = 0;
//    double max_edge_length = 0.0;
//    double edge_length_2d = 0.0;
//    for (size_t c=0; c<planes.size(); c++) {
//        int name = -1;
//        if (planes[c].hasShape(CONTOUR_WN_ID)) name = CONTOUR_WN_ID;
//        else if (planes[c].hasShape(CONTOUR_DENSITY_ID)) name = CONTOUR_DENSITY_ID;
//        if (planes[c].hasShape(name)) {
//            auto &marchingSquares = planes[c].getShape(name);
//            std::cout << "part " << c << ": ";
//            MatrixX2d contour_eig = marchingSquares.points();
//            //try curve fitting
//            CombinedCurve outerCurve;
//            //search for intersection edges to split up the exterior curve
//            if (!adjacency[c].empty()) {
//                const auto& edges = adjacency[c];
//                int totalConstraints = 0;
//                std::vector<MatrixX2d> projectedNeighbors;
//                std::vector<Matrix2d> edges_mat;
//                edges_mat.reserve(edges.size());
//                for (const auto &edge : edges) {
//                    if (edge.first >= planes.size()) {
//                        MatrixX2d projectedNeighbor(clusters[edge.first].size(), 2);
//                        int offset = 0;
//                        for (int k : clusters[edge.first]) {
//                            Vector3d pt = cloud_->P.row(k).transpose();
//                            double dist = std::fabs(planes[c].basis().row(2) * pt + planes[c].offset());
//                            if (dist < voxel_width * 2) {
//                                projectedNeighbor.row(offset) = planes[c].project(pt.transpose());
//                                offset++;
//                            }
//                        }
//                        projectedNeighbors.emplace_back(offset, 2);
//                        projectedNeighbors.back() = projectedNeighbor.block(0, 0, offset, 2);
//                    } else {
//                        for (size_t i=0; i<edge.second.intersectionEdges.size(); ++i) {
//                            Matrix2d mat;
//                            mat << planes[c].project(edge.second.intersectionEdges.getEdge(i).first.transpose()), planes[c].project(
//                                    edge.second.intersectionEdges.getEdge(i).second.transpose());
//                            RowVector2d ab = mat.row(1) - mat.row(0);
//                            if (ab.norm() > 2 * voxel_width) {
//                                mat.row(0) += ab * voxel_width;
//                                mat.row(1) -= ab * voxel_width;
//                                edges_mat.push_back(mat);
//                            }
//                        }
//                    }
//                }
//                auto start_t = clock();
//                double L2 = outerCurve.fit(contour_eig, knot_curvature, max_knots, bezier_cost, line_cost, curve_weight, 0, -1, 5, Vector2d(0, 0), Vector2d(0, 0), edges_mat, projectedNeighbors,
//                                           {voxel_width * 1.5});
//                auto total_t = clock() - start_t;
//                float time_sec = static_cast<float>(total_t) / CLOCKS_PER_SEC;
//                std::cout << "found "<< outerCurve.size() <<" curves with " << edges_mat.size() << " line constraints, " << projectedNeighbors.size() << " curve constraints, and " << L2 << " total error" << "in " << time_sec << " seconds" << std::endl;
//            } else {
//                auto start_t = clock();
//                double L2 = outerCurve.fit(contour_eig, knot_curvature, max_knots, bezier_cost, line_cost, curve_weight);
//                auto total_t = clock() - start_t;
//                float time_sec = static_cast<float>(total_t) / CLOCKS_PER_SEC;
//                std::cout << "found "<< outerCurve.size() <<" curves with " << L2 << " total error" << "in " << time_sec << " seconds" << std::endl;
//            }
//            //visualize
//            if (outerCurve.size() > 0) {
//                MatrixX2d outer_sampled_curve = outerCurve.uniformSample(30);
//                vis_shapes[c].emplace_back();
//                for (int p = 0; p < outer_sampled_curve.rows(); p++) {
//                    vis_shapes[c].back().push_back(planes[c].points3D(outer_sampled_curve.row(p)).transpose());
//                }
//                for (int p = 0; p < contour_eig.rows(); p++) {
//                    double edge_length_3d = (vis_shapes[c].back()[p] -
//                                             vis_shapes[c].back()[(p + 1) % contour_eig.rows()]).norm();
//                    if (edge_length_3d > max_edge_length) {
//                        max_edge_length = std::max(max_edge_length, edge_length_3d);
//                        edge_length_2d = (contour_eig.row(p) - contour_eig.row((p + 1) % contour_eig.rows())).norm();
//                    }
//                }
//                //
//
//                std::vector<std::shared_ptr<Primitive>> holes;
//                for (auto &child : marchingSquares.children()) {
//                    MatrixX2d hole_eig = child->points();
//                    CombinedCurve holeCurve;
//                    holeCurve.fit(hole_eig, knot_curvature, max_knots, bezier_cost, line_cost, curve_weight);
//                    /*MatrixX2d inner_sampled_curve = holeCurve.uniformSample(10);
//                    //visualize holes
//                    vis_shapes[c].emplace_back();
//                    for (size_t p = 0; p < inner_sampled_curve.rows(); p++) {
//                        vis_shapes[c].back().push_back(planes[c].points3D(inner_sampled_curve.row(p)).transpose());
//                    }*/
//                    //
//                    holes.emplace_back(new PolyCurveWithHoles(std::move(holeCurve)));
//                }
//                if (holes.empty()) {
//                    planes[c].addShape(CONTOUR_CURVES_ID, std::make_shared<PolyCurveWithHoles>(std::move(outerCurve)));
//                } else {
//                    std::cout << "found " << holes.size() << " inner contours";
//                    planes[c].addShape(CONTOUR_CURVES_ID, std::make_shared<PolyCurveWithHoles>(std::move(outerCurve), std::move(holes)));
//                }
//                ++numSuccess;
//            }
//        }
//    }
//    std::cout << "maximum edge length: " << max_edge_length << " with 2D edge " << edge_length_2d << std::endl;
//    return numSuccess;
//}

/*void FeatureExtractor::split_shapes(double image_derivative_threshold, double offset_tolerance) {
    int totalCuts = 0;
    // index of edge, start and end t
    using CutIndex = std::tuple<int, double, double>;
    cut_edges.resize(planes.size());
    for (size_t c=0; c<planes.size(); c++) {
        if (planes[c].getCurrentShape() != CONTOUR_CURVES_ID) continue;
        if (pruned_cluster_visibility[c].empty()) continue;

        //collect cuts
        // base rays which may cover multiple cuts
        std::vector<Ray2d> edges;
        // index into list of rays, with start and end t for each cut
        std::vector<CutIndex> cuts;
        //TODO: use offset planes for more candidates
        for (const auto &pair : adjacency[c]) {
            if (pair.first >= planes.size() || !planes[pair.first].hasShape()) continue; //skip cylinders
            // 3D endpoints of the edge
            Matrix<double, 2, 3> edgeMat3d;
            edgeMat3d << pair.second.intersectionEdge.first.transpose(),
                    pair.second.intersectionEdge.second.transpose();
            // 2D endpoints of the edge
            Matrix<double, 2, 2> edgeMat2d = planes[c].project(edgeMat3d);
            RowVector2d rayDir2d = (edgeMat2d.row(1) - edgeMat2d.row(0)).normalized();
            RowVector3d rayDir3d = (edgeMat3d.row(1) - edgeMat3d.row(0)).normalized();
            // normal of the edge, lies in the plane of this part
            RowVector3d n = planes[pair.first].basis().row(2);
            // number of image views of this part
            int imgCount = pruned_cluster_visibility[c].size();
            // speed in world space to traverse at most 1 pixel per step in each image
            std::vector<double> speeds;
            speeds.reserve(imgCount);
            for (auto image_id : pruned_cluster_visibility[c]) {
                auto &image = reconstruction_->images[image_id];
                Vector2d dadt = reconstruction_->directionalDerivative(edgeMat3d.row(0).transpose(), n.transpose(), image_id);
                Vector2d dbdt = reconstruction_->directionalDerivative(edgeMat3d.row(1).transpose(), n.transpose(), image_id);
                speeds.push_back(1.0/std::max(dadt.norm(), dbdt.norm()));
            }
            // step size in world coordinates to ensure no pixels are missed in the images
            double worldSpeed = *std::min_element(speeds.begin(), speeds.end());
            int pixelDistance = static_cast<int>(std::floor(offset_tolerance / worldSpeed));
            //std::cout << "pixel distance for edge " << pair.first << ": " << pixelDistance << std::endl;
            // total offsets to consider
            int N = pixelDistance * 2 + 1;
            // detect all unique intersection segments on this line via perturbation
            // start and end t values detected for this edge
            std::vector<std::pair<double, double>> intervals;
            {
                bool contained = true;
                double lowerBound = std::numeric_limits<double>::lowest();
                double upperBound = std::numeric_limits<double>::max();
                for (int i = 0; i < N; i++) {
                    double magnitude = static_cast<double>(i - pixelDistance) * worldSpeed;
                    Matrix<double, 2, 3> edgeMat3d_i = edgeMat3d.array().array().rowwise() + n.array() * magnitude;
                    Matrix<double, 2, 2> edgeMat2d_i = planes[c].project(edgeMat3d_i);
                    auto intersections = planes[c].getShape(CONTOUR_DENSITY_ID).intersect(
                            Ray2d(edgeMat2d_i.row(0).transpose(), rayDir2d.transpose()));
                    if (intersections.empty()) {
                        contained = false;
                        break;
                    }
                    //update global bounds
                    upperBound = std::min(upperBound, intersections.back().t);
                    lowerBound = std::max(lowerBound, intersections.front().t);
                    //keep track of the minimum bounds of each interval
                    if (intersections.size() % 2 == 0 && intersections.size() >= intervals.size() * 2) {
                        if (intervals.size() * 2 < intersections.size()) {
                            intervals.clear();
                            intervals.resize(intersections.size() / 2,
                                             std::make_pair(std::numeric_limits<double>::lowest(),
                                                            std::numeric_limits<double>::max()));
                        }
                        for (int j = 0; j < intervals.size(); j++) {
                            //visualize
                            //cut_edges[c].emplace_back(edgeMat3d_i.row(0) + rayDir3d * intersections[j*2] + planes[c].basis().row(2) * offset_tolerance * 0.5, edgeMat3d_i.row(0) + rayDir3d * intersections[j*2+1] + planes[c].basis().row(2) * offset_tolerance * 0.5);
                            intervals[j].first = std::max(lowerBound,
                                                          std::max(intervals[j].first, intersections[j * 2].t));
                            intervals[j].second = std::min(upperBound,
                                                           std::min(intervals[j].second, intersections[j * 2 + 1].t));
                        }
                    }
                }
                if (!contained) continue; //if any of the perturbed lines are outside the shape, discard all cuts
            }
            intervals.erase(std::remove_if(intervals.begin(), intervals.end(), [](auto a) {return a.first >= a.second;}), intervals.end());
            //std::cout << "found " << intervals.size() << " intervals: ";
            edges.emplace_back(edgeMat2d.row(0).transpose(), rayDir2d.transpose());
            //accumulate cuts for all sub-intervals
            int keptIntervals = 0;
            int checkedIntervals = 0;
            for (const auto & interval : intervals) {
                //std::cout << '(' << interval.first << ", " << interval.second << "), ";
                //discard cuts that intersect previous cuts
                //TODO: move this to a later step
//                bool intersected = false;
//                cuts.back().start = interval.first;
//                cuts.back().end = interval.second;
//                for (const auto &cut : cuts) {
//                    cuts[std::get<0>(cut)].start = std::get<1>(cut);
//                    cuts[std::get<0>(cut)].end = std::get<2>(cut);
//                    double t;
//                    if (cuts.back().intersect(cuts[std::get<0>(cut)], t)) {
//                        intersected = true;
//                        break;
//                    }
//                }
//                if (intersected) continue;
                checkedIntervals++;
                // 3D endpoints of this cut
                Matrix<double, 2, 3> subEdge3d;
                subEdge3d.row(0) = edgeMat3d.row(0) + interval.first * rayDir3d;
                subEdge3d.row(1) = edgeMat3d.row(0) + interval.second * rayDir3d;
                // compute image derivatives
                // score of this cut (avg of maximum derivatives per image)
                double score = 0.0;
                // number of votes this cut received
                int imgVotes = 0;
                double weightAcc = 0.0;

                for (int j=0; j<imgCount; j++) {
                    int image_id = pruned_cluster_visibility[c][j];
                    auto &image = reconstruction_->images[image_id];
                    double maxAbsLumDerivative = 0.0;
                    double totalWeight = 0.0;
                    for (int i = 0; i < N; i++) {
                        double magnitude = static_cast<double>(i - pixelDistance) * worldSpeed;
                        // endpoints of the edge in camera space
                        Matrix<double, 2, 2> edgeMatCam = reconstruction_->project(subEdge3d.array().rowwise() + magnitude * n.array(), image_id);
                        // derivatives of pixel coordinates w.r.t. world position at cut endpoints
                        Vector2d dadt = reconstruction_->directionalDerivative(subEdge3d.row(0).transpose(), n.transpose(), image_id).normalized();
                        Vector2d dbdt = reconstruction_->directionalDerivative(subEdge3d.row(1).transpose(), n.transpose(), image_id).normalized();
                        // analytical luminance derivative
                        double lum = integrate_image(image.getDerivativeX(),
                                                   edgeMatCam.row(0).transpose(),
                                                   edgeMatCam.row(1).transpose(),
                                                   true,
                                                   dadt,
                                                   dbdt,
                                                   image.getDerivativeY())(0)/16.0;
                        lum = std::fabs(lum);
                        if (lum > maxAbsLumDerivative) {
                            maxAbsLumDerivative = lum;
                        }
                        totalWeight += (edgeMatCam.row(1) - edgeMatCam.row(0)).norm();
                    }
                    if (maxAbsLumDerivative > image_derivative_threshold) {
                        imgVotes++;
                    }
                    std::cout << "max derivative: " << maxAbsLumDerivative << std::endl;
                    double newweight = totalWeight + weightAcc;
                    score = (score * weightAcc + maxAbsLumDerivative * totalWeight) / newweight;
                    weightAcc = newweight;
                }
                std::cout << "votes for edge " << c << '.' << pair.first << '.' << interval.first << ": " << imgVotes << '/' << imgCount << std::endl;

                //DEBUG VISUALIZATION
//                for (int j=0; j<pruned_cluster_visibility[c].size(); j++) {
//                    int image_id = pruned_cluster_visibility[c][j];
//                    auto &image = reconstruction_->images[image_id];
//                    cv::Mat visImageA = image.getImage(true).clone();
//                    cv::Mat visImageB = image.getImage(true).clone();
//                    for (int i = 0; i < N; i+=4) {
//                        double magnitude = static_cast<double>(i - pixelDistance) * worldSpeed;
//                        Matrix<double, 2, 2> edgeMatCam = reconstruction_->project(
//                                subEdge3d.array().rowwise() + magnitude * n.array(), image_id);
//                        Vector2d dadt = reconstruction_->directionalDerivative(subEdge3d.row(0).transpose(), n.transpose(), image_id) * offset_tolerance;
//                        Vector2d dbdt = reconstruction_->directionalDerivative(subEdge3d.row(1).transpose(), n.transpose(), image_id) * offset_tolerance;
//
//                        cv::Point2d ptA(edgeMatCam(0, 0), edgeMatCam(0, 1));
//                        cv::Point2d ptB(edgeMatCam(1, 0),edgeMatCam(1, 1));
//                        cv::line(visImageA, ptA,
//                                ptB,
//                                static_cast<uchar>(std::min(255.0, std::abs(luminanceDerivatives[i] * 10))),
//                                2);
//                        cv::line(visImageB, ptA,
//                                 ptB,
//                                 static_cast<uchar>(std::min(255.0, std::abs(finiteDifferenceDerivatives[i] * 10))),
//                                 2);
//                        for (auto & im : std::array<cv::Mat*, 2>{&visImageA, &visImageB}) {
//                            cv::Point2d offseta(dadt.x(), dadt.y());
//                            cv::Point2d offsetb(dbdt.x(), dbdt.y());
//                            auto color = static_cast<uchar>((i*255)/N);
//                            cv::line(*im, ptA, ptA + offseta, color, 4);
//                            cv::line(*im, ptB, ptB + offsetb, color, 4);
//                        }
//                    }
//                    cv::imwrite("cut_deriv_" + std::to_string(c) + "_" + std::to_string(image_id) + "_" + std::to_string(checkedIntervals) + "_analytic.png", visImageA);
//                    cv::imwrite("cut_deriv_" + std::to_string(c) + "_" + std::to_string(image_id) + "_" + std::to_string(checkedIntervals) + "_discrete.png", visImageB);
//                }
                //END
                //visualize
                // if there is a peak, record this edge
                if (imgVotes*2 > imgCount) {
                    cut_edges[c].emplace_back(subEdge3d.row(0) + planes[c].basis().row(2) * offset_tolerance, subEdge3d.row(1) + planes[c].basis().row(2) * offset_tolerance);
                    cuts.emplace_back(edges.size() - 1, interval.first, interval.second);
                    keptIntervals++;
                    totalCuts++;
                }
            }
            std::cout << "checked " << checkedIntervals << " intervals, kept " << keptIntervals << std::endl;
        }
        // recompute shapes with all cuts
        std::cout << "found " << cuts.size() << " cuts for part " << c << std::endl;
    }
    std::cout << "total cuts: " << totalCuts << std::endl;
}*/


bool
FeatureExtractor::detect_adjacency(double adjacency_threshold, double norm_adjacency_threshold, int shape1, int shape2,
                                   MultiRay3d &intersectionEdges, bool &convex) {
    //two planes
    bool adjacent;
    if (shape1 < planes.size() && shape2 < planes.size()) {
        if (!planes[shape1].hasShape() || !planes[shape2].hasShape()) return false;
        Vector3d norm1 = planes[shape1].basis().row(2).transpose();
        Vector3d norm2 = planes[shape2].basis().row(2).transpose();
        if (std::fabs(norm1.dot(norm2)) > norm_adjacency_threshold) {
            return false;
        }
        if (planes[shape1].intersect(planes[shape2], intersectionEdges, adjacency_threshold)) {
            bool adj2 = false, adj1 = false;
            for (int index : clusters[shape1]) {
                Vector3d pt = cloud_->P.row(index).transpose();
                if (planes[shape2].contains3D(pt, adjacency_threshold, adjacency_threshold)) {
                    adj2 = true;
                    break;
                }
            }
            for (int index : clusters[shape2]) {
                Vector3d pt = cloud_->P.row(index).transpose();
                if (planes[shape1].contains3D(pt, adjacency_threshold, adjacency_threshold)) {
                    adj1 = true;
                    break;
                }
            }
            adjacent = adj2 && adj1;
        } else {
            return false;
        }
    } else if (shape1 < planes.size()) {
        //plane-cylinder
        const auto &cyl = cylinders[shape2-planes.size()];
        if (cyl.dir().dot(planes[shape1].normal()) < 0.866) return false;
        bool adj2 = false, adj1 = false;
        for (int index : clusters[shape1]) {
            Vector3d pt = cloud_->P.row(index).transpose();
            if (cyl.contains3D(pt, adjacency_threshold, adjacency_threshold)) {
                adj2 = true;
                break;
            }
        }
        for (int index : clusters[shape2]) {
            Vector3d pt = cloud_->P.row(index).transpose();
            if (planes[shape1].contains3D(pt, adjacency_threshold, adjacency_threshold)) {
                adj1 = true;
                break;
            }
        }
        adjacent = adj1 && adj2;
    } else {
        return false;
    }
    if (adjacent) {
        if (shape1 < planes.size()) {
            std::vector<size_t> densities(1, 0);
            neighborDensity(adjacency_threshold, adjacency_threshold, shape1, shape2, intersectionEdges,
                            densities, convex);
            std::cout << "adjacent density " << shape1 << '-' << shape2 <<  ": " << densities[0] << std::endl;
        }
    }
    return adjacent;
}

void FeatureExtractor::detect_adjacencies(double adjacency_threshold, double norm_adjacency_threshold) {
    adjacency.clear();
    adjacency.resize(clusters.size());
    for (int shape1 = 0; shape1 < planes.size() - 1; shape1++) {
        for (int shape2 = shape1+1; shape2 < clusters.size(); shape2++) {
            MultiRay3d intersectionEdges;
            bool convex;
            if (detect_adjacency(adjacency_threshold, norm_adjacency_threshold, shape1, shape2, intersectionEdges,
                                 convex)) {
                NeighborData nd;
                nd.intersectionEdges = std::move(intersectionEdges);
                nd.convex = convex;
                adjacency[shape2][shape1] = nd;
                adjacency[shape1][shape2] = std::move(nd);
            }
        }
    }
}

void FeatureExtractor::detect_parallel(double adjacency_threshold, double norm_parallel_threshold) {
    parallel.clear();
    parallel.resize(planes.size());
    for (int shape1 = 0; shape1 < planes.size() - 1; shape1++) {
        if (!planes[shape1].hasShape()) continue;
        for (int shape2 = shape1 + 1; shape2 < planes.size(); shape2++) {
            if (!planes[shape2].hasShape()) continue;
            if (std::fabs(planes[shape1].basis().row(2).dot(planes[shape2].basis().row(2))) >= norm_parallel_threshold) {
                if (planes[shape1].overlap(planes[shape2], -1, adjacency_threshold)) {
                    RowVector3d centroid1 = planes[shape1].points3D().colwise().mean();
                    RowVector3d centroid2 = planes[shape2].points3D().colwise().mean();
                    double dist1 = (centroid2-centroid1).dot(planes[shape1].basis().row(2));
                    double dist2 = (centroid1-centroid2).dot(planes[shape2].basis().row(2));
                    sorted_insert(parallel[shape1], shape2, dist1);
                    sorted_insert(parallel[shape2], shape1, dist2);
                }
            }
        }
    }
}

/*void FeatureExtractor::conditional_part_connections(double adjacency_threshold, double norm_adjacency_threshold, double norm_parallel_threshold) {
    conditional_connections.clear();
    conditional_connections.resize(planes.size());
    //DEBUG
    int num_found_parallel = 0;
    int num_found_ortho = 0;
    for (int shape1=0; shape1 < planes.size(); shape1++) {
        if (!planes[shape1].hasShape()) continue;
        for (int d=0; d<depths[shape1].size(); d++) {
            for (int shape2 = shape1 + 1; shape2 < planes.size(); shape2++) {
                if (!planes[shape2].hasShape()) continue;
                if (adjacency[shape1].find(shape2) == adjacency[shape1].end() && sorted_find(conditional_connections[shape1], shape2) == conditional_connections[shape1].end()) {
                    //add orthogonal adjacencies found at this offset depth
                    Vector3d pt_a, pt_b;
                    if (detect_adjacency(adjacency_threshold, norm_adjacency_threshold, shape1, shape2, pt_a, pt_b,
                                         -depths[shape1][d])) {
                        sorted_insert(conditional_connections[shape1], shape2, std::make_pair(2, std::make_pair(pt_a, pt_b)));
                        std::cout << "added ortho (" << d << ", " << shape2 << ") to " << shape1 << std::endl;
                        num_found_ortho++;
                    }
                }

                //add the neighbors of planes that are parallel at this depth
                if (std::fabs(planes[shape1].basis().row(2).dot(planes[shape2].basis().row(2))) >= norm_parallel_threshold) {
                    if (planes[shape1].overlap(planes[shape2], adjacency_threshold, adjacency_threshold, -depths[shape1][d])) {
                        for (const auto &adj : adjacency[shape2]) {
                            if (adjacency[shape1].find(adj.first) == adjacency[shape1].end() && sorted_find(conditional_connections[shape1], adj.first) == conditional_connections[shape1].end()) {
                                Vector3d offset_pt_a = adj.second.first + depths[shape1][d] * planes[shape1].basis().row(2).transpose();
                                Vector3d offset_pt_b = adj.second.second + depths[shape1][d] * planes[shape1].basis().row(2).transpose();
                                sorted_insert(conditional_connections[shape1], adj.first, std::make_pair(d, std::make_pair(offset_pt_a, offset_pt_b)));
                                std::cout << "added parallel (" << d << ", " << adj.first << ") to " << shape1 << std::endl;
                                num_found_parallel++;
                            }
                        }
                    }
                }
            }
        }
    }
    std::cout << "found " << num_found_ortho << " orthogonal neighbors and " << num_found_parallel << " through parallel connections" << std::endl;
}*/

void FeatureExtractor::detect_visibility(int max_views_per_cluster, double threshold, int min_view_support) {
    cluster_visibility.reserve(planes.size());
    pruned_cluster_visibility.reserve(planes.size());
    visible_clusters.reserve(reconstruction_->images.size());
    //for each bbox, for each point in the bbox, increment scores for each view
    for (size_t c=0; c<clusters.size(); c++) {
        sorted_map<int32_t, size_t> counts; //count points supporting each viewpoint per cluster
        if (min_view_support > 0) {
            for (const auto &point : reconstruction_->points) {
                bool contained = c < planes.size() ? planes[c].contains3D(point.second.xyz_, threshold, threshold)
                                                                  : cylinders[c - planes.size()].contains3D(point.second.xyz_, threshold, threshold);
                if (contained) {
                    for (int32_t image_id : point.second.image_ids_) {
                        if (!sorted_insert(counts, image_id, static_cast<size_t>(1))) {
                            sorted_get(counts, image_id) += 1;
                        }
                    }
                }
            }
        }
        std::vector<int32_t> views;
        std::vector<double> dotprods;
        //threshold the (view, cluster) pairs based on counts (if the min_view_threshold is nonzero) and relative normal
        for (const auto &pair : reconstruction_->images) {
            double dotprod = 1.0;
            if (min_view_support > 0 && (!sorted_contains(counts, pair.first) ||
                    sorted_get(counts, pair.first) < min_view_support)) {
                continue;
            }
            Vector3d camera_pos = pair.second.origin();
            bool visible = false;
            if (c < planes.size()) {
                Vector3d plane_center = planes[c].points3D(RowVector2d(0, 0)).cast<double>().transpose();
                Vector3d camera_offset = (camera_pos - plane_center).normalized();
                dotprod = camera_offset.dot(planes[c].basis().row(2).cast<double>());
                visible = dotprod > 0.0;
            } else visible = true; //assume cylinders are visible from any angle
            if (visible) {
                views.push_back(pair.first);
                dotprods.push_back(dotprod);
                visible_clusters[pair.first].push_back(c);
            }
        }
        std::vector<int> view_indices(views.size());
        std::iota(view_indices.begin(), view_indices.end(), 0);
        std::sort(view_indices.begin(), view_indices.end(), [&](int a, int b) {return dotprods[a] > dotprods[b];});
        std::vector<int32_t> pruned_views;
        for (int i=0; i<view_indices.size() && i < max_views_per_cluster; i++) {
            pruned_views.push_back(views[view_indices[i]]);
        }
        std::sort(pruned_views.begin(), pruned_views.end());
        std::sort(views.begin(), views.end());
        cluster_visibility.push_back(std::move(views));
        pruned_cluster_visibility.push_back(std::move(pruned_views));
    }
    for (auto &pair : visible_clusters) {
        std::sort(pair.second.begin(), pair.second.end());
    }

    //populate pruned visibility map, so that number of views samples stays constant with more reconstruction data for performance
    /*std::unordered_set<int32_t> visited_views;
    pruned_cluster_visibility.resize(planes.size());
    for (int c=0; c<planes.size(); c++) {
        const auto &group = cluster_visibility[c];
        int num_completed = 0;
        //scan for already loaded views to reuse
        for (auto image_id : group) {
            if (num_completed >= max_views_per_cluster) break;
            if (visited_views.find(image_id) != visited_views.end()) {
                pruned_cluster_visibility[c].push_back(image_id);
                num_completed++;
            }
        }
        //add any new missing views
        for (auto image_id : group) {
            if (num_completed >= max_views_per_cluster) break;
            if (visited_views.find(image_id) == visited_views.end()) {
                pruned_cluster_visibility[c].push_back(image_id);
                visited_views.insert(image_id);
                num_completed++;
            }
        }
    }
    return visited_views.size();*/
}

/*void FeatureExtractor::extract_image_shapes(const Settings &settings, double threshold, const Ref<const MatrixX3d> &colors) {
    for (BoundedPlane &plane : planes) {
        plane.setCurrentShape(1);
    }
    std::unordered_map<int32_t, DepthSegmentation> image_labels;
    //image_labels.reserve(reconstruction_->images.size());
    //attempt to reuse segmentations as much as possible
    for (int c=0; c<planes.size(); c++) {
        const auto &group = cluster_visibility[c];
        int num_completed = 0;
        for (auto image_id : group) {
            if (image_labels.find(image_id) != image_labels.end()) {
                num_completed++;
            }
        }
        for (auto image_id : group) {
            if (image_labels.find(image_id) == image_labels.end() && num_completed < settings.segmentation_max_views_per_cluster) {
                Image &image = reconstruction_->images[image_id];
                const auto &camera = reconstruction_->cameras[image.camera_id_];
                if (camera.model_id_ != 1) {
                    std::cerr << "unsupported camera model" << std::endl;
                    continue;
                }
                cv::Mat segmentation_display;
                DepthSegmentation segmentation(planes, cylinders, visible_clusters[image_id], image, camera, cloud_, settings.segmentation_scale, settings.correction_factor);
                if (!segmentation.isValid()) {
                    std::cerr << "image/depth data not properly loaded for " << image.image_name_ << std::endl;
                    continue;
                }
                segmentation.computeDepthSegmentation(threshold);
                display_segmentation(segmentation.getSegmentation(), segmentation_display, colors);
                cv::imwrite("depth_seg_" + std::to_string(image_id) + ".png", segmentation_display);

                if (settings.segmentation_min_pixel_support > 0) {
                    segmentation.outlierRemoval(settings.segmentation_min_pixel_support);
                    display_segmentation(segmentation.getSegmentation(), segmentation_display, colors);
                    cv::imwrite("depth_seg_" + std::to_string(image_id) + "_outlier.png", segmentation_display);
                }
                std::cout << "segmenting image " << image_id << "...";
                auto t_start = clock();
                segmentation.energyMinimization(settings.segmentation_data_weight, settings.segmentation_smoothing_weight, settings.segmentation_penalty, settings.segmentation_sigma, settings.segmentation_gmm_components, settings.segmentation_iters, settings.segmentation_levels);
                auto t_duration = clock() - t_start;
                float time_sec = static_cast<float>(t_duration)/CLOCKS_PER_SEC;
                std::cout << " completed in " << time_sec << " seconds" << std::endl;
                display_segmentation(segmentation.getSegmentation(), segmentation_display, colors);
                cv::imwrite("depth_seg_" + std::to_string(image_id) + "_grow.png", segmentation_display);

                std::cout << "cleaning segmentation... " << std::endl;
                t_start = clock();
                segmentation.cleanSegmentation(3, settings.segmentation_clean_precision_threshold, settings.segmentation_clean_recall_threshold, true, false);
                t_duration = clock() - t_start;
                time_sec = static_cast<float>(t_duration)/CLOCKS_PER_SEC;
                std::cout << " completed in " << time_sec << " seconds" << std::endl;
                display_segmentation(segmentation.getSegmentation(), segmentation_display, colors);
                cv::imwrite("depth_seg_" + std::to_string(image_id) + "_cleaned.png", segmentation_display);
                image_labels.emplace(image_id, segmentation);
                num_completed++;
            }
        }
    }

    std::cout << "extracting shapes from images" << std::endl;
    vis_segmentation_shapes.reserve(planes.size());
    for (size_t c=0; c < planes.size(); c++) {
        const std::vector<int32_t>& group = cluster_visibility[c];
        const BoundedPlane& bbox = planes[c];
        std::vector<std::vector<std::vector<Vector3d>>> all_shapes_3D;
        for (int32_t image_id : group) {
            Image& image = reconstruction_->images[image_id];
            const auto &camera = reconstruction_->cameras[image.camera_id_];
            if (camera.model_id_ != 1) {
                std::cout << "unsupported camera model" << std::endl;
                continue;
            }
            Array2d fdist(camera.params_.data());
            Array2d principal_point(camera.params_.data() + 2);
            Vector3d camera_pos = -(image.rot_.conjugate() * image.trans_);
            auto it = image_labels.find(image_id);
            if (it == image_labels.end()) continue;
            cv::Mat im_label = it->second.getSegmentation() == c;
            std::vector<std::vector<cv::Point>> contours;
            cv::Mat hierarchy;
            cv::findContours(im_label, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_TC89_L1);
            size_t num_valid = 0;
            std::vector<std::vector<Vector3d>> contours3D;
            contours3D.reserve(contours.size());
            for (const auto &contour : contours) {
                std::vector<Vector3d> contour3D;
                contour3D.reserve(contour.size());
                for (const cv::Point &pt : contour) {
                    Vector3d ray_dir(pt.x / settings.segmentation_scale, pt.y / settings.segmentation_scale, 1);
                    ray_dir.head(2) -= principal_point.matrix();
                    ray_dir.head(2).array() /= (fdist);
                    ray_dir.normalize();
                    ray_dir = image.rot_.conjugate() * ray_dir;
                    double t;
                    if (bbox.intersectRay(camera_pos, ray_dir, t)) {
                        num_valid++;
                    }
                    Vector3d pt3d = camera_pos + t * ray_dir;
                    contour3D.push_back(std::move(pt3d));
                }
                contours3D.push_back(std::move(contour3D));
            }
            all_shapes_3D.push_back(std::move(contours3D));
        }
        std::vector<std::vector<Vector3d>> combined_shapes;
        for (const auto &shape : all_shapes_3D) {
            std::copy(shape.begin(), shape.end(), std::back_inserter<std::vector<std::vector<Vector3d>>>(combined_shapes));
        }
        vis_segmentation_shapes.push_back(std::move(combined_shapes));
    }
}*/

bool FeatureExtractor::neighborDensity(double thickness_spacing, double margin, int basePrimitive, int neighborPrimitive, const MultiRay3d &intersectionEdges, std::vector<size_t> &localDensities, bool &convex) {
    convex = true;
    size_t thickness_steps = localDensities.size();
    const auto& extruded_primitive = planes[basePrimitive];
    if (neighborPrimitive < planes.size()) {
        BoundedPlane &neighbor = planes[neighborPrimitive];
        if (!neighbor.hasShape()) return false;

        Vector2d origin2d = neighbor.project(intersectionEdges.o.transpose()).transpose();
//        Vector2d ptB = neighbor.project(intersectionEdge.second.transpose()).transpose();
        // obtain 2D point cloud belonging to the neighboring primitive,
        // project them onto the projected axis of extrusion
        size_t N = clusters[neighborPrimitive].size();
        MatrixX3d neighborPoints(N, 3);
        for (size_t p = 0; p < N; p++) {
            neighborPoints.row(p) = cloud_->P.row(clusters[neighborPrimitive][p]);
        }
        MatrixX2d neighborPoints2D = neighbor.project(neighborPoints).rowwise() - origin2d.transpose();
        neighborPoints.resize(0, 3);
        //assume extrusion along negative normal
        Vector2d dir = -neighbor.project(
                extruded_primitive.basis().row(2)).transpose().normalized();
        Vector2d tangent = neighbor.project(intersectionEdges.d.transpose()).transpose();
        ArrayXd x = neighborPoints2D * tangent;
        ArrayXd z = neighborPoints2D * dir;
        //double z_bottom = z.maxCoeff();
        //check that most points are on the positive side of extrusion
        //TODO: threshold parameter for second condition
        //TODO: second condition should be for points within intersection segment only to prevent loop backs
        //or maybe not, since the only faces that would have this problem shouldn't be cut faces anyway
//        if (zStart + margin < zEnd /* && z_bottom > zStart - 3*thickness_spacing */) {
            //accumulate samples for all points in this cluster
            //size_t num_in_range = 0;
            size_t num_backward = 0;
            for (size_t p = 0; p < N; p++) {
                bool insideInterval = false;
                for (size_t i=0; i<intersectionEdges.size(); ++i) {
                    if (intersectionEdges.ranges[i].first + margin <= x(p) && x(p) < intersectionEdges.ranges[i].second - margin) {
                        insideInterval = true;
                        break;
                    }
                }
                if (insideInterval) {
                    //num_in_range++;
                    int index = static_cast<int>(std::floor(z[p] / thickness_spacing));
                    if (index >= 0) {
                        if (index < thickness_steps) {
                            localDensities[index]++;
                        }
                    } else if (index == -1) {
                        num_backward++;
                    }
                }
            }
            std::cout << "backwards " << basePrimitive << '-' << neighborPrimitive << ": " << num_backward << " vs " << localDensities[0] << std::endl;
            if (num_backward > localDensities[0]) {
                convex = false;
            }
//        } else return false;
    } else {
        Cylinder &neighbor = cylinders[neighborPrimitive - planes.size()];
        size_t N = clusters[neighborPrimitive].size();
        MatrixX3d neighborPoints(N, 3);
        for (size_t p = 0; p < N; p++) {
            neighborPoints.row(p) = cloud_->P.row(clusters[neighborPrimitive][p]);
        }
        //assume extrusion along negative normal
        Vector3d dir = -extruded_primitive.basis().row(2).transpose();
        double zStart = extruded_primitive.offset();
        ArrayXd z = neighborPoints * dir;
        double zEnd = std::numeric_limits<double>::lowest();
        for (size_t p = 0; p < N; p++) {
            if (z(p) > zEnd) {
                zEnd = z(p);
            }
        }
//        if (zStart + margin < zEnd) {
            size_t numBackward = 0;
            for (size_t p = 0; p < N; p++) {
                //num_in_range++;
                int index = static_cast<int>(std::floor((z[p] - zStart) / thickness_spacing));
                if (index >= 0) {
                    if (index < thickness_steps) {
                        localDensities[index]++;
                    }
                } else if (index == -1) {
                    numBackward++;
                }
            }
            if (numBackward > localDensities[0]) {
                convex = false;
            }
            //maxNeighborDepth = std::max(zEnd - zStart + depthLowerBound, maxNeighborDepth);
//        } else return false;
    }
    return true;
}


double spatial_discount(double fac, double depth) {
    return exp(-fac * depth);
}

void FeatureExtractor::compute_depths(double thickness_spacing, int thickness_steps, double edge_threshold, double edge_detection_threshold, double spatial_discounting_factor) {
    depths.clear();
    depths.resize(planes.size());
    opposing_planes.clear();
    opposing_planes.resize(planes.size());
    double depthLowerBound = OFFSET_STEPS * thickness_spacing; //minimum depth value
    for (size_t c=0; c < planes.size(); c++) {
        BoundedPlane &extruded_primitive = planes[c];
        //double maxNeighborDepth = 0;
        bool hasOpposingPlane = false;
        double nearestPlaneDist = thickness_steps * thickness_spacing;
        int parallelInd = -1;
        bool hasDepth = false;
        if (extruded_primitive.hasShape()) {
            double area = extruded_primitive.getShape(extruded_primitive.getCurrentShape()).area();
            double circumference = extruded_primitive.getShape(extruded_primitive.getCurrentShape()).circumference();

            std::cout << "finding depth for part " << c << std::endl;
            //first collect points and create data structures for analysis
            std::vector<size_t> pointDensities(thickness_steps, 0); // number of points per depth slice
            std::vector<size_t> localDensities(thickness_steps, 0); // densities of individual neighbors (to be reused)
            std::vector<double> colorAvg(thickness_steps, 0); // color average per depth slice
            std::vector<double> weights(thickness_steps, 0); // weights for weighted average
            std::vector<size_t> usedNeighbors;
            usedNeighbors.reserve(adjacency[c].size());
            double maxDepth = 5 * area / circumference;
//            maxDepth = std::min(maxDepth, (maxPt_-minPt_).minCoeff()/2);
            std::cout << "maxDepth: " << maxDepth << std::endl;

            //look for parallel planes to constrain depth
            for (const auto &pair : parallel[c]) {
                if (-pair.second > depthLowerBound && -pair.second < nearestPlaneDist) {
                    nearestPlaneDist = -pair.second;
                    parallelInd = pair.first;
                }
            }
            std::cout << "nearest plane dist: " << nearestPlaneDist << "from plane " << parallelInd << std::endl;
            //add all nearby planes to opposing list
            for (const auto &pair : parallel[c]) {
                if (-pair.second > depthLowerBound && std::abs(-pair.second-nearestPlaneDist) < edge_threshold) {
                    double dotProd = planes[pair.first].basis().row(2).dot(extruded_primitive.basis().row(2));
                    if (dotProd < 0) {
                        hasOpposingPlane = true;
                        break;
                    }
                }
            }

            // add point densities of neighbors by projecting to their planes
            for (const auto &pair : adjacency[c]) {
                if (pair.second.convex) {
                    size_t n = pair.first;
                    std::cout << "using neighbor " << n << std::endl;

                    bool concave;
                    neighborDensity(thickness_spacing, edge_threshold, c, n, pair.second.intersectionEdges,
                                    localDensities, concave);

                    //commit update
                    for (size_t b = 0; b < thickness_steps; b++) {
                        pointDensities[b] += localDensities[b];
                        localDensities[b] = 0;
                    }
                    usedNeighbors.push_back(n);
                }
            }
            std::cout << "neighbors used: " << usedNeighbors.size() << "/" << adjacency[c].size()
                      << std::endl;
//            std::cout << '(' << numPlanes << " planes and " << numCylinders << " cylinders" << ')' << std::endl;
            if (!usedNeighbors.empty()) {
                // Detect depths at which there are maximum discontinuities
                std::vector<double> derivatives = robust_derivative(pointDensities, false);
                std::transform(derivatives.begin(), derivatives.end(), derivatives.begin(), [](double a) {return -a;});
                std::vector<double> derivativesSuppressed(derivatives.size());
                nonmax_suppression<double>(derivatives.begin(), derivatives.end(), derivativesSuppressed.begin());


                /*std::vector<double> lum_derivatives = robust_derivative(colorAvg, true);
                std::vector<double> lum_derivatives_suppressed(lum_derivatives.size());
                nonmax_suppression<double>(lum_derivatives.begin(), lum_derivatives.end(), lum_derivatives_suppressed.begin());*/

                if (spatial_discounting_factor > 0) {
                    for (int i = 0; i < derivativesSuppressed.size(); i++) {
                        derivativesSuppressed[i] *= spatial_discount(spatial_discounting_factor,
                                                                      ((i + OFFSET_STEPS) + 0.5) *
                                                                      thickness_spacing);
                    }
                    /*for (int i = 0; i < lum_derivatives_suppressed.size(); i++) {
                        lum_derivatives_suppressed[i] *= spatial_discount(spatial_discounting_factor,
                                                                          ((i+OFFSET_STEPS) + 0.5) * thickness_spacing);
                    }*/
                }

                size_t depthUpperBoundIndex = std::min(derivatives.size(), static_cast<size_t>(std::floor((maxDepth - depthLowerBound) / thickness_spacing)));

                if (depthUpperBoundIndex == 0) {
                    std::cout << "warning: zero upper bound depth index" << std::endl;
                } else {
                    auto maxDerivIt = std::max_element(derivativesSuppressed.begin(),
                                                       derivativesSuppressed.begin() + depthUpperBoundIndex);

                    //std::vector<size_t> max_lum_deriv_indices = top_k_indices(lum_derivatives_suppressed.begin(), lum_derivatives_suppressed.end(), num_candidates);
                    //TODO: use both derivative sets intelligently
                    depths[c] = ((std::distance(derivativesSuppressed.begin(), maxDerivIt) + OFFSET_STEPS) + 0.5) *
                                thickness_spacing;
                    hasDepth = true;
                }
            }

            if (depths[c] < edge_threshold) {
                depths[c] = 0;
                hasDepth = false;
            }

            if (hasOpposingPlane && nearestPlaneDist < maxDepth) {
                if (!hasDepth) {
                    std::cout << "adding parallel plane distance to " << parallelInd << ": " << nearestPlaneDist
                              << std::endl;
                    depths[c] = nearestPlaneDist;
                    hasDepth = true;
                } else {
                    if (depths[c] > nearestPlaneDist - edge_threshold) {
                        std::cout << "snapping to plane " << parallelInd << std::endl;
                        depths[c] = nearestPlaneDist;
                    }
                }
            }
        }
        if (hasDepth) {
            std::cout << "depth chosen: " << depths[c] << "; min shape dist: " << nearestPlaneDist << std::endl;
            //add all nearby planes to opposing list
            for (const auto &pair : parallel[c]) {
                if (-pair.second > depthLowerBound && std::abs(-pair.second-depths[c]) < edge_threshold) {
                    double dotProd = planes[pair.first].basis().row(2).dot(extruded_primitive.basis().row(2));
                    if (dotProd < 0) {
                        opposing_planes[c].push_back(pair.first);
                    }
                }
            }
        } else {
            std::cout << "No depth or neighbors found for " << c << std::endl;
            depths[c] = -1;
        }
        /*for (int i=0; i<max_lum_deriv_indices.size(); i++) {
            int ind = max_lum_deriv_indices[i];
            if (lum_derivatives[ind] > edge_detection_threshold) {
                depths[c].push_back(((ind+OFFSET_STEPS) + 0.5) * thickness_spacing);
            } else {
                break;
            }
        }*/
    }
}

bool FeatureExtractor::filter_depths(int num_clusters, double min_eigenvalue, bool modify_existing, double minDepth) {

    if (!modify_existing) {
        bool hasMissing = false;
        for (auto d : depths) {
            if (d < 0) {
                hasMissing = true;
                break;
            }
        }
        if (!hasMissing) return true;
    }
    GaussianMixture gmm(num_clusters, 1, min_eigenvalue);
    //size_t totalDepths = std::accumulate(depths.begin(), depths.end(), 0, [](size_t a, const std::vector<double> &b) {return a + b.size();});
    std::vector<double> data;
    for (auto d : depths) {
        if (d > 0)
            data.push_back(d);
    }
    Map<VectorXd> dataMap(data.data(), data.size());
    int iters = gmm.learn(dataMap);
    if (!gmm.success()) {
        return false;
    }
    std::cout << "clustered depths model: " << std::endl;
    std::cout << gmm << std::endl;

    if (modify_existing) {
        MatrixXd logprob = gmm.logp_z_given_data(dataMap);
        size_t data_ind = 0;
        for (auto d : depths) {
            int ind;
            logprob.row(data_ind).maxCoeff(&ind);
            d = gmm.means()(ind);
            data_ind++;
        }
    }
    // add most likely depth to empty depth lists
    int ind;
    gmm.log_probs().maxCoeff(&ind);
    int num_added = 0;
    for (int c=0; c<planes.size(); ++c) {
        if (depths[c] < 0) {
            depths[c] = gmm.means()(ind);

            //find minimum dot product with normal
            Vector3d n = planes[c].normal();
            double minH = std::numeric_limits<double>::max();
            for (int r=0; r<cloud_->P.rows(); ++r) {
                minH = std::min(minH, cloud_->P.row(r).dot(n));
            }

            double currDotProd = -planes[c].offset();
            double oppositeDotProd = -(planes[c].offset() + depths[c]);

            if (oppositeDotProd < minH) {
                std::cout << "plane "<< c << " out of bounds at "<< depths[c] <<"; fixing to ";
                depths[c] = std::max(minDepth, currDotProd - minH);
                std::cout << depths[c] << std::endl;
            }
            num_added++;
        }
    }
    std::cout << "added " << gmm.means()(ind) << " to " << num_added << " parts" << std::endl;
    return true;
}

void FeatureExtractor::setCurrentShape(int idx) {
    for (auto &plane : planes) {
        plane.setCurrentShape(idx);
    }
}

/*void FeatureExtractor::compute_joint_depths(const Ref<const MatrixX3d> &allpoints, std::vector<PointMembership> &table) {
    size_t N = allpoints.rows();
    std::map<std::pair<int, int>, std::pair<int, int>> joint_depths;
    for (int c1=0; c1<planes.size(); c1++) {
        for (const auto &pair : adjacency[c1]) {
            int c2 = pair.first;
            if (c2 >= planes.size()) continue;
            Vector3d ab = pair.second.intersectionEdge.second - pair.second.intersectionEdge.first;
            double norm2 = ab.squaredNorm();
            for (int i = 0; i < N; i++) {
                int d1 = get_min_depth_index(c1, allpoints.row(i).transpose());
                int d2 = get_min_depth_index(c2, allpoints.row(i).transpose());
                if (d1 < 0 || d2 < 0) {
                    continue;
                }
                double dot = (allpoints.row(i).transpose() - pair.second.intersectionEdge.first).dot(ab);
                if (dot >= 0 && dot <= norm2) {
                    int c1tmp = c1;
                    int c2tmp = c2;
                    if (c2tmp < c1tmp) {
                        std::swap(c1tmp, c2tmp);
                        std::swap(d1, d2);
                    }
                    auto key = std::make_pair(c1tmp, c2tmp);
                    table[i].joint_depth_membership[key] = std::make_pair(d1, d2);
                }
            }
        }
    }
}*/

/*std::vector<PointMembership> FeatureExtractor::process_samples(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const std::vector<size_t> &indices, const std::vector<size_t> &sample_counts) {
    int N = indices.size();
    std::vector<size_t> sorted_indices(N);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(), [&](int a, int b) {return sample_counts[a] > sample_counts[b];});
    MatrixX3d allpoints(N, 3);
    for (int i=0; i<N; i++) {
        allpoints.row(i) = (*cloud)[indices[sorted_indices[i]]].getVector3fMap().transpose().cast<double>();
    }
    PointMembership def;
    def.depth_membership.resize(planes.size());
    def.shape_membership.resize(planes.size());
    std::vector<PointMembership> table(N, def);
    //single parts
    std::vector<int> point_indices(N);
    VectorXd point_depths(N);
    std::iota(point_indices.begin(), point_indices.end(), 0);
    for (int c=0; c<planes.size(); c++) {
        const auto &part = planes[c];
        MatrixX2d projected = part.project(allpoints);
        for (int i=0; i<N; i++) {
            //TODO: handle multiple shapes
            table[i].shape_membership[c].push_back(part.contains(projected.row(i).transpose()));
        }

        //handle depths
        point_depths = -((part.basis().row(2) * allpoints.transpose()).array() + part.offset()).transpose();

        std::sort(point_indices.begin(), point_indices.end(),
                [&](int a, int b) {
            return point_depths(a) < point_depths(b);
        });
        int depth_ptr = 0;
        if (depths[c].empty()) {
            for (int i=0; i<N; i++) {
                table[i].depth_membership[c] = 0;
            }
        } else {
            for (int i = 0; i < N; i++) {
                double depth = point_depths(point_indices[i]);
                double depthbound = depths[c][depth_ptr];
                while (depth > depthbound && depth_ptr < depths[c].size() - 1) {
                    depthbound = depths[c][++depth_ptr];
                }
                if (depth <= depthbound && depth >= 0) {
                    table[i].depth_membership[c] = depth_ptr;
                } else {
                    table[i].depth_membership[c] = -1;
                    for (int j = 0; j < table[i].shape_membership[c].size(); j++) {
                        table[i].shape_membership[c][j] = false;
                    }
                }
            }
        }
    }
    //volumes
    for (size_t i=0; i<table.size(); i++) {
        table[i].count = sample_counts[i];
    }

    //joint depth extrusions
    compute_joint_depths(allpoints, table);
    return table;
}

std::vector<PointMembership>
FeatureExtractor::process_samples(const Ref<const MatrixX3d> &cloud, const std::vector<size_t> &indices, const std::vector<size_t> &sample_counts,
                                  const std::vector<std::vector<int>> &labels) {
    int N = indices.size();
    PointMembership def;
    def.depth_membership.resize(planes.size());
    def.shape_membership.resize(planes.size());
    std::vector<PointMembership> table(N, def);
    for (size_t i=0; i<N; i++) {
        for (size_t c=0; c<planes.size(); c++) {
            if (labels[i][c] < 0) {
                table[i].shape_membership[c].push_back(false);
                table[i].depth_membership[c] = 0;
            } else {
                table[i].shape_membership[c].push_back(true);
                table[i].depth_membership[c] = labels[i][c];
            }
        }
        table[i].count = sample_counts[i];
    }
    compute_joint_depths(cloud, table);
    return table;
}

bool FeatureExtractor::save_samples(const std::string &filename, const std::vector<PointMembership> &samples, int max_constraints) {
    if (samples.empty()) return false;
    std::ofstream of(filename);
    if (!of) return false;
    of << planes.size() << std::endl;
    for (int i=0; i<planes.size(); i++) {
        of << samples[0].shape_membership[i].size() << ' ' << depths[i].size() << std::endl;
    }
    for (int i=0; i<planes.size(); i++) {
        for (const auto &pair : adjacency[i]) {
            if (pair.first < planes.size()) {
                of << pair.first << ' ';
            }
        }
        of << std::endl;
    }
    for (int i=0; i<planes.size(); i++) {
        for (const auto &pair : conditional_connections[i]) {
            of << pair.second.first << ' ' << pair.first << ' ';
        }
        of << std::endl;
    }
    //TODO: compute volumes based on depth and shape
    //
    int num_samples = max_constraints > 0 ? std::min(max_constraints, static_cast<int>(samples.size())) : samples.size();
    of << num_samples << std::endl;
    for (int i=0; i<num_samples; i++) {
        const auto & sample = samples[i];
        for (int j=0; j<planes.size(); j++) {
            of << sample.depth_membership[j] << ' ';
        }
        of << std::endl;
    }
    for (int i=0; i<num_samples; i++) {
        const auto & sample = samples[i];
        for (int j=0; j<planes.size(); j++) {
            for (int k : sample.shape_membership[j]) {
                of << k << ' ';
            }
        }
        of << std::endl;
    }
    for (int i=0; i<num_samples; i++) {
        const auto & sample = samples[i];
        for (const auto &pair : sample.joint_depth_membership) {
            of << pair.first.first << ' ' << pair.first.second << ' ' << pair.second.first << ' ' << pair.second.second << ' ';
        }
        of << std::endl;
    }
    for (const auto & sample : samples) {
        of << sample.count << std::endl;
    }
    return true;
}*/

/*bool FeatureExtractor::save_samples(const std::string &filename, pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const std::vector<size_t> &indices, const std::vector<size_t> &sample_counts) {
    return save_samples(filename, process_samples(cloud, indices, sample_counts));
}*/

/*bool FeatureExtractor::save_samples(const std::string &filename,
                                    pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
                                    const std::vector<size_t> &indices,
                                    const std::vector<size_t> &sample_counts,
                                    const std::vector<std::vector<int>> &labels,
                                    int max_constraints) {
    return save_samples(filename, process_samples(cloud, indices, sample_counts, labels), max_constraints);
}

MatrixX3d FeatureExtractor::generate_samples_biased() {
    pcl::PointCloud<pcl::PointXYZ>::Ptr samples(new pcl::PointCloud<pcl::PointXYZ>);
    for (int i=0; i<planes.size(); i++) {
        //sample one point on the face and duplicate it along the extrusion direction
        Vector3d dir = -planes[i].basis().row(2).transpose();
        Vector3d point_on_face = cloud_->row(clusters[i][clusters[i].size()/2]).head(3).transpose();
        double lastdepth = 0.0;
        for (double depth : depths[i]) {
            double meandepth = (depth + lastdepth) * 0.5;
            Vector3d centerpoint = point_on_face + dir * meandepth;
            samples->push_back(pcl::PointXYZ(centerpoint.x(), centerpoint.y(), centerpoint.z()));
            lastdepth = depth;
        }
        lastdepth = 0.0;
        //sample points for each combination of depths at the midpoint of each connection edge
        for (const auto &pair : adjacency[i]) {
            Vector3d midpoint = (pair.second.first + pair.second.second) * 0.5;
            int j = static_cast<int>(pair.first);
            Vector3d dirj = -planes[j].basis().row(2).transpose();
            for (int k=0; k<depths[i].size(); k++) {
                double depthi = depths[i][k];
                double meandepthi = (depthi + lastdepth) * 0.5;
                double lastdepthj = 0.0;
                for (double depthj : depths[j]) {
                    double meandepthj = (depthj + lastdepthj) * 0.5;
                    Vector3d edgepoint = midpoint + dir * meandepthi + dirj * meandepthj;
                    samples->push_back(pcl::PointXYZ(edgepoint.x(), edgepoint.y(), edgepoint.z()));
                    lastdepthj = depthj;
                }
                lastdepth = depthi;
            }
        }
    }
    return samples;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr FeatureExtractor::generate_samples_random(size_t count) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr samples(new pcl::PointCloud<pcl::PointXYZ>);
    std::uniform_real_distribution<double> x_sampler(minPt_.x(), maxPt_.x());
    std::uniform_real_distribution<double> y_sampler(minPt_.y(), maxPt_.y());
    std::uniform_real_distribution<double> z_sampler(minPt_.z(), maxPt_.z());
    for (size_t i=0; i<count; i++) {
        pcl::PointXYZ pt;
        pt.x = x_sampler(random_);
        pt.y = y_sampler(random_);
        pt.z = z_sampler(random_);
        samples->push_back(pt);
    }
    return samples;
}*/

/*int FeatureExtractor::get_min_depth_index(int part, const Ref<const Vector3d> &point) {
    double depth = -(planes[part].basis().row(2) * point + planes[part].offset());
    if (depth < 0) {
        return -1;
    } else {
        auto it = std::lower_bound(depths[part].begin(), depths[part].end(), depth);
        if (it == depths[part].end()) {
            return -1;
        } else {
            return std::distance(depths[part].begin(), it);
        }
    }
}*/

/*void FeatureExtractor::analyze_samples(pcl::PointCloud<pcl::PointXYZ>::ConstPtr samples, std::vector<size_t> &indices, std::vector<size_t> &sample_counts, std::vector<std::vector<int>> &all_labels, int min_overlap, double contour_threshold, int point_cloud_stride, int k) {
    std::map<std::vector<int>, std::pair<std::vector<size_t>, double>> mappings;
    std::vector<int> part_sizes(planes.size(), 0);
    bool use_winding_number = contour_threshold > 0;
    MatrixX3d Q;
    size_t num_potentially_inside = 0;
    std::vector<size_t> point_index_to_wn_index;
    if (use_winding_number) {
        Q.resize(samples->size(), 3);
        point_index_to_wn_index.resize(samples->size(), -1);
    }
    for (size_t p=0; p<samples->size(); p++) {
        Vector3d pt_e = (*samples)[p].getVector3fMap().cast<double>();
        std::vector<int> labels(planes.size());
        int overlap = 0;
        for (int i=0; i<planes.size(); i++) {
            if (!planes[i].hasShape() || !planes[i].contains(planes[i].project(pt_e.transpose()).transpose())) {
                labels[i] = -1;
            } else {
                labels[i] = get_min_depth_index(i, pt_e);
                if (labels[i] >= 0) {
                    overlap++;
                    part_sizes[i]++;
                }
            }
        }
        if (overlap >= min_overlap) {
            mappings[labels].first.push_back(p);
            if (use_winding_number) {
                point_index_to_wn_index[p] = num_potentially_inside;
                Q.row(num_potentially_inside++) = pt_e.transpose();
            }
        }
    }
    VectorXd WN;
    if (use_winding_number) {
        Q = Q.block(0, 0, num_potentially_inside, 3);
        WN = query_winding_number(Q);
    }

    //populate scores
    for (auto &pair : mappings) {
        //default
        //pair.second.second = pair.second.first.size();

        double maxscore = 0.0;
        for (int i=0; i<planes.size(); i++) {
            if (pair.first[i] >= 0) {
                double score = static_cast<double>(pair.second.first.size())/part_sizes[i];
                if (score > maxscore) {
                    maxscore = score;
                }
            }

        }
        pair.second.second = maxscore;
    }

    std::vector<const std::pair<const std::vector<int>, std::pair<std::vector<size_t>, double>> *> pairs;
    pairs.reserve(mappings.size());
    for (const auto &pair : mappings) {
        pairs.push_back(&pair);
    }
    std::sort(pairs.begin(), pairs.end(), [&](const std::pair<const std::vector<int>, std::pair<std::vector<size_t>, double>> * a, const std::pair<const std::vector<int>, std::pair<std::vector<size_t>, double>> * b) {return a->second.second > b->second.second;});
    for (size_t i=0; i<mappings.size(); i++) {
        const auto &term_indices = pairs[i]->second.first;
        size_t inside_count = 0;
        if (use_winding_number) {
            for (size_t p = 0; p < term_indices.size(); p++) {
                if (WN(point_index_to_wn_index[p]) > contour_threshold) {
                    ++inside_count;
                }
            }
            std::cout << "inside count " << i << ": " << inside_count << '/' << term_indices.size() << std::endl;
        }
        if (!use_winding_number || 2 * inside_count > term_indices.size()) {
            sample_counts.push_back(term_indices.size());
            indices.push_back(term_indices[std::uniform_int_distribution<size_t>(0, term_indices.size()-1)(random_)]);
            all_labels.push_back(pairs[i]->first);
        }
    }
}*/



void FeatureExtractor::export_globfit(const std::string &filename, int stride, double scale) {
    std::ofstream of(filename);
    of << "# Number of Points" << std::endl;
    std::vector<int> indices(cloud_->P.rows());
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<int> inv_indices(cloud_->P.rows(), -1);
    int N = cloud_->P.rows();
    if (stride > 1) {
        N /= stride;
        std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device{}()));
    }
    of << N << std::endl;
    of << "# Here comes the " << N << " Points" << std::endl <<
    "# point_x point_y point_z normal_x normal_y normal_z confidence" << std::endl;
    for (int i=0; i<N; i++) {
        inv_indices[indices[i]] = i;
        RowVector3d pt = cloud_->P.row(indices[i]);
        RowVector3d nt = cloud_->N.row(indices[i]);
        of << pt(0) * scale << ' ' << pt(1) * scale << ' ' << pt(2) * scale << ' ' << nt(0) << ' ' << nt(1) << ' ' << nt(2)
           << ' ' << 1 << std::endl;
    }
    of << "# End of Points" << std::endl << std::endl << "# Number of Primitives" << std::endl;
    of << planes.size() + cylinders.size() << std::endl;
    of << "# Here comes the " << planes.size() + cylinders.size() << " Primitives" << std::endl;
    int primitive_index = 0;
    for (int i = 0; i < planes.size(); i++) {
        std::vector<int> point_indices;
        if (stride > 1) {
            for (int j : clusters[i]) {
                if (inv_indices[j] >= 0) {
                    point_indices.push_back(inv_indices[j]);
                }
            }
        } else point_indices = clusters[i];
        if (!point_indices.empty()) {
            of << "# Primitive " << primitive_index++ << std::endl;
            of << "# plane normal_x normal_y normal_z d" << std::endl;
            const auto &plane = planes[i];
            of << "plane " << plane.basis()(2, 0) << ' ' << plane.basis()(2, 1) << ' ' << plane.basis()(2, 2) << ' '
               << plane.offset() * scale << std::endl;
            of << "# points idx_1 idx_2 idx_3 ... " << std::endl;
            of << "points ";
            for (int j=0; j<point_indices.size(); j++) {
                of << point_indices[j];
                if (j < point_indices.size()-1) {
                    of << ' ';
                }
            }
            of << std::endl << std::endl;
        }
    }
    for (int i = 0; i < cylinders.size(); i++) {
        std::vector<int> point_indices;
        if (stride > 1) {
            for (int j : clusters[planes.size() + i]) {
                if (inv_indices[j] >= 0) {
                    point_indices.push_back(inv_indices[j]);
                }
            }
        } else point_indices = clusters[planes.size() + 1];
        if (!point_indices.empty()) {
            of << "# Primitive " << primitive_index++ << std::endl;
            of << "# cylinder normal_x normal_y normal_z point_x point_y point_z radius" << std::endl;
            const auto &cylinder = cylinders[i];
            of << "cylinder " << cylinder.dir().x() << ' ' << cylinder.dir().y() << ' ' << cylinder.dir().z() << ' '
               << cylinder.point().x() * scale << ' ' << cylinder.point().y() * scale << ' ' << cylinder.point().z() * scale << ' '
               << cylinder.radius() * scale << std::endl;
            of << "# points idx_1 idx_2 idx_3 ... " << std::endl;
            of << "points ";
            for (int j=0; j<point_indices.size(); j++) {
                of << point_indices[j];
                if (j < point_indices.size()-1) {
                    of << ' ';
                }
            }
            of << std::endl << std::endl;
        }
    }
    of << std::endl << "# End of Primitives" << std::endl;
}

bool parse_point_indices(std::istream &is, std::vector<size_t> &indices) {
    size_t ind;
    std::string lbl;
    if (is >> lbl && lbl == "points") {
        while (is >> ind) {
            indices.push_back(ind);
        }
        return true;
    } else {
        return false;
    }
}

bool getline_ignore_empty(std::istream &is, std::string &line) {
    while (std::getline(is, line)) {
        if (line.find_first_not_of("\t\n\v\f\r") == std::string::npos || line.find('#') != std::string::npos) {
            continue;
        } else {
            return true;
        }
    }
    return false;
}

bool FeatureExtractor::import_globfit(const std::string &filename, double support, double scale) {
    clusters.clear();
    planes.clear();
    cylinders.clear();
    std::ifstream is(filename);
    if (!is) {
        std::cerr << "file not found " << filename << std::endl;
        return false;
    }
    std::string line;
    size_t num_points = 0;
    getline_ignore_empty(is, line);
    {
        std::istringstream is_line(line);
        is_line >> num_points;
        if (is_line.fail()) {
            return false;
        }
    }
    cloud_->P.resize(num_points, 3);
    cloud_->N.resize(num_points, 3);
    for (size_t i = 0; i < num_points; i++) {
        if (!getline_ignore_empty(is, line)) {
            return false;
        }
        double x, y, z, nx, ny, nz, conf;
        std::istringstream is_line(line);
        is_line >> x >> y >> z >> nx >> ny >> nz >> conf;
        if (is_line.fail()) {
            return false;
        }
        cloud_->P.row(i) << x*scale, y*scale, z*scale;
        cloud_->N.row(i) << nx, ny, nz;
    }
    auto min_points = static_cast<size_t>(support * cloud_->P.rows());
    size_t num_primitives = 0;
    size_t num_planes = 0;
    getline_ignore_empty(is, line);
    {
        std::istringstream is_line(line);
        is_line >> num_primitives;
        if (is_line.fail()) {
            return false;
        }
    }
    for (size_t i=0; i<num_primitives; i++) {
        if (!getline_ignore_empty(is, line)) {
            return false;
        }
        std::string primitive_type;
        std::istringstream is_params(line);
        is_params >> primitive_type;
        if (is_params.fail()) {
            return false;
        }
        if (!getline_ignore_empty(is, line)) {
            return false;
        }
        std::istringstream is_indices(line);
        size_t ind;
        std::string lbl;
        std::vector<int> cluster;
        if (is_indices >> lbl && lbl == "points") {
            while (is_indices >> ind) {
                cluster.push_back(ind);
            }
        } else {
            return false;
        }
        if (primitive_type == "plane") {
            double nx, ny, nz, offset;
            is_params >> nx >> ny >> nz >> offset;
            if (is_params.fail()) {
                return false;
            }
            planes.emplace_back(Vector3d(nx, ny, nz), offset * scale);
            clusters.insert(clusters.begin() + num_planes, std::move(cluster));
            num_planes++;
        } else if (primitive_type == "cylinder") {
            double nx, ny, nz, px, py, pz, r;
            is_params >> nx >> ny >> nz >> px >> py >> pz >> r;
            if (is_params.fail()) {
                return false;
            }
            Vector3d normal(nx, ny, nz);
            Vector3d basepoint(px*scale, py*scale, pz*scale);
            double start = std::numeric_limits<double>::max();
            double end = std::numeric_limits<double>::lowest();
            for (int index : cluster) {
                Vector3d point3d = cloud_->P.row(index).transpose();
                double projection = normal.dot(point3d - basepoint);
                start = std::min(start, projection);
                end = std::max(end, projection);
            }
            cylinders.emplace_back(basepoint, normal, r, start, end);
            clusters.push_back(std::move(cluster));
        } else {
            std::cerr << "unrecognized primitive type: " << primitive_type << std::endl;
            return false;
        }
    }
    compute_bounds();
    size_t num_flipped = recompute_normals();
    size_t num_split = split_clusters(min_points);
    std::cout << "flipped " << num_flipped << " normals and split " << num_split << " clusters" << std::endl;
    compute_point_labels();
    return true;
}

Vector3d FeatureExtractor::minPt() const {
    return minPt_;
}

Vector3d FeatureExtractor::maxPt() const {
    return maxPt_;
}

void FeatureExtractor::setPointCloud(PointCloud3::Handle cloud) {
    cloud_ = std::move(cloud);
    windingNumberDirty_ = true;
    compute_bounds();
}
PointCloud3::Handle FeatureExtractor::getPointCloud() {
    return cloud_;
}


//void FeatureExtractor::load_primitives(const std::string &filename) {
    //TODO
//}
