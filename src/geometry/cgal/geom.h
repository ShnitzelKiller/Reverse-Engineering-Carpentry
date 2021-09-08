//
// Created by James Noeckel on 8/19/20.
//

#pragma once
#include "utils/typedefs.hpp"
#include <CGAL/property_map.h>
#include <CGAL/Point_with_normal_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_2.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::FT                                           FT;

//3D
typedef std::pair<Kernel::Point_3, Kernel::Vector_3> Point_with_normal;
typedef std::vector<Point_with_normal> Pwn_vector;
typedef CGAL::First_of_pair_property_map<Point_with_normal> Point_map;
typedef CGAL::Second_of_pair_property_map<Point_with_normal> Normal_map;

//2D
typedef Kernel::Point_2                           Point_2;
typedef Kernel::Line_2                            Line_2;
typedef Kernel::Segment_2                         Segment;
typedef CGAL::Polygon_2<Kernel>                   Polygon_2;

/** ============================= 2D ================================== */

/**
 * @param points Nx2 matrix of points
 * @param hull Nx2 output matrix of convex hull points
 */
void convex_hull(const Eigen::Ref<const Eigen::MatrixX2d> &points, Eigen::MatrixX2d &hull);

void alpha_shapes(const Eigen::Ref<const Eigen::MatrixX2d> &shape, std::vector<std::vector<Eigen::Vector2d>> &edgepoints, std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> &raw_edges, float alpha);

/**
 * @param points Nx2 matrix of points
 * @param bbox oriented bounding box as 4x2 matrix of 2D points
 */
void min_bbox(const Eigen::Ref<const Eigen::MatrixX2d> &points, Eigen::Ref<Eigen::Matrix<double, 4, 2>> bbox);

/**
 * @param poly multiply connected polygon (counter-clockwise orientation)
 * @param facets output facets contained in the polygon
 */
void polygon_triangulation(const std::vector<std::vector<Eigen::Vector2d>> &poly, std::vector<Eigen::Vector3i> &facets);

/** ============================= 3D ================================== */

/**
 *
 * @param cloud input point cloud
 * @param plane_params output plane params
 * @param cylinder_params output cylinder params
 * @param clusters vector of vectors of indices, with the plane clusters first; clusters[plane_params.size() + i] -> cylinder_params[i]
 * @param epsilon distance to consider components connected
 * @param support minimum number of points in a cluster
 * @param probability probability of missing an optimal primitive (lower is better, but slower)
 * @return success
 */
bool efficient_ransac(PointCloud3::Handle cloud, std::vector<PlaneParam> &plane_params, std::vector<CylinderParam> &cylinder_params, std::vector<std::vector<int>> &clusters, double epsilon, size_t min_points, double probability, bool cylinders=true, double cluster_epsilon=-1.0f, double normal_threshold=-1.0f);