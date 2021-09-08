//
// Created by James Noeckel on 3/16/20.
//

#pragma once
#include "geometry/primitives3/BoundedPlane.h"
#include "geometry/primitives3/Cylinder.h"
#include "utils/typedefs.hpp"
#include "utils/settings.h"
#include "reconstruction/ReconstructionData.h"
#include "utils/sorted_data_structures.hpp"

#include <vector>
//#include <pcl/point_types.h>
//#include <pcl/point_cloud.h>
#include <random>
#include "geometry/WindingNumberData.h"
#include "geometry/primitives3/MultiRay3d.h"

enum {
    CONTOUR_WN_ID,
    CONTOUR_DENSITY_ID,
    CONTOUR_ERODED_ID,
    CONTOUR_CURVES_ID,
    BBOX_ID,
    CONVEX_HULL_ID
};

struct PointMembership {
    std::vector<std::vector<bool>> shape_membership;
    std::vector<int> depth_membership;
    std::map<std::pair<int, int>, std::pair<int, int>> joint_depth_membership;
    size_t count;
};

struct NeighborData {
    bool convex;
    MultiRay3d intersectionEdges;
};

struct FeatureExtractor {
    explicit FeatureExtractor(PointCloud3::Handle cloud, ReconstructionData::Handle reconstruction, std::mt19937 &random);
    explicit FeatureExtractor(ReconstructionData::Handle reconstruction, std::mt19937 &random);

    void setPointCloud(PointCloud3::Handle cloud);
    PointCloud3::Handle getPointCloud();

    //feature detection pipeline
    void compute_winding_number(int winding_number_stride, int k_n);
    Eigen::VectorXd query_winding_number(const Eigen::Ref<const Eigen::MatrixX3d> &Q) const;
    void detect_primitives(double threshold, double support, double probability, bool use_cylinders, double cluster_epsilon, double normal_threshold);
    void detect_bboxes();
    void detect_contours(double voxel_width, double threshold, double min_contour_hole_ratio=5, bool use_winding_number=true, int erosion=0, bool debug_vis=false);
//    int detect_curves(double voxel_width, double knot_curvature, int max_knots, double bezier_cost, double line_cost, double curve_weight);
//    void split_shapes(double image_derivative_threshold, double offset_tolerance);
    void detect_adjacencies(double adjacency_threshold, double norm_adjacency_threshold);
    void detect_parallel(double adjacency_threshold, double norm_parallel_threshold);
    /**
     * Populate data structures indicating which views can see which clusters: cluster_visibility, visible_clusters, and pruned_cluster_visibility
     * @param max_views_per_cluster maximum number of views in each list for pruned_cluster_visibility
     * @param threshold threshold to use to consider SFM points in a cluster, if using min_view_support
     * @param min_view_support Minimum number of SFM points supported by a given view for a cluster to be considered visible (default: ignore this requirement)
     */
    void detect_visibility(int max_views_per_cluster, double threshold=-1, int min_view_support=-1);
    /**
     *
     * @param thickness_unit smallest unit of depth
     * @param thickness_steps number of steps of size thickness_unit to consider
     * @param edge_threshold
     * @param edge_detection_threshold
     * @param num_candidates
     * @param spatial_discounting_factor s factor to use to weight derivative peaks by depth using the expression exp(-s * depth)
     * @param max_depth
     */
    void compute_depths(double thickness_unit, int thickness_steps, double edge_threshold, double edge_detection_threshold, double spatial_discounting_factor=0);
    bool filter_depths(int num_clusters, double min_eigenvalue, bool modify_existing=false, double minDepth=0);
    void extract_image_shapes(const Settings &settings, double threshold, const Eigen::Ref<const Eigen::MatrixX3d> &colors);

//    void conditional_part_connections(double adjacency_threshold, double norm_adjacency_threshold, double norm_parallel_threshold);

    //IO
    void export_globfit(const std::string &filename, int stride=1, double scale=1);
    //void load_primitives(const std::string &filename);

    /** import points and primitives from globfit */
    bool import_globfit(const std::string &filename, double support, double scale=1);

    ///**
    // * Given a point cloud, populate tables indicating when each point is contained in each part/part combination
    // */
    //std::vector<PointMembership> process_samples(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const std::vector<size_t> &indices, const std::vector<size_t> &sample_counts);
    /**
     * Given a labeling from analyze_samples(), compute the point membership table
     */
    /*std::vector<PointMembership> process_samples(const Eigen::Ref<const MatrixX3d> &cloud, const std::vector<size_t> &indices, const std::vector<size_t> &sample_counts, const std::vector<std::vector<int>> &labels);
    Eigen::MatrixX3d generate_samples_biased();
    Eigen::MatrixX3d generate_samples_random(size_t count);
    void analyze_samples(pcl::PointCloud<pcl::PointXYZ>::ConstPtr samples, std::vector<size_t> &indices, std::vector<size_t> &sample_counts, std::vector<std::vector<int>> &labels, int min_overlap=1, double contour_threshold=-1, int point_cloud_stride=1, int k=15);
    bool save_samples(const std::string &filename, const std::vector<PointMembership> &samples, int max_constraints=-1);
    //bool save_samples(const std::string &filename, pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const std::vector<size_t> &indices, const std::vector<size_t> &sample_counts);
    bool save_samples(const std::string &filename, pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, const std::vector<size_t> &indices, const std::vector<size_t> &sample_counts, const std::vector<std::vector<int>> &labels, int max_constraints=-1);*/

    Eigen::Vector3d minPt() const;
    Eigen::Vector3d maxPt() const;
    void setCurrentShape(int idx);
    /** groups -> points */
    std::vector<std::vector<int>> clusters;
    /** points -> groups (-1 if no group) */
    std::vector<int> point_labels;
    /** each element is (part index, segment from shape intersection used by detect_adjacencies) */
    std::vector<std::unordered_map<int, NeighborData>> adjacency;
    /** each element is (part index, approx. signed distance along normal) */
    std::vector<sorted_map<int, double>> parallel;
    /** each element is (part index, min_depth) */
//    std::vector<sorted_map<int, std::pair<int, Edge3d>>> conditional_connections;
    std::vector<BoundedPlane> planes;
    std::vector<Cylinder> cylinders;
    std::vector<std::vector<int32_t>> cluster_visibility;
    std::unordered_map<int32_t, std::vector<int>> visible_clusters;
    std::vector<std::vector<int32_t>> pruned_cluster_visibility;
    std::vector<double> depths;
    /** which plane (if any) is the other's depth constraint */
    std::vector<std::vector<size_t>> opposing_planes;

    //objects for visualizing intermediate steps
//    std::vector<std::vector<std::vector<Eigen::Vector3d>>> vis_segmentation_shapes; //from segmentation
    /** plane -> curveIndex -> vertex */
    std::vector<std::vector<std::vector<Eigen::Vector3d>>> vis_shapes;
//    std::vector<std::vector<std::vector<Eigen::Vector3d>>> vis_shapes_raw;
//    std::vector<std::vector<Edge3d>> cut_edges;
private:
    /**
     * Accumulate neighboring point densities along direction of base plane's normal
     * @param spacing spacing of bins
     * @param basePrimitive index of base plane
     * @param neighborPrimitive index of neighboring plane/cylinder
     * @param densities
     * @return
     */
    bool neighborDensity(double spacing, double margin, int basePrimitive, int neighborPrimitive, const MultiRay3d &intersectionEdge, std::vector<size_t> &densities, bool &convex);
//    void compute_joint_depths(const Eigen::Ref<const Eigen::MatrixX3d> &allpoints, std::vector<PointMembership> &table);
//    int get_min_depth_index(int part, const Eigen::Ref<const Eigen::Vector3d> &point);
    bool detect_adjacency(double adjacency_threshold, double norm_adjacency_threshold, int shape1, int shape2, MultiRay3d &intersectionEdges, bool &convex);
    void compute_bounds();
    size_t recompute_normals();
    size_t split_clusters(int min_support);
    void compute_point_labels();
    PointCloud3::Handle cloud_;
    ReconstructionData::Handle reconstruction_;
    Eigen::Vector3d minPt_, maxPt_;
    std::mt19937 &random_;
    //winding number attributes
    WindingNumberData::Handle windingNumberData_ = std::make_shared<WindingNumberData>();
    bool windingNumberDirty_ = true;
};

