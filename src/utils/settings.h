#pragma once

#include <string>
#include <cmath>

struct Settings {
    float final_scale=20.0f;
    int random_seed=1234;
    /** larger values means finer detail, e.g. more primitives fit more precisely to smaller features */
    float master_resolution=300.0f;
    //float threshold = -1.0f;
    float normal_threshold = -1;//0.4;
    /** if greater than 1, merge coplanar points more aggressively, if less than 1, more islands */
    float clustering_factor = 0.5f;
    float constraint_factor = 3.0f;
    //float cluster_epsilon = -1.0f;
    /** probability of missing primitives in RANSAC (lower = higher quality) */
    float probability = 0.01f;
    /** percent of total points that need to be included in a primitive for it to be valid */
    float support = 0.25f;
    /** detect cylinders along with planes */
    int use_cylinders = 0;
    //float adjacency_threshold = -1;
    /** if greater than 1, consider more primitives adjacent, if less, fewer */
    float adjacency_factor = 2.0f;
    //float adjacency_edge_threshold = -1;
    ///** if greater than 1, generate intersection edges between more primitives, if less, fewer */
    //float adjacency_edge_factor = 2.0f;
    float alpha = -1;
    /** resolution of grid for finding cut paths and depths */
    float voxel_resolution = 100.0f;
    float max_contour_hole_ratio = 5.0f;
    int k_n = 15;
    int use_winding_number = 0;
    int winding_number_stride = 10;
    int use_geocuts = 0;
    /** threshold for marching squares */
    float contour_threshold = 0.25f;
    size_t min_view_support = -1;
    int max_views_per_cluster = 5;
    float max_view_angle = 3*static_cast<float>(M_PI)/8;
    float max_view_angle_cos;

    /** number of curves used to represent outer shapes */
    int outer_curve_count = 7;
    /** number of curves used to represent inner holes */
    int inner_curve_count = 5;
    /** minimum angle at curve interfaces (radians) */
    float min_knot_angle = 0.34f;
    /** maximum number of knots to consider in curve fitting algorithm (-1: no restriction) */
    int max_knots = 50;
    /** weight to apply to curves (higher than 1 means more costly) */
    float curve_weight = 1.0f;
    /** cost of extra line segments in shape fitting */
    float line_cost = 0.05f;
    /** cost of extra bezier curves in shape fitting */
    float curve_cost = 0.1f;

    float edge_detection_threshold = 0.0f;
    float max_thickness = -1.0f;
    int thickness_resolution = 100;
    /** total thickness candidates to initially keep per part */
    int thickness_candidates = 1;
    /** number of unique depths to allow in final model */
    int thickness_clusters = 5;
    /** factor by which derivative scores are divided for every diameter traversed by depth */
    float thickness_spatial_discount_factor = 100.0f;
    /** maximum dot product between two planes' normals to be considered adjacent */
    float norm_adjacency_threshold = 0.2f;
    ///** minimum absolute value of dot product between two planes' normals to be considered parallel (for finding otherOpposing pairs of parallel planes) */
    float norm_parallel_threshold = 0.9f;

    /** minimum dot product for aligning parts to be parallel */
    float align_parallel_threshold = 0.99f;

    /** number of initial samples to generate in the model bounding box */
    size_t sample_count = 1000000;
    /** minimum number of samples in an intersection term to be exported to the constraint solver (<0 means do not filter) */
    int min_samples = -1;
    /** maximum number of intersection terms exported to the constraints file */
    int max_constraints = -1;
    /** minimum number of parts that must potentially overlap with a point */
    int min_part_overlap = 1;
    std::string points_filename = "points.ply";
    std::string mesh_filename;
    std::string reconstruction_path;
    std::string image_path;
    std::string depth_path;
    std::string globfit_export;
    std::string globfit_import;
    float globfit_scale=1.0f;
    std::string result_path="solution";
    std::string curvefit_checkpoint;
    std::string segmentation_checkpoint;
    std::string connection_checkpoint;
    std::string selection_checkpoint;
    std::string oldresult_checkpoint;
    std::string connector_mesh;
    float connector_spacing=1;
    float connector_scale=1;
    float image_scale = 0.5f;
    int max_image_resolution=std::numeric_limits<int>::max();
    int min_pixels_per_voxel=5;
    float correction_factor = 1.0f;
    int patch_radius = 3;
    int use_segmentation = 0;
    int segmentation_max_views_per_cluster = 3;
    int segmentation_iters = 5;
    float segmentation_scale = 0.5f;
    float segmentation_data_weight = 1.0f;
    float segmentation_smoothing_weight = 50.0f;
    int segmentation_8way = false;
    float segmentation_penalty = 100.0f;
    float segmentation_sigma = 50.0f;
    float segmentation_precision = 1;
    int segmentation_levels = 1;
    int segmentation_gmm_components = 5;
    float segmentation_gmm_min_variance = 10.0f;
    int segmentation_min_pixel_support = 200;
    float segmentation_clean_precision_threshold = 0.5f;
    float segmentation_clean_recall_threshold = 0.9f;
    size_t population_size = 1;
    int generations = 2000;
    float max_overlap_ratio=0.5f;
    int alignment_stride=10;
    double alignment_tol=1e-6;
    int visualization_stride=1;
    int visualize=1;
    int debug_visualization=0;

    std::string settings_path;
    bool store_line(const std::string &key, const std::string &value);
    bool parse_file(const std::string &settings_filename);
    void setDerived();
};

std::ostream &operator<<(std::ostream &o, const Settings &s);