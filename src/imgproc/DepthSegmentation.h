//
// Created by James Noeckel on 2/4/20.
//

#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "reconstruction/CameraIntrinsics.h"
#include "reconstruction/Image.h"
#include "geometry/primitives3/BoundedPlane.h"
#include "geometry/primitives3/Cylinder.h"
#define BACKGROUND_VALUE 255


class DepthSegmentation {
public:
    DepthSegmentation(const std::vector<BoundedPlane> &bboxes, const std::vector<Cylinder> &cylinders, const std::vector<int> &bbox_clusters, Image &image, const CameraIntrinsics &camera, PointCloud3::Handle cloud, float scale=1.0f, float correction_factor=1.0f, int colorspace=CV_BGR2Lab);
    void computeDepthSegmentation(float threshold);
    void outlierRemoval(int min_support);
    void energyMinimization(float segmentation_data_weight, float segmentation_smoothing_weight, float segmentation_penalty, float segmentation_sigma, int num_components, int segmentation_iters, int levels);
    /**
     * Reassign connected components based on how many pixels fall within their associated cluster shape's 2D projection.
     * The reassignment is based on which cluster shape's 2D projection covers the most pixels first, then which cluster has
     * the most neighboring pixels to the connected component being reassigned. The thresholds used to decide whether a
     * connected component needs to be reassigned must both be violated for reassignment to occur.
     * For example, a recall threshold of 0.9 means that even if less than the required number of pixels in a connected component
     * are valid, if it completely covers at least 90% of the pixels of the cluster, it is not reassigned.
     * To use only one threshold, set the other one greater than 1.
     * @param iterations Number of times to repeat the reassignment
     * @param precision_threshold minimum fraction of pixels in a connected component that need to lie inside the associated cluster
     * @param recall_threshold minimum fraction of total pixels of a cluster that need to be covered by each connected component
     * @param optimal_cluster_reassignment if true, first reassign as many connected components as possible using just the scores for each cluster
     * @param allow_background_conversion Allow connected components to be set to background if most pixels are neighboring it
     */
    void cleanSegmentation(int iterations=3, float precision_threshold=0.5f, float recall_threshold=0.9f, bool optimal_cluster_reassignment=true, bool allow_background_conversion=false);
    const std::vector<double> &scores();
    cv::Mat getSegmentation() const;
    bool isValid() const;
private:
    PointCloud3::Handle cloud_;
    cv::Mat segmentation_;
    cv::Mat depth_map_;
    cv::Mat image_;
    float scale_;
    float correction_factor_;
    Eigen::Vector3d trans_;
    Eigen::Quaterniond rot_;
    Eigen::Array2d principal_point_;
    Eigen::Array2d fdist_;
    std::reference_wrapper<const std::vector<BoundedPlane>> bboxes_;
    std::reference_wrapper<const std::vector<Cylinder>> cylinders_;
    std::reference_wrapper<const std::vector<int>> visible_clusters_;
    int width_, height_;

    bool dirty_ = true;
    std::vector<double> scores_;
    std::vector<int> planar_clusters_;
    std::vector<int> cylinder_clusters_;
    std::vector<bool> use_cylinder_exteriors_;
    std::unordered_map<int, int> bbox_indices_;
    std::unordered_map<int, int> cylinder_indices_;
    std::unordered_map<int, int> global_indices_;
};