//
// Created by James Noeckel on 12/10/19.
//

#pragma once

#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <opencv2/opencv.hpp>

struct Image {
    std::string image_name_;
    std::string depth_name_;
    int32_t camera_id_;

    /** transformation from world to camera space: rot_ * pt + trans_
     * camera frame is +z forward */
    Eigen::Quaterniond rot_;
    Eigen::Vector3d trans_;

    std::vector<Eigen::Vector2d> xys_;
    std::vector<int64_t> point3D_ids_;
    static std::unordered_map<int32_t, Image> parse_file(const std::string &filename, const std::string &image_path, const std::string &depth_path, double scale=1.0);

    Eigen::Vector3d origin() const;
    Eigen::Vector3d direction() const;

    void clearImages();

    /**
     * @return image in 8UC3 format
     */
    cv::Mat getImage(bool grayscale=false);

    /**
     * @return depth map in float32 format
     */
    cv::Mat getDepthGeometric();
    /**
     * @return X derivative image (scaled by 16)
     */
    cv::Mat getDerivativeX();
    /**
     * @return Y derivative image (scaled by 16)
     */
    cv::Mat getDerivativeY();

    std::string image_path_;
    std::string depth_path_;
    double scale_ = 1.0;
private:
    cv::Mat img_;
    cv::Mat derivative_x_;
    cv::Mat derivative_y_;
    cv::Mat depth_geom_;
    double depth_scale_ = 1.0;
    bool loaded_image_ = false;
    bool loaded_grayscale_ = false;
    bool computed_derivative_ = false;
    bool loaded_depth_geom_ = false;
    void computeDerivatives();
};

std::ostream &operator<<(std::ostream &o, const Image &im);

