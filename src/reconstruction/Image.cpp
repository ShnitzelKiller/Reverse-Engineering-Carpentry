//
// Created by James Noeckel on 12/10/19.
//

#include "Image.h"
#include <fstream>
#include <memory>

#include "parsing.hpp"
#include <iostream>
#include <utility>

std::unordered_map<int32_t, Image> Image::parse_file(const std::string &filename, const std::string &image_path, const std::string &depth_path, double scale) {
    size_t size;
    std::unique_ptr<char[]> memblock = read_file(filename, size);
    const char* ptr = memblock.get();
    std::unordered_map<int32_t, Image> images;
    if (!ptr) return images;
    uint64_t num_images;
    ptr = read_object(ptr, num_images);
    for (size_t i=0; i<num_images; i++) {
        Image image;
        image.scale_ = scale;
        image.image_path_ = image_path;
        image.depth_path_ = depth_path;
        int32_t image_id;
        ptr = read_object(ptr, image_id);
        Eigen::Vector4d qvec;
        ptr = read_object(ptr, qvec);
        image.rot_ = Eigen::Quaterniond(qvec(0), qvec(1), qvec(2), qvec(3));
        ptr = read_object(ptr, image.trans_);
        ptr = read_object(ptr, image.camera_id_);
        ptr = read_object(ptr, image.image_name_);
        image.depth_name_ = image.image_name_ + ".geometric.bin";
        uint64_t num_points2D;
        ptr = read_object(ptr, num_points2D);
        image.xys_.resize(num_points2D);
        image.point3D_ids_.resize(num_points2D);
        for (size_t j=0; j<num_points2D; j++) {
            ptr = read_object(ptr, image.xys_[j]);
            ptr = read_object(ptr, image.point3D_ids_[j]);
        }
        std::cout << "Image " << image_id << ": " << image << std::endl;
        images.insert(std::make_pair(image_id, image));
    }
    return images;
}

Eigen::Vector3d Image::origin() const {
    return -(rot_.conjugate() * trans_);
}

Eigen::Vector3d Image::direction() const {
    return rot_.conjugate() * Eigen::Vector3d(0, 0, 1);
}

cv::Mat Image::getImage(bool grayscale) {
    if (!loaded_image_) {
        std::cout << "loading " << image_path_ + image_name_ << std::endl;
        img_ = cv::imread(image_path_ + image_name_);
        if (grayscale) {
            cv::cvtColor(img_, img_, CV_BGR2GRAY);
        }
        if (scale_ != 1) {
            cv::Mat temp;
            cv::resize(img_, temp, cv::Size(), scale_, scale_);
            img_ = temp;
        }
        loaded_grayscale_ = grayscale;
        loaded_image_ = true;
    } else if (grayscale != loaded_grayscale_) {
        loaded_image_ = false;
        getImage(grayscale);
    }
    return img_;
}

cv::Mat Image::getDepthGeometric() {
    if (!loaded_depth_geom_) {
        std::string ending = depth_name_.substr(depth_name_.rfind('.') + 1);
        std::transform(ending.begin(), ending.end(), ending.begin(),
                       [](unsigned char c) -> unsigned char { return std::tolower(c); });
        std::string filename = depth_path_ + depth_name_;
        if (ending == "exr") {
            depth_geom_ = cv::imread(filename, cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE);
            if (!depth_geom_.empty()) {
                loaded_depth_geom_ = true;
            } else {
                std::cerr << "failed to load " << filename << std::endl;
            }
        } else if (ending == "bin") {
            std::string num;
            size_t offset;
            int width, height, channels;
            std::ifstream is(filename);
            if (!is) {
                std::cerr << "file not found: \"" << filename << '"' << std::endl;
                return depth_geom_;
            }
            try {
                std::getline(is, num, '&');
                width = std::stoi(num);
                std::getline(is, num, '&');
                height = std::stoi(num);
                std::getline(is, num, '&');
                channels = std::stoi(num);
                offset = is.tellg();

            } catch (std::invalid_argument &exc) {
                std::cerr << "error reading header of " << filename << std::endl;
                return depth_geom_;
            }
            if (channels != 1) {
                std::cerr << "wrong number of channels in depth image" << std::endl;
                return depth_geom_;
            }
            size_t size;
            std::unique_ptr<char[]> memblock = read_file(filename, size, offset);
            if (size < width * height * 4) {
                std::cerr << "not enough image data for dimensions " << width << "x" << height << std::endl;
                return depth_geom_;
            }
            const char *ptr = memblock.get();
            depth_geom_ = cv::Mat(height, width, CV_32FC1);
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    ptr = read_object(ptr, depth_geom_.at<float>(i, j));
                    depth_geom_.at<float>(i, j) *= depth_scale_;
                }
            }
            loaded_depth_geom_ = true;
        } else {
            std::cerr << "unrecognized file type " << ending << " for " << filename << std::endl;
        }
    }
    return depth_geom_;
}

void Image::clearImages() {
    img_ = cv::Mat();
    depth_geom_ = cv::Mat();
    derivative_x_ = cv::Mat();
    derivative_y_ = cv::Mat();
    loaded_image_ = false;
    loaded_depth_geom_ = false;
    computed_derivative_ = false;
}

cv::Mat Image::getDerivativeX() {
    if (!computed_derivative_) {
        computeDerivatives();
    }
    return derivative_x_;
}

cv::Mat Image::getDerivativeY() {
    if (!computed_derivative_) {
        computeDerivatives();
    }
    return derivative_y_;
}

void Image::computeDerivatives() {
    cv::Mat img = getImage(true);
    cv::Scharr(img, derivative_x_, CV_16S, 1, 0);
    cv::Scharr(img, derivative_y_, CV_16S, 0, 1);
    computed_derivative_ = true;
}

std::ostream &operator<<(std::ostream &o, const Image &im) {
    o << "rot (" << im.rot_.w() << ", " << im.rot_.x() << ", " << im.rot_.y() << ", " << im.rot_.z() << "), trans (" << im.trans_.transpose() << "), camera ID: " << im.camera_id_ << ", image name: " << im.image_name_ << ", num points: " << im.point3D_ids_.size();
    return o;
}
