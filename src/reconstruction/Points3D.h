//
// Created by James Noeckel on 12/11/19.
//

#pragma once


#include <cstdint>
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>

struct Point3D {
    Eigen::Vector3d xyz_;
    Eigen::Matrix<unsigned char, 3, 1> rgb_;
    double error_;
    std::vector<int32_t> image_ids_;
    std::vector<int32_t> point2D_idxs_;
    static std::unordered_map<uint64_t, Point3D> parse_file(const std::string &filename);
};

