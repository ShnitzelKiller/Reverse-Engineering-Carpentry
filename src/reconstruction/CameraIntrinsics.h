//
// Created by James Noeckel on 12/10/19.
//

#pragma once
#include <vector>
#include <unordered_map>

struct CameraIntrinsics {
    int32_t model_id_;
    uint64_t width_, height_;
    std::vector<double> params_;
    static std::unordered_map<int32_t, CameraIntrinsics> parse_file(const std::string &filename);
};

