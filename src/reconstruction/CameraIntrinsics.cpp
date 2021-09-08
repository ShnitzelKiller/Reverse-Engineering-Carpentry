//
// Created by James Noeckel on 12/10/19.
//

#include "CameraIntrinsics.h"

#include "parsing.hpp"

#include <fstream>
#include <iostream>
#include <memory>

size_t num_params(int model_id) {
    switch (model_id) {
        case 0:
            return 3;
        case 1:
            return 4;
        case 2:
            return 4;
        case 3:
            return 5;
        case 4:
            return 8;
        case 5:
            return 8;
        case 6:
            return 12;
        case 7:
            return 5;
        case 8:
            return 4;
        case 9:
            return 5;
        case 10:
            return 12;
        default:
            return 0;
    }
}

std::unordered_map<int32_t, CameraIntrinsics> CameraIntrinsics::parse_file(const std::string &filename) {
    size_t size;
    std::unique_ptr<char[]> memblock = read_file(filename, size);

    const char *ptr = memblock.get();
    std::unordered_map<int32_t, CameraIntrinsics> cameras;
    if (!ptr) return cameras;
    uint64_t num_cameras;
    ptr = read_object(ptr, num_cameras);
    std::cout << "num cameras: " << num_cameras << std::endl;
    for (size_t i=0; i<num_cameras; i++) {
        CameraIntrinsics camera;
        int32_t camera_id;
        ptr = read_object(ptr, camera_id);
        ptr = read_object(ptr, camera.model_id_);
        ptr = read_object(ptr, camera.width_);
        ptr = read_object(ptr, camera.height_);
        size_t n_params = num_params(camera.model_id_);
        camera.params_.resize(n_params);
        ptr = read_object(ptr, camera.params_);
        std::cout << "camera " << camera_id << ": " << camera.width_ << "x" << camera.height_ << ", type " << camera.model_id_ << std::endl;
        std::cout << "params: ";
        for (size_t j=0; j<n_params; j++) {
            std::cout << camera.params_[j] << " ";
        }
        std::cout << std::endl;
        cameras[camera_id] = camera;
    }
    return cameras;
}
