//
// Created by James Noeckel on 12/11/19.
//

#include <fstream>
#include <iostream>
#include "Points3D.h"
#include "parsing.hpp"

std::unordered_map<uint64_t, Point3D> Point3D::parse_file(const std::string &filename) {
    size_t size;
    std::unique_ptr<char[]> memblock = read_file(filename, size);
    const char *ptr = memblock.get();
    std::unordered_map<uint64_t, Point3D> points;
    if (!ptr) return points;
    uint64_t num_points;
    ptr = read_object(ptr, num_points);
    std::cout << num_points << " point3D in reconstruction" << std::endl;
    for (size_t i=0; i<num_points; i++) {
        Point3D point;
        uint64_t point3D_id;
        ptr = read_object(ptr, point3D_id);
        ptr = read_object(ptr, point.xyz_);
        ptr = read_object(ptr, point.rgb_);
        ptr = read_object(ptr, point.error_);
        size_t track_length;
        ptr = read_object(ptr, track_length);
        //std::cout << "point " << point.point3D_id_ << ": trans " << point.xyz_.transpose() << ", rgb " << point.rgb_.transpose() << ", track length " << track_length << std::endl;
        std::vector<int32_t> image_ids(track_length);
        std::vector<int32_t> point2D_idxs(track_length);
        for (size_t j=0; j<track_length; j++) {
            ptr = read_object(ptr, image_ids[j]);
            ptr = read_object(ptr, point2D_idxs[j]);
        }
        std::vector<int> indices(track_length);
        for (size_t j=0; j<track_length; j++) {
            indices[j] = j;
        }
        //jointly sort by image id for easy lookup
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {return image_ids[a] < image_ids[b];});
        point.image_ids_.resize(track_length);
        point.point2D_idxs_.resize(track_length);
        for (size_t j=0; j<track_length; j++) {
            point.image_ids_[j] = image_ids[indices[j]];
            point.point2D_idxs_[j] = point2D_idxs[indices[j]];
        }
        points[point3D_id] = point;
    }
    return points;
}
