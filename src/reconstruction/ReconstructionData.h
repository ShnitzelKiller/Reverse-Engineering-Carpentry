//
// Created by James Noeckel on 3/15/20.
//

#pragma once
#include "Points3D.h"
#include "Image.h"
#include "CameraIntrinsics.h"
#include <memory>

struct ReconstructionData {
    std::unordered_map<int32_t, Image> images;
    std::unordered_map<uint64_t, Point3D> points;
    std::unordered_map<int32_t, CameraIntrinsics> cameras;

    /** Align two reconstructions using the same set of input images.
     * find transform T such that
     * T * this = other
     * based on corresponding camera positions */
    void findAffineTransform(ReconstructionData &other, double &scale, Eigen::Quaterniond& rot, Eigen::Vector3d &trans);

    bool load_bundler_file(const std::string &filename, const std::string &depth_path="");
    bool load_colmap_reconstruction(const std::string &filename, const std::string &image_path, const std::string &depth_path);
    bool export_colmap_reconstruction(const std::string &filename) const;
    bool export_image_resolutions(const std::string &filename) const;
    bool export_rhino_camera(const std::string &filename) const;

    /** set the scale factor of loaded images (also affects result of project(), directionalDericative(), and initRay() accordingly)*/
    void setImageScale(double scale);

    /**
     * project world space points to image pixel coordinates (x, y)
     * @param world_points nx3 matrix of points
     * @param image_id which view to project to
     * @return nx2 matrix of pixel positions
     */
    Eigen::MatrixX2d project(const Eigen::Ref<const Eigen::MatrixX3d> &world_points, int32_t image_id) const;

    /**
     * d/dt(proj(x3d + n * t))
     * D[proj(x3d+n*t)] * n
     * @param loc location of 3d point
     * @param dir direction of motion
     * @param image_id which view to use for projection
     * @return
     */
    Eigen::RowVector2d directionalDerivative(const Eigen::Ref<const Eigen::Vector3d> &loc, const Eigen::Ref<const Eigen::Vector3d> &dir, int32_t image_id) const;
    /**
     * Initialize ray with origin at camera position and direction through the specified pixel in the image plane
     * @param i (scaled) pixel row (floating point)
     * @param j (scaled) pixel column (floating point)
     * @param image_id
     * @return ray direction
     */
    Eigen::Vector3d initRay(double i, double j, int32_t image_id) const;

    /**
     * Epipolar line direction in the plane of image 1 at the position (i, j)
     * @param i (scaled) pixel row in image 1 (floating point)
     * @param j (scaled) pixel column in image 1 (floating point)
     * @param image1_id
     * @param image2_id
     * @return
     */
    Eigen::Vector2d epipolar_line(double i, double j, int32_t image1_id, int32_t image2_id) const;

    /** (x, y) resolution */
    Eigen::Vector2d resolution(int32_t image_id) const;

    /** un-load all image data */
    void clearImages();

    /**
     * Map IDS from this reconstruction's image IDS to the other based on image filenames
     * @param other
     * @return
     */
    std::unordered_map<int32_t, int32_t> correspondingImages(const ReconstructionData &other);

    typedef std::shared_ptr<ReconstructionData> Handle;
};

