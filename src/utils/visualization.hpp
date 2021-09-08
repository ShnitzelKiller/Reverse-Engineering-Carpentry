//
// Created by James Noeckel on 11/19/19.
//

#pragma once
#include <Eigen/Dense>
#include "typedefs.hpp"
#include "geometry/primitives3/BoundedPlane.h"
#include "geometry/primitives3/Cylinder.h"
#include <memory>

//void visualize_points(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, float point_size=1);
//void visualize_reconstruction(pcl::PointCloud<pcl::PointNormal>::ConstPtr cloud, pcl::PointCloud<pcl::PointXYZ>::ConstPtr sparsecloud, std::vector<Edge> &point_directions, Eigen::RowVector3d color=Eigen::RowVector3d(0, 1, 1));

class Visualizer {
public:
    explicit Visualizer(float point_size=1, int stride=1);
    ~Visualizer();
    void align_camera(const Eigen::Ref<const Eigen::MatrixX3d> &points);
    void visualize_points(const PointCloud3& cloud);
    void visualize_sample_points(const PointCloud3& cloud, const Eigen::Ref<const Eigen::MatrixX3d> &colors = Eigen::RowVector3d(1, 1, 1));
    void visualize_sample_points(const PointCloud3& cloud, std::vector<size_t> &indices, const Eigen::Ref<const Eigen::MatrixX3d> &colors = Eigen::RowVector3d(1, 1, 1));
    void visualize_clusters(const PointCloud3& cloud, const std::vector<std::vector<int>> &clusters);
    void visualize_primitives(const std::vector<BoundedPlane> &bboxes, const std::vector<Cylinder> &cylinders);
    void visualize_edges(const std::vector<Edge3d> &edges, const std::vector<Eigen::Vector3d> &color);
    void visualize_shapes(const std::vector<std::vector<std::vector<Eigen::Vector3d>>> &contours);
    void visualize_shapes(const std::vector<std::vector<Edge3d>> &contours);
    void visualize_mesh(const Eigen::Ref<const Eigen::MatrixX3d> &V, const Eigen::Ref<const Eigen::MatrixX3i> &F);
    void clear_mesh();
    void launch();
    Eigen::Vector3d color(size_t i) const;
    Eigen::MatrixX3d colors() const;
    void update_colors(size_t size);
private:
    std::unique_ptr<class VisualizerImpl> impl_;
    int stride_;
};