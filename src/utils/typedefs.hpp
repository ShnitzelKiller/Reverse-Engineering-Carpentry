#pragma once
#include <Eigen/Dense>
#include <memory>

typedef std::pair<Eigen::Vector3d, double> PlaneParam; //normal, offset
typedef std::pair<std::pair<Eigen::Vector3d, Eigen::Vector3d>, double> CylinderParam; //(point, direction), radius
typedef std::pair<Eigen::Vector3d, Eigen::Vector3d> Edge3d;
typedef std::pair<Eigen::Vector2d, Eigen::Vector2d> Edge2d;

struct PointCloud3 {
    Eigen::MatrixX3d P;
    Eigen::MatrixX3d N;
    typedef std::shared_ptr<PointCloud3> Handle;
};

struct PointCloud2 {
    Eigen::MatrixX2d P;
    typedef std::shared_ptr<PointCloud2> Handle;
};
