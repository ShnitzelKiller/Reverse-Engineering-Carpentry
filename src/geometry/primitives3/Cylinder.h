//
// Created by James Noeckel on 2/11/20.
//

#pragma once
#include <Eigen/Dense>

class Cylinder {
public:
    Cylinder(const Eigen::Ref<const Eigen::Vector3d> &point, const Eigen::Ref<const Eigen::Vector3d> &dir, double radius, double start, double end);
    double start() const;
    double end() const;
    double radius() const;
    Eigen::Vector3d point() const;
    Eigen::Vector3d dir() const;
    Eigen::Vector3d normal(const Eigen::Ref<const Eigen::Vector3d> &point) const;
    bool contains3D(const Eigen::Ref<const Eigen::Vector3d> &point, double threshold, double margin=0) const;
    /**
     * Intersect 3D ray with this cylinder
     * @param ray_origin
     * @param ray_direction
     * @param t_start intersection enter distance
     * @param t_end intersection exit distance
     * @param margin margin around the bbox for detecting intersection
     * @param inside_start true if there is an intersection, and the entering intersection is inside the cylinder bounds
     * @param inside_end true if there is an intersection, and the exiting intersection is inside the cylinder bounds
     */
    void intersect3D(const Eigen::Ref<const Eigen::Vector3d> &ray_origin, const Eigen::Ref<const Eigen::Vector3d> &ray_direction, double &t_start, double &t_end, bool &inside_start, bool &inside_end, double margin= 0.0f) const;
private:
    double start_, end_, radius_;
    Eigen::Vector3d point_;
    Eigen::Vector3d dir_;
};

