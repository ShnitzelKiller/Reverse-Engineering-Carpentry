//
// Created by James Noeckel on 2/11/20.
//

#include "Cylinder.h"

using namespace Eigen;

Cylinder::Cylinder(const Ref<const Vector3d> &point, const Ref<const Vector3d> &dir,
                   double radius, double start, double end) : point_(point), dir_(dir.normalized()), radius_(radius), start_(start), end_(end) {

}

double Cylinder::start() const {
    return start_;
}
double Cylinder::end() const {
    return end_;
}
double Cylinder::radius() const {
    return radius_;
}
Vector3d Cylinder::point() const {
    return point_;
}
Vector3d Cylinder::dir() const {
    return dir_;
}

bool Cylinder::contains3D(const Ref<const Vector3d> &point, double threshold, double margin) const {
    Vector3d disp = point-point_;
    double coord = dir_.dot(disp);
    double rad = (disp - dir_ * coord).norm();
    return std::fabs(rad - radius_) <= threshold && coord >= start_-margin && coord <= end_+margin;
}

Vector3d Cylinder::normal(const Ref<const Vector3d> &point) const {
    Vector3d disp = point-point_;
    double coord = dir_.dot(disp);
    return (disp - dir_ * coord).normalized();
}

void Cylinder::intersect3D(const Ref<const Vector3d> &ray_origin, const Ref<const Vector3d> &ray_direction, double &t_start, double &t_end,
                           bool &inside_start,
                           bool &inside_end,
                           double margin) const {
    Vector3d oc = ray_origin-point_;
    double card = dir_.dot(ray_direction);
    double caoc = dir_.dot(oc);
    double a = 1.0f - card * card;
    double b = oc.dot(ray_direction) - caoc * card;
    double c = oc.dot(oc) - caoc * caoc - radius_ * radius_;
    double h = b*b-a*c;
    if (h < 0) {
        inside_start = false;
        inside_end = false;
        return;
    }
    h = sqrt(h);
    t_start = (-b - h)/a;
    t_end = (-b + h)/a;
    Vector3d projected_pt1 = oc + t_start * ray_direction;
    Vector3d projected_pt2 = oc + t_end * ray_direction;
    double coord1 = dir_.dot(projected_pt1);
    inside_start = coord1 >= start_-margin && coord1 <= end_+margin;
    double coord2 = dir_.dot(projected_pt2);
    inside_end = coord2 >= start_-margin && coord2 <= end_+margin;
}
