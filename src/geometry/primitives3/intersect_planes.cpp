//
// Created by James Noeckel on 9/4/20.
//

#include "intersect_planes.h"

using namespace Eigen;

void intersect_planes(const Ref<const RowVector3d> &n1, const Ref<const RowVector3d> &n2, double offset1, double offset2, Ref<Vector3d> o, Ref<Vector3d> d) {
    Matrix<double, 2, 3> M;
    M.row(0) = n1;
    M.row(1) = n2;
    Vector2d b(-offset1, -offset2);
    // use QR decomposition to find particular solution for ray origin
    o = M.colPivHouseholderQr().solve(b);
    d = M.row(0).transpose().cross(M.row(1).transpose());
}