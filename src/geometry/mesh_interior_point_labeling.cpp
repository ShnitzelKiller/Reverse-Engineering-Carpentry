//
// Created by James Noeckel on 3/26/20.
//

#include "mesh_interior_point_labeling.h"
#include <igl/AABB.h>

std::vector<bool> mesh_interior_point_labeling(const Eigen::MatrixX3d &points, const Eigen::MatrixX3d &V, const Eigen::MatrixX3i &F) {
    std::vector<bool> labels(points.rows());
    igl::AABB<Eigen::MatrixX3d, 3> aabb;
    aabb.init(V, F);
    for (int i=0; i<points.rows(); i++) {
        std::vector<igl::Hit> hits;
        bool intersected = aabb.intersect_ray(V, F, points.row(i), Eigen::RowVector3d(0, 0, 1), hits);
        if (!hits.empty()) {
            std::sort(hits.begin(), hits.end(), [](igl::Hit a, igl::Hit b) {return a.t < b.t;});
            int num_distinct_hits = 0;
            double last_hit = -1000.0;
            for (auto &hit : hits) {
                if (hit.t != last_hit) {
                    num_distinct_hits++;
                    last_hit = hit.t;
                }
            }
            labels[i] = (num_distinct_hits % 2) == 1;
        } else {
            labels[i] = false;
        }
    }
    return labels;
}