//
// Created by James Noeckel on 3/26/20.
//

#include "mesh_io.h"

#include <igl/read_triangle_mesh.h>

bool load_mesh(const std::string &filename, Eigen::MatrixX3d &V, Eigen::MatrixX3i &F) {
    return igl::read_triangle_mesh(filename, V, F);
}