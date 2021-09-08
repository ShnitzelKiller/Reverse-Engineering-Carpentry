//
// Created by James Noeckel on 10/21/20.
//

#pragma once
#include "geometry/primitives3/SurfaceCompletion.h"
#include <Eigen/Dense>
#include "utils/visualization.hpp"
#include <igl/copyleft/marching_cubes.h>

enum {
    VIS_SURFACE,
    VIS_INSIDE,
    VIS_OUTSIDE
};

void displaySegmentation(SurfaceCompletion &surf, Visualizer &vis, int mode=VIS_SURFACE, bool use_distfield=true);