//
// Created by James Noeckel on 12/7/20.
//

#include "colorAtIndex.h"
#include "color_conversion.hpp"

Eigen::RowVector3d colorAtIndex(size_t index, size_t total) {
    double h = static_cast<double>(index) / total * 360;
    double intensity = 1;
    double rf, gf, bf;
    hsv2rgb<double>(h, 1.0, 1.0, rf, gf, bf);
    return {rf * intensity, gf * intensity, bf * intensity};
}