//
// Created by James Noeckel on 12/11/20.
//

#pragma once
#include "Primitive.h"

/**
 * Measure the average thickness of the primitive "under" the specified edge, assuming the edge follows the boundary counter-clockwise
 * @param primitive
 * @param edge
 * @param sample_spacing spacing between sample rays to average thickness
 * @param meanDistance output mean distance to the primitive (gap)
 * @return average thickness
 */
double primitive_thickness(const Primitive &primitive, const Edge2d &edge, double sample_spacing, double &meanDistance);
