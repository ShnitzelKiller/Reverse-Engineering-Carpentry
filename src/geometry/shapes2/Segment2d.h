//
// Created by James Noeckel on 7/22/20.
//

#pragma once

#include <Eigen/Dense>
#include "Ray2d.h"

/**
 * A segment [a, b)
 */
class Segment2d {
public:
    Segment2d(Eigen::Vector2d a, Eigen::Vector2d b);
    /**
     * Intersect ray with this segment, ensuring the numerically correct number of intersections for a connected chain
     * of segments. For a closed connected chain, there will always be even number of crossings on a ray
     * with start and end containing the max number of intersections.
     * @param other
     * @param t distance along ray
     * @param t2 lerp position along segment (0-1)
     * @param entering whether the ray is "entering" a shape comprised of counter-clockwise segments like this one
     * @return true if intersection is in the range of the ray's [start, end)
     */
    bool intersect (const Ray2d &other, double &t, double &t2, bool &entering);
    Eigen::Vector2d a_, b_;
};

