//
// Created by James Noeckel on 7/22/20.
//

#include "Segment2d.h"

using namespace Eigen;

Segment2d::Segment2d(Vector2d a, Vector2d b) : a_(std::move(a)), b_(std::move(b)) {

}

bool Segment2d::intersect(const Ray2d &other, double &t, double &t2, bool &entering) {
    Vector2d n(other.d.y(), -other.d.x()); //right facing normal of the ray;
    double h = n.dot(other.o); //depends only on ray
    double ah = n.dot(a_);
    double bh = n.dot(b_);
    entering = ah < bh; //assuming a counter-clockwise boundary, segment points should appear left-to-right if entering
    if ((ah <= h && h < bh) || (bh <= h && h < ah)) {
        t2 = (h-ah)/(bh-ah);
        double ta = (a_-other.o).dot(other.d);
        double tb = (b_-other.o).dot(other.d);
        t = (1-t2) * ta + t2 * tb;
        return t >= other.start && t < other.end;
    }
    return false;
}