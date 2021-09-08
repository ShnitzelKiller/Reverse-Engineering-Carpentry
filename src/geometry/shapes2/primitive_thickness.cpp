//
// Created by James Noeckel on 12/11/20.
//

#include "primitive_thickness.h"
//#include <iostream>

using namespace Eigen;

double primitive_thickness(const Primitive &primitive, const Edge2d &edge, double sample_spacing, double& meanDistance) {
    Vector2d dir(edge.second - edge.first);
    double len = dir.norm();
    dir /= len;
    Vector2d n = Vector2d(-dir.y(), dir.x()); //left-facing normal to shoot a ray inside
    int steps = static_cast<int>(std::round(len/sample_spacing));
    Vector2d increment = dir * sample_spacing;
    double meanThickness = 0.0;
    meanDistance = 0.0;
    int numHits = 0;
    if (steps == 0) {
        steps = 1;
        increment = dir * len;
    }
    Ray2d currRay(edge.first + 0.5 * increment, n, 0);
//    std::cout << steps << " steps, starting at " << currRay.o << std::endl;
    for (int i=0; i<steps; ++i) {
        const auto intersections = primitive.intersect(currRay);
//        std::cout << "intersection count : " << intersections.size() << std::endl;
//        int ind=0;
        double lastT = 0;
        for (const auto &intersection : intersections) {
//            std::cout << (intersection.entering ? "entering" : "exiting") << " intersection at " << intersection.t << "(intersection " << ind << ')' << std::endl;
            if (intersection.entering) {
                lastT = intersection.t;
            } else {
                meanThickness += intersection.t - lastT;
                break;
            }
//            ++ind;
        }
        if (!intersections.empty()) {
            ++numHits;
            meanDistance += lastT;
        }
        currRay.o += increment;
    }
    meanThickness /= steps;
    meanDistance = numHits == 0 ? 0 : meanDistance / numHits;
    return meanThickness;
}