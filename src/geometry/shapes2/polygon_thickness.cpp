//
// Created by James Noeckel on 1/5/21.
//

#include "polygon_thickness.h"
#include "Primitive.h"
//#include <iostream>

using namespace Eigen;

double polygon_thickness(const Ref<const MatrixX2d> &contour) {
    Polygon polygon(contour);
    double squaredMaxRadius = 0.0;
    for (size_t i=0; i<contour.rows(); ++i) {
        size_t iPrev = i > 0 ? i-1 : contour.rows()-1;
        size_t iNext = (i+1) % contour.rows();
        Vector2d tangent = (contour.row(iNext) - contour.row(iPrev)).transpose();
        //inside pointing
        Vector2d normal(tangent.y(), -tangent.x());
        normal.normalize();
        Ray2d ray(contour.row(i).transpose(), normal, 0);
        auto intersections = polygon.intersect(ray);
//        std::cout << "intersections: " << intersections.size() << std::endl;
        double t = 0;
        for (const auto &intersection : intersections) {
            if (!intersection.entering) {
//                std::cout << "found exiting" << std::endl;
                t = intersection.t;
                break;
            }
        }
        double radius = t * 0.5;
//        std::cout << "radius: " << radius << std::endl;
        Vector2d midpoint = contour.row(i).transpose() + radius * normal;
        double squaredMinRadius = radius * radius;
        for (size_t j=0; j<contour.rows(); ++j) {
            squaredMinRadius = std::min(squaredMinRadius, (contour.row(j).transpose() - midpoint).squaredNorm());
        }
        squaredMaxRadius = std::max(squaredMaxRadius, 2 * squaredMinRadius);
    }
    return 2 * std::sqrt(squaredMaxRadius);
}