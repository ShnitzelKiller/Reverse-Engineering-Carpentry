//
// Created by James Noeckel on 2/13/20.
//

#include "geom.h"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/convex_hull_2.h>
#include <CGAL/Convex_hull_traits_adapter_2.h>
#include <CGAL/property_map.h>
#include <vector>
#include <numeric>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_2 Point_2;
typedef CGAL::Convex_hull_traits_adapter_2<K,
        CGAL::Pointer_property_map<Point_2>::type > Convex_hull_traits_2;

void convex_hull(const Eigen::Ref<const Eigen::MatrixX2d> &shape, Eigen::MatrixX2d &hull) {
    std::vector<Point_2> points;
    points.reserve(shape.rows());
    for (size_t i=0; i<shape.rows(); i++) {
        Point_2 point(shape(i, 0), shape(i, 1));
        points.push_back(point);
    }
    std::vector<std::size_t> indices(points.size()), out;
    std::iota(indices.begin(), indices.end(),0);
    CGAL::convex_hull_2(indices.begin(), indices.end(), std::back_inserter(out),
                        Convex_hull_traits_2(CGAL::make_property_map(points)));

    hull.resize(out.size(), 2);
    for (size_t i=0; i<out.size(); ++i) {
        size_t ind = out[i];
        hull.row(i) = shape.row(ind);
    }
}