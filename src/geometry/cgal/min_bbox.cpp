#include "geom.h"


#include <CGAL/min_quadrilateral_2.h>
#include <iostream>
#include <CGAL/ch_graham_andrew.h>


void min_bbox(const Eigen::Ref<const Eigen::MatrixX2d> &cloud, Eigen::Ref<Eigen::Matrix<double, 4, 2>> bbox)
{
  std::vector<Point_2> points;
  points.reserve(cloud.rows());
  for (size_t i=0; i<cloud.rows(); i++) {
    Point_2 point(cloud(i, 0), cloud(i, 1));
    points.push_back(point);
  }
  std::vector<Point_2> hull_points;
  CGAL::ch_graham_andrew(points.begin(), points.end(), std::back_inserter(hull_points));
  
  Polygon_2 p_m;
  CGAL::min_rectangle_2(hull_points.begin(), hull_points.end(), std::back_inserter(p_m));
  for (int i=0; i<4; ++i) {
      bbox(i, 0) = p_m[i].x();
      bbox(i, 1) = p_m[i].y();
  }
}