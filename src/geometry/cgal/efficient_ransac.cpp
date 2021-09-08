#include "geometry/cgal/geom.h"
#include <CGAL/Shape_detection/Efficient_RANSAC.h>

typedef CGAL::Shape_detection::Efficient_RANSAC_traits<Kernel, Pwn_vector, Point_map, Normal_map> Traits;
typedef CGAL::Shape_detection::Efficient_RANSAC<Traits> Efficient_ransac;
typedef CGAL::Shape_detection::Plane<Traits> Plane;
typedef CGAL::Shape_detection::Cylinder<Traits> Cylinder;

bool efficient_ransac(PointCloud3::Handle cloud,
                      std::vector<PlaneParam> &plane_params,
                      std::vector<CylinderParam> &cylinder_params,
                      std::vector<std::vector<int>> &clusters,
                      double epsilon, size_t min_points, double probability,
                      bool cylinders,
                      double cluster_epsilon,
                      double normal_threshold) {
    if (cloud->N.rows() < cloud->P.rows()) return false;
    Pwn_vector points;
    points.reserve(cloud->P.rows());
    for (int i = 0; i < cloud->P.rows(); i++) {
        Point_with_normal point(Kernel::Point_3(cloud->P(i, 0), cloud->P(i, 1), cloud->P(i, 2)),
                                Kernel::Vector_3(cloud->N(i, 0), cloud->N(i, 1), cloud->N(i, 2)));
        points.push_back(point);
    }

    Efficient_ransac ransac;
    ransac.set_input(points);
    ransac.add_shape_factory<Plane>();
    if (cylinders) {
        ransac.add_shape_factory<Cylinder>();
    }

    Efficient_ransac::Parameters parameters;
    parameters.epsilon = epsilon;
    if (min_points > 0) {
        parameters.min_points = min_points;
    }
    parameters.probability = probability;
    if (normal_threshold > 0) {
	    parameters.normal_threshold = normal_threshold;
    }
    if (cluster_epsilon > 0) {
	    parameters.cluster_epsilon = cluster_epsilon;
    }
    ransac.detect(parameters);
    Efficient_ransac::Shape_range shapes = ransac.shapes();
    Efficient_ransac::Shape_range::iterator it = shapes.begin();
    clusters = std::vector<std::vector<int>>(shapes.size());
    size_t plane_index = 0;
    size_t cylinder_index = ransac.planes().size();
    for (; it != shapes.end(); it++) {
        if (auto plane=dynamic_cast<Plane*>(it->get())) {
            std::vector<int> cluster;
            auto &inds = plane->indices_of_assigned_points();
            cluster.reserve(inds.size());
            std::copy(inds.begin(), inds.end(), std::back_inserter(cluster));

            Eigen::Vector3d norm(
                    plane->plane_normal().x(),
                    plane->plane_normal().y(),
                    plane->plane_normal().z()
                    );

            FT d = plane->d();

            // Flip normal of plane based on average point normal
            double sum_dot_product = 0;
            for (int i : cluster) {
                Eigen::Vector3d point_norm(points[i].second.x(), points[i].second.y(), points[i].second.z());
                sum_dot_product += point_norm.dot(norm);
            }
            if (sum_dot_product < 0) {
                norm = -norm;
                d = -d;
            }
            PlaneParam param(norm, d);
            plane_params.push_back(param);
            clusters[plane_index++] = std::move(cluster);
        } else if (auto cyl = dynamic_cast<Cylinder*>(it->get())) {
            std::vector<int> cluster;
            auto &inds = cyl->indices_of_assigned_points();
            cluster.reserve(inds.size());
            std::copy(inds.begin(), inds.end(), std::back_inserter(cluster));


            CylinderParam param(std::make_pair(Eigen::Vector3d(cyl->axis().point().x(), cyl->axis().point().y(), cyl->axis().point().z()),
                    Eigen::Vector3d(cyl->axis().to_vector().x(), cyl->axis().to_vector().y(), cyl->axis().to_vector().z())),
                    cyl->radius());
            cylinder_params.push_back(param);
            clusters[cylinder_index++] = std::move(cluster);
        }


    }

    //Reorder point cloud since underlying indices have been shuffled
    for (int i = 0; i < cloud->P.rows(); i++) {
        cloud->P(i, 0) = points[i].first.x();
        cloud->P(i, 1) = points[i].first.y();
        cloud->P(i, 2) = points[i].first.z();
        cloud->N(i, 0) = points[i].second.x();
        cloud->N(i, 1) = points[i].second.y();
        cloud->N(i, 2) = points[i].second.z();
    }
    return true;
}

