//
// Created by James Noeckel on 3/13/20.
//

#include "point_cloud_io.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/property_map.h>
#include <CGAL/IO/read_xyz_points.h>
#include <CGAL/IO/read_ply_points.h>
#include <utility> // defines std::pair
#include <vector>
#include <fstream>
#include <iostream>

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point;
typedef Kernel::Vector_3 Vector;
typedef std::pair<Point, Vector> Pwn;

bool load_pointcloud(const std::string &filename, PointCloud3::Handle &cloud) {
    std::string ending = filename.substr(filename.rfind('.') + 1);
    std::transform(ending.begin(), ending.end(), ending.begin(),
                   [](unsigned char c) -> unsigned char { return std::tolower(c); });
    int type = -1;
    if (ending == "xyz") {
        type = 0;
    } else if (ending == "ply") {
        type = 1;
    } else {
        std::cerr << "Invalid filetype " << ending << std::endl;
        return false;
    }
    std::vector<Pwn> points;
    std::ifstream in(filename);
    if (!in ||
        (type == 0 ? !CGAL::read_xyz_points(
                in, std::back_inserter(points),
                CGAL::parameters::point_map(CGAL::First_of_pair_property_map<Pwn>()).
                        normal_map(CGAL::Second_of_pair_property_map<Pwn>()))
                   : !CGAL::read_ply_points(
                        in, std::back_inserter(points),
                        CGAL::parameters::point_map(CGAL::First_of_pair_property_map<Pwn>()).
                                normal_map(CGAL::Second_of_pair_property_map<Pwn>())
                ))) {
        std::cerr << "Error: cannot read file " << filename << std::endl;
        return false;
    }
    cloud = std::make_shared<PointCloud3>();
    cloud->P.resize(points.size(), 3);
    cloud->N.resize(points.size(), 3);
    for (size_t i=0; i<points.size(); ++i) {
        Pwn &point = points[i];
        cloud->P(i, 0) = point.first.x();
        cloud->P(i, 1) = point.first.y();
        cloud->P(i, 2) = point.first.z();
        cloud->N(i, 0) = point.second.x();
        cloud->N(i, 1) = point.second.y();
        cloud->N(i, 2) = point.second.z();
    }
    return true;
}

