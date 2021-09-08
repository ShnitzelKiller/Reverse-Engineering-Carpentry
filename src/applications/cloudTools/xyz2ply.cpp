//
// Created by James Noeckel on 3/13/20.
//

#include "reconstruction/point_cloud_io.h"
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/ply_io.h>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "usage: " << argv[0] << " <filename>.xyz <output>.ply" << std::endl;
        return 1;
    }
    std::string filename = argv[1];
    std::string outname = argv[2];
    PointCloud3::Handle cloud;
    {
        auto start_t = clock();
        if (!load_pointcloud(filename, cloud)) {
            return 1;
        }
        auto total_t = clock() - start_t;
        float time_sec = static_cast<float>(total_t) / CLOCKS_PER_SEC;
        std::cout << "loaded in " << time_sec << " seconds" << std::endl;
    }

    pcl::PLYWriter writer;
    pcl::PointCloud<pcl::PointNormal>::Ptr cloudpcl(new pcl::PointCloud<pcl::PointNormal>);
    cloudpcl->reserve(cloud->P.rows());
    for (size_t p=0; p<cloud->P.rows(); ++p) {
        pcl::PointNormal pt;
        pt.x = cloud->P(p, 0);
        pt.y = cloud->P(p, 1);
        pt.z = cloud->P(p, 2);
        pt.normal_x = cloud->N(p, 0);
        pt.normal_y = cloud->N(p, 1);
        pt.normal_z = cloud->N(p, 2);
        cloudpcl->push_back(pt);
    }
    writer.write(outname, *cloudpcl, true, true);

    return 0;
}