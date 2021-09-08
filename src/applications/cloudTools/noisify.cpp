//
// Created by James Noeckel on 7/9/20.
//

#include <pcl/io/ply_io.h>
#include "math/NormalRandomVariable.h"

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cout << "usage: " << argv[0] << " <filename>.ply stdev <output>.ply" << std::endl;
        return 1;
    }
    std::string filename = argv[1];
    double stdev = std::stod(argv[2]);
    std::string outname = argv[3];


    pcl::PointCloud<pcl::PointNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointNormal>);
    pcl::io::loadPLYFile(filename, *cloud);
    Eigen::Matrix3d covar = Eigen::Matrix3d::Identity() * (stdev * stdev);
    NormalRandomVariable var(covar);
    for (size_t i=0; i<cloud->size(); i++) {
        Eigen::Vector3d displacement = var();
        (*cloud)[i].getVector3fMap() += displacement.cast<float>();
    }

    pcl::PLYWriter writer;
    writer.write(outname, *cloud, true, true);

    return 0;
}