#include <pcl/io/ply_io.h>
#include <pcl/ModelCoefficients.h>

#include <Eigen/Dense>
#include <iostream>

#include "tools/segmentation.hpp"

int main(int argc, char **argv)
{
  if (argc < 2) {
    std::cout << "usage: " << argv[0] << " data.ply [ distance [ support ] ]" << std::endl;
    return 1;
  }
  float distance_threshold = 0.1f;
  if (argc >= 3) {
    try {
      distance_threshold = std::stof(argv[2]);
    } catch (std::exception &e) {
      std::cerr << e.what() << std::endl;
      return 1;
    }
  }
  int support = 4;
  if (argc >= 4) {
    try {
      support = std::stoi(argv[3]);
    } catch (std::exception &e) {
      std::cerr << e.what() << std::endl;
      return 1;
    }
    if (support < 0) {
      std::cerr << "invalid support" << std::endl;
      return 1;
    }
  }
  //load data
  std::string path =  argv[1];
  pcl::PLYReader reader;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  reader.read(path, *cloud);
  std::cout << "size: " << cloud->points.size() << std::endl;

  //segment data
  std::vector<pcl::PointIndices> inliers;
  std::vector<pcl::ModelCoefficients> params;
  segment(cloud, distance_threshold, (size_t) support, inliers, params);

  std::cout << "detected " << inliers.size() << " planes" << std::endl;
  /*for (const pcl::PointIndices &inlier : inliers) {
    std::cout << "inliers: " << inlier.indices.size() << std::endl;
  }*/
  
  //TODO: save as bvg
}
