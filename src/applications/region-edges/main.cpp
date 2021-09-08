#include <pcl/io/ply_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/common/centroid.h>

#include <Eigen/Dense>
#include <iostream>

#include <map>
#include <memory>

#include "utils/sorted_data_structures.hpp"

int main(int argc, char **argv)
{
  if (argc < 3) {
    std::cout << "usage: " << argv[0] << " data.ply output.ply [ k_normals [ k_region [ alph_max ] ] ]" << std::endl;
    return 1;
  }
  int k_n = 10;
  int k_r = 10;
  float alpha = 5 / 180.0f * (float) M_PI;
  try {
    if (argc >= 4) {
      k_n = std::stoi(argv[3]);
    }
  
    if (argc >= 5) {
      k_r = std::stoi(argv[4]);
    }
  
    if (argc >= 6) {
      alpha = std::stof(argv[5]) / 180.0f * (float) M_PI;
    }
  } catch (std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  const std::string path = argv[1];
  const std::string outpath = argv[2];
  pcl::PLYReader reader;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  reader.read(path, *cloud);
  std::cout << "size: " << cloud->points.size() << std::endl;
  
  //normal estimation
  pcl::search::Search<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod (tree);
  normal_estimator.setInputCloud (cloud);
  normal_estimator.setKSearch (k_n);
  normal_estimator.compute (*normals);
  
  // region growing
  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
  // reg.setMinClusterSize (5);
  // reg.setMaxClusterSize (1000000);
  reg.setSearchMethod (tree);
  reg.setNumberOfNeighbours (k_r);
  reg.setInputCloud (cloud);
  reg.setInputNormals (normals);
  reg.setSmoothnessThreshold (alpha);
  // reg.setCurvatureThreshold (1.0);

  std::vector <pcl::PointIndices> clusters;
  reg.extract (clusters);

  std::cout << "Number of clusters is equal to " << clusters.size () << std::endl;

  std::map<size_t, size_t> histogram;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored(new pcl::PointCloud<pcl::PointXYZRGB>);
  colored->is_dense = true;
  for (size_t i=0; i<clusters.size(); i++) {
    const unsigned char r = rand() % 255;
    const unsigned char g = rand() % 255;
    const unsigned char b = rand() % 255;
    for (size_t p=0; p<clusters[i].indices.size(); p++) {
      const pcl::PointXYZ &point = (*cloud)[clusters[i].indices[p]];
      pcl::PointXYZRGB newpoint;
      newpoint.x = point.x;
      newpoint.y = point.y;
      newpoint.z = point.z;
      newpoint.r = r;
      newpoint.g = g;
      newpoint.b = b;
      colored->push_back(newpoint);
    }
    if (histogram.find(clusters[i].indices.size()) == histogram.end()) {
      histogram[clusters[i].indices.size()] = 1;
    } else {
      histogram[clusters[i].indices.size()] += 1;
    }
  }
  for (const auto &pair : histogram) {
    std::cout << "clusters with size " << pair.first << ": " << pair.second << std::endl;
  }
  pcl::PLYWriter writer;
  writer.write(outpath, *colored, true, false);

  //build adjacency graph
  std::cout << "building adjacency graph" << std::endl;

  //build inverse cluster map
  std::unique_ptr<size_t[]> cluster_map = std::make_unique<size_t[]>(cloud->size());
  for (size_t c=0; c<clusters.size(); c++) {
    for (size_t p=0; p<clusters[c].indices.size(); p++) {
      cluster_map[clusters[c].indices[p]] = c;
    }
  }

  //find nodes with neighbors in another cluster
  std::cout << "finding neighboring clusters" << std::endl;
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(cloud);
  std::unique_ptr<size_t[]> cluster_adjacency_map = std::make_unique<size_t[]>(cloud->size());
  for (size_t p=0; p<cloud->size(); p++) {
    std::vector<int> ids;
    std::vector<float> dists;
    cluster_adjacency_map[p] = cluster_map[p];
    if (kdtree.nearestKSearch((*cloud)[p], k_r+1, ids, dists) > 0) { //TODO: Choose a better K
      for (int id : ids) {
	if (cluster_map[id] != cluster_map[p]) {
	  cluster_adjacency_map[p] = cluster_map[id];
	}
      }
    }
  }

  size_t count=0;
  for (size_t p=0; p<cloud->size(); p++) {
    if (cluster_adjacency_map[p] != cluster_map[p]) {
      count++;
    }
  }

  std::cout << "found " << count << " frontier vertices" << std::endl;

  //build graph
  std::cout << "building graph" << std::endl;
  std::vector<std::vector<size_t>> adjacency(clusters.size());
  for (size_t c=0; c<clusters.size(); c++) {
    for (size_t p=0; p<clusters[c].indices.size(); p++) {
      size_t c_adj = cluster_adjacency_map[clusters[c].indices[p]];
      if (c_adj != c) {
	insert_sorted_vector(adjacency[c], c_adj);
	insert_sorted_vector(adjacency[c_adj], c);
      }
    }
  }

  //get cluster summaries
  std::cout << "computing cluster statistics" << std::endl;
  pcl::PointCloud<pcl::PointXYZ>::Ptr centroids(new pcl::PointCloud<pcl::PointXYZ>);
  for (size_t c=0; c<clusters.size(); c++) {
    pcl::CentroidPoint<pcl::PointXYZ> centroid;
    for (size_t p=0; p<clusters[c].indices.size(); p++) {
      centroid.add((*cloud)[clusters[c].indices[p]]);
    }
    pcl::PointXYZ centroid_result;
    centroid.get(centroid_result);
    centroids->push_back(centroid_result);
  }

  //std::cout << "adding edges" << std::endl;
  //for (size_t c=0; c<clusters.size(); c++) {
  //  
  //}
  
  
  return 0;
}
