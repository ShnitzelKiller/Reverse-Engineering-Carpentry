#include "geometry/mesh_interior_point_labeling.h"

#include "utils/settings.h"
#include "utils/visualization.hpp"

#include "reconstruction/point_cloud_io.h"
#include "reconstruction/ReconstructionData.h"
#include "construction/FeatureExtractor.h"
#include "reconstruction/Solution.h"
#include "reconstruction/mesh_io.h"

#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "usage: " << argv[0] << " <config>.txt" << std::endl;
        return 1;
    }

    Settings settings;
    if (!settings.parse_file(argv[1])) {
        return 1;
    }

    std::cout << "============= settings =============\n" << settings << "============================\n\n";

    std::srand(settings.random_seed);
    std::mt19937 random_generator(settings.random_seed);
    // load sparse reconstruction data
    ReconstructionData::Handle reconstruction(new ReconstructionData);
    if (!settings.reconstruction_path.empty()) {
        if (settings.reconstruction_path.rfind(".out") != std::string::npos) {
            if (!reconstruction->load_bundler_file(settings.reconstruction_path, settings.depth_path)) {
                std::cerr << "failed to load bundler file " << settings.reconstruction_path << std::endl;
                return 1;
            }
        } else {
            if (!reconstruction->load_colmap_reconstruction(settings.reconstruction_path, settings.image_path,
                                                           settings.depth_path)) {
                std::cerr << "failed to load reconstruction in path " << settings.reconstruction_path << std::endl;
                return 1;
            }
        }
    }
    FeatureExtractor features(reconstruction, random_generator);
    Visualizer visualizer(2, settings.visualization_stride);

    if (settings.globfit_import.empty()) {
        PointCloud3::Handle cloud;
        std::cout << "loading point cloud..." << std::endl;
        auto start_t = clock();
        if (!load_pointcloud(settings.points_filename, cloud)) {
            return 1;
        }
        auto total_t = clock() - start_t;
        float time_sec = static_cast<float>(total_t) / CLOCKS_PER_SEC;
        std::cout << "loaded " << cloud->P.rows() << " points in " << time_sec << " seconds" << std::endl;
        features.setPointCloud(std::move(cloud));
    } else {
        std::cout << "loading globfit..." << std::endl;
        auto start_t = clock();
        if (!features.import_globfit(settings.globfit_import)) {
            std::cerr << "failed to import globfit" << std::endl;
            return 1;
        }
        auto total_t = clock() - start_t;
        float time_sec = static_cast<float>(total_t) / CLOCKS_PER_SEC;
        std::cout << "imported " << features.getPointCloud()->P.rows() << " points, " << features.planes.size() << " planes and " << features.cylinders.size()
                  << " cylinders in " << time_sec << " seconds" << std::endl;
    }

    Eigen::Vector3d bbox_min = features.minPt();
    Eigen::Vector3d bbox_max = features.maxPt();

    double diameter = (bbox_max-bbox_min).maxCoeff();
    double threshold = diameter/settings.master_resolution;
    double voxel_width = diameter / settings.voxel_resolution;
    double adjacency_threshold = voxel_width * settings.adjacency_factor;
    double bezier_cost = settings.curve_cost * threshold * threshold;
    double line_cost = settings.line_cost * threshold * threshold;
    double min_curvature = settings.min_knot_angle / diameter;
    std::cout << "bezier and line cost: " << bezier_cost << ", " << line_cost << std::endl;
    //double adjacency_edge_threshold = threshold * settings.adjacency_edge_factor;
    double cluster_epsilon = settings.clustering_factor < 0 ? settings.clustering_factor : threshold * settings.clustering_factor;
    std::cout << "bounding box: min " << bbox_min.transpose() << ", max: " << bbox_max.transpose() << " (diameter " << diameter << ')' << std::endl;

    //SEGMENTATION
    if (settings.globfit_import.empty())
    {
        auto start_t = clock();
        std::cout << "detecting primitives... ";
        features.detect_primitives(threshold, settings.support / 100, settings.probability,
                                   settings.use_cylinders, cluster_epsilon, settings.normal_threshold);
        auto total_t = clock() - start_t;
        float time_sec = static_cast<float>(total_t) / CLOCKS_PER_SEC;
        std::cout << "found " << features.planes.size() << " planes and " << features.cylinders.size()
                  << " cylinders in " << time_sec << " seconds" << std::endl;
    }
    if (!settings.globfit_export.empty()) {
        std::cout << "saving globfit input to " << settings.globfit_export << "..." << std::endl;
        features.export_globfit(settings.globfit_export, settings.visualization_stride, 1.0/diameter);
    }
    features.detect_bboxes();
    features.detect_contours(voxel_width, settings.contour_threshold);
    features.setCurrentShape(2); // use marching squares contour to find adjacency
    std::cout << "finding adjacency" << std::endl;
    features.detect_adjacencies(adjacency_threshold, settings.norm_adjacency_threshold);
    //PRINT / VISUALIZE ADJACENT POINTS/CLUSTERS
    {
        std::cout << "adjacencies:" << std::endl;
        for (size_t c = 0; c < features.adjacency.size(); c++) {
            std::cout << "cluster " << c << "(" << features.adjacency[c].size() << "): ";
            for (const auto &pair : features.adjacency[c]) {
                unsigned long adj = pair.first;
                std::cout << adj << ", ";
            }
            std::cout << std::endl;
        }
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> edges;
        for (const auto &adj : features.adjacency) {
            for (const auto &pair : adj) {
                edges.push_back(pair.second);
            }
        }
        visualizer.visualize_edges(edges, std::vector<Eigen::Vector3d>(1, Eigen::Vector3d(1, 1, 1)));

        if (settings.debug_visualization) {
            //visualize projection of edges
            for (auto &pair : reconstruction->images) {
                const auto &camera = reconstruction->cameras[pair.second.camera_id_];
                cv::Mat img = pair.second.getImage();
                cv::Mat imgresize;
                cv::resize(img, imgresize, cv::Size(camera.width_, camera.height_));
                for (const auto &edge : edges) {
                    Eigen::Vector3d pt_a = edge.first;
                    Eigen::Vector3d pt_b = edge.second;
                    Eigen::Matrix<double, 2, 3> pts_ab;
                    pts_ab << pt_a.transpose(), pt_b.transpose();
                    std::cout << "pts_ab: " << std::endl << pts_ab << std::endl;
                    Eigen::Matrix<double, 2, 2> pix_ab = reconstruction->project(pts_ab, pair.first);
                    cv::line(imgresize, cv::Point2d(pix_ab(0, 0), pix_ab(0, 1)),
                             cv::Point2d(pix_ab(1, 0), pix_ab(1, 1)),
                             cv::Scalar(0, 0, 255), 2);
                }
                cv::imwrite("alledges_" + std::to_string(pair.first) + ".png", imgresize);
            }
        }
    }
    // detect shapes
    /*{
        auto start_t = clock();
        std::cout << "detecting shapes... ";
        int numSuccess = features.detect_curves(voxel_width, min_curvature, settings.max_knots, bezier_cost, line_cost,
                                                settings.curve_weight);
        if (numSuccess < features.planes.size()) {
            std::cout << "Warning: " << numSuccess << '/' << features.planes.size() << " shapes found"  << std::endl;
            //return 1;
        }
        auto total_t = clock() - start_t;
        float time_sec = static_cast<float>(total_t) / CLOCKS_PER_SEC;
        std::cout << "finished in " << time_sec << " seconds" << std::endl;
    }*/
    features.setCurrentShape(2);
    features.detect_parallel(voxel_width, settings.norm_parallel_threshold);
    std::cout << "parallel:" << std::endl;
    for (size_t c=0; c<features.parallel.size(); c++) {
        if (!features.parallel[c].empty()) {
            std::cout << "cluster " << c << ": ";
            for (const auto &pair : features.parallel[c]) {
                std::cout << pair.first << " at " << pair.second << ", ";
            }
            std::cout << std::endl;
        }
    }

    //VISUALIZE SHAPES
    visualizer.update_colors(features.clusters.size());
    visualizer.visualize_shapes(features.vis_shapes_raw);
    //visualizer.visualize_shapes(features.vis_shapes);

    // COMPUTE VISIBILITY GROUPS
    features.detect_visibility(settings.max_views_per_cluster, threshold, settings.min_view_support);
    // print visibility groups
    /*std::cout << "visibility groups: " << std::endl;
    for (size_t c=0; c<features.cluster_visibility.size(); c++) {
        std::cout << "views of cluster " << c << ": ";
        for (auto v : features.cluster_visibility[c]) {
            std::cout << "im " << v << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "pruned visibility groups: " << std::endl;
    for (size_t c=0; c<features.pruned_cluster_visibility.size(); c++) {
        std::cout << "views of cluster " << c << ": ";
        for (auto v : features.pruned_cluster_visibility[c]) {
            std::cout << "im " << v << ", ";
        }
        std::cout << std::endl;
    }*/
    //std::cout << "total images used: " << num_images << '/' << reconstruction.images.size() << std::endl;

    // subdivide shapes
    {
        auto start_t = clock();
        std::cout << "subdividing shapes... ";
        features.split_shapes(settings.edge_detection_threshold, voxel_width);
        auto total_t = clock() - start_t;
        float time_sec = static_cast<float>(total_t) / CLOCKS_PER_SEC;
        std::cout << "finished in " << time_sec << " seconds" << std::endl;
    }
    //visualize cuts
    {
        visualizer.visualize_shapes(features.cut_edges);
    }

    /*if (settings.use_segmentation) {
        std::cout << "precomputing segmentations for each view" << std::endl;
        features.extract_image_shapes(settings, threshold, visualizer.colors());
        visualizer.visualize_shapes(features.vis_segmentation_shapes);
    }*/

    //EXTRUSION DEPTH DETECTION
    features.setCurrentShape(2);
    std::cout << "finding depths..." << std::endl;
    features.compute_depths(diameter / settings.thickness_resolution * 1.01, settings.thickness_resolution, voxel_width, settings.edge_detection_threshold, settings.thickness_candidates, settings.norm_parallel_threshold, std::log(settings.thickness_spatial_discount_factor) / diameter, settings.max_thickness);
    if (settings.thickness_clusters > 0) {
        std::cout << "filtering depths..." << std::endl;
        features.filter_depths(settings.thickness_clusters, diameter * 1e-6, false);
    }
    //visualize depths
    {
        std::vector<std::vector<std::vector<Eigen::Vector3d>>> extruded_shapes(features.planes.size());
        for (int c=0; c < features.vis_shapes_raw.size(); c++) {
            for (double d : features.depths[c]) {
            //double d = features.depths[c].back();
                Eigen::Vector3d displacement = -d * features.planes[c].basis().row(2).transpose().cast<double>();
                if (!features.vis_shapes_raw[c].empty()) {
                    extruded_shapes[c].emplace_back();
                    for (const auto &pt : features.vis_shapes_raw[c][0]) {
                        extruded_shapes[c].back().emplace_back(pt + displacement);
                    }
                }
            }
        }
        visualizer.visualize_shapes(extruded_shapes);
    }
    //

    /*std::cout << "finding depth-conditional connections" << std::endl;
    features.setCurrentShape(3); //use advanced contour for finding conditional adjacencies and point processing
    features.conditional_part_connections(voxel_width, settings.norm_adjacency_threshold, settings.norm_parallel_threshold);
    //visualize:
    {
        for (int c=0; c<features.conditional_connections.size(); c++) {
            if (!features.conditional_connections[c].empty()) {
                std::cout << "conditional connections for part " << c << ": ";
                for (const auto &pair : features.conditional_connections[c]) {
                    std::cout << '(' << pair.second.first << ", " << pair.first << "), ";
                }
                std::cout << std::endl;
            }
        }
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> edges;
        std::vector<Eigen::Vector3d> colors;
        for (int i=0; i<features.conditional_connections.size(); i++) {
            const auto &adj = features.conditional_connections[i];
            for (const auto &pair : adj) {
                edges.push_back(pair.second.second);
                colors.emplace_back(visualizer.color(i));
            }
        }
        visualizer.visualize_edges(edges, colors);
    }*/
#if 0
    std::cout << "generating samples..." << std::endl;
    //pcl::PointCloud<pcl::PointXYZ>::Ptr samples = features.generate_samples_biased();
    pcl::PointCloud<pcl::PointXYZ>::Ptr samples = features.generate_samples_random(settings.sample_count);
    std::cout << "generated " << samples->size() << " samples" << std::endl;
    std::cout << "filtering samples..." << std::endl;
    std::vector<std::size_t> filtered_indices;
    std::vector<size_t> sample_counts;
    std::vector<std::vector<int>> labels;
    features.analyze_samples(samples, filtered_indices, sample_counts, labels, settings.min_part_overlap, settings.contour_threshold, settings.winding_number_stride, settings.k_n);
    std::cout << "filtered to " << filtered_indices.size() << " samples" << std::endl;
    /*std::cout << "intersection term sizes: ";
    for (size_t s : sample_counts) {
        std::cout << s << ", ";
    }
    std::cout << std::endl;*/

    if (settings.min_samples > 0) {
        size_t N = filtered_indices.size();
        std::vector<size_t> new_filtered_indices;
        std::vector<size_t> new_sample_counts;
        std::vector<std::vector<int>> new_labels;
        for (size_t i=0; i<N; i++) {
            if (sample_counts[i] >= settings.min_samples) {
                new_filtered_indices.push_back(filtered_indices[i]);
                new_sample_counts.push_back(sample_counts[i]);
                new_labels.push_back(labels[i]);
            }
        }
        filtered_indices = std::move(new_filtered_indices);
        sample_counts = std::move(new_sample_counts);
        labels = std::move(new_labels);
        std::cout << "kept " << sample_counts.size() << '/' << N << " samples with at least " << settings.min_samples << " count" << std::endl;
    }

    //load mesh
    if (!settings.mesh_filename.empty()) {
        std::cout << "loading mesh" << std::endl;
        Eigen::MatrixX3d V;
        Eigen::MatrixX3i F;
        auto start_t = clock();
        if (!load_mesh(settings.mesh_filename, V, F)) {
            std::cerr << "failed to load " << settings.mesh_filename << std::endl;
            return 1;
        }
        auto total_t = clock() - start_t;
        float time_sec = static_cast<float>(total_t) / CLOCKS_PER_SEC;
        std::cout << "loaded in " << time_sec << " seconds" << std::endl;
        std::cout << "num vertices: " << V.size() << std::endl << "num faces: " << F.size() << std::endl;
        visualizer.visualize_mesh(V, F);
        Eigen::MatrixX3d P(filtered_indices.size(), 3);
        for (int i=0; i<filtered_indices.size(); i++) {
            P.row(i) = (*samples)[filtered_indices[i]].getVector3fMap().transpose().cast<double>();
        }
        std::vector<bool> labeling = mesh_interior_point_labeling(P, V, F);
        //pcl::PointCloud<pcl::PointXYZ>::Ptr new_filtered_samples(new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<size_t> new_filtered_indices;
        std::vector<size_t> new_sample_counts;
        for (int i=0; i<filtered_indices.size(); i++) {
            if (labeling[i]) {
                new_filtered_indices.push_back(filtered_indices[i]);
                new_sample_counts.push_back(sample_counts[i]);
            }
        }
        std::cout << new_filtered_indices.size() << '/' << filtered_indices.size() << " samples kept using mesh interior" << std::endl;
        filtered_indices = std::move(new_filtered_indices);
        sample_counts = std::move(new_sample_counts);
    }

    std::cout << "saving constraints file..." << std::endl;
    if (!features.save_samples("result.txt", samples, filtered_indices, sample_counts, labels, settings.max_constraints)) {
        std::cerr << "failed to save constraints file" << std::endl;
    } else {
        std::cout << "saved " << (settings.max_constraints > 0 ? std::min(filtered_indices.size(), static_cast<size_t>(settings.max_constraints)) : filtered_indices.size()) << std::endl;
    }

    //visualizer.visualize_points(cloud);
    //visualizer.visualize_sample_points(samples);
    visualizer.visualize_sample_points(samples, filtered_indices, Eigen::RowVector3d(1, 0, 0));
    visualizer.visualize_clusters(features.getPointCloud(), features.clusters);
    /*features.setCurrentShape(0); //visualize bboxes
    visualizer.visualize_primitives(features.planes, features.cylinders);*/
    features.setCurrentShape(3);
    visualizer.align_camera(features.getPointCloud());
    visualizer.launch();

    // LOAD THE SOLUTION
    std::cout << "press enter to load solution file" << std::endl;
    std::string input_line;
    std::getline(std::cin, input_line);
    //std::vector<BoundedPlane> part_planes;
    std::vector<std::vector<std::vector<Eigen::Vector3d>>> part_vis_shapes;
    std::vector<size_t> solution_samples;
    std::vector<size_t> non_solution_samples;
    {
        Solution solution(features);
        if (!solution.Load("solution.sol")) {
            std::cerr << "failed to load solver output" << std::endl;
            return 1;
        }
        solution_samples.resize(solution.num_samples);
        non_solution_samples.resize(filtered_indices.size() - solution.num_samples);
        for (int i=0; i<filtered_indices.size(); i++) {
            if (i < solution.num_samples) {
                solution_samples[i] = filtered_indices[i];
            } else {
                non_solution_samples[i-solution.num_samples] = filtered_indices[i];
            }
        }
        std::ofstream solution_of("solution.xml");
        solution_of << solution << std::endl;
        for (int part_id : solution.part_ids) {
            part_vis_shapes.push_back(features.vis_shapes[part_id]);
        }
    }


    Visualizer visualizer2;
    visualizer2.update_colors(part_vis_shapes.size());

    visualizer2.visualize_sample_points(samples, solution_samples, Eigen::RowVector3d(0, 1, 0));
    visualizer2.visualize_sample_points(samples, non_solution_samples, Eigen::RowVector3d(1, 0, 0));
    {
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> extruded_edges;
        std::vector<Eigen::Vector3d> extruded_colors;
        for (int i = 0; i < features.extrusion_edges.size(); i++) {
            for (double d : features.depths[i]) {
                for (int n = 0; n < features.extrusion_edges[i].size(); n++) {
                    auto &edge = features.extrusion_edges[i][n];
                    Eigen::Vector3d displacement = -d * features.planes[i].basis().row(2).transpose().cast<double>();
                    extruded_edges.emplace_back(edge.first + displacement, edge.second + displacement);
                    extruded_colors.emplace_back(visualizer.color(i));
                }
            }
        }
        visualizer2.visualize_edges(extruded_edges, extruded_colors);
    }
    visualizer2.visualize_shapes(part_vis_shapes);
    visualizer2.launch();
#else
    visualizer.launch();
#endif
    return 0;
}
