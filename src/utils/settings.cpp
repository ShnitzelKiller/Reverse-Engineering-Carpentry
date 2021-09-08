    //
// Created by James Noeckel on 3/30/20.
//

#include "settings.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

std::string joinpaths(const std::string &base, const std::string &path) {
    if (path.empty()) {
        return "";
    } else if (path[0] == '/') {
        return path;
    } else {
        return base + path;
    }
}

std::ostream &operator<<(std::ostream &o, const Settings &s) {
    o << "Settings: ";
    o << "random seed: " << s.random_seed << std::endl;
    o << "master resolution: " << s.master_resolution << std::endl;
    o << "final scale: " << s.final_scale << std::endl;
    o << "normal threshold: " << s.normal_threshold << std::endl;
    o << "clustering factor: " << s.clustering_factor << std::endl;
    o << "constraint factor: " << s.constraint_factor << std::endl;
    o << "probability: " << s.probability << std::endl;
    o << "use cylinders: " << s.use_cylinders << std::endl;
    o << "alpha: " << s.alpha << std::endl;
    o << "voxel resolution: " << s.voxel_resolution << std::endl;
    o << "min contour hole ratio: " << s.max_contour_hole_ratio << std::endl;
    o << "k_n: " << s.k_n << std::endl;
    o << "use winding number: " << s.use_winding_number << std::endl;
    if (s.use_winding_number) {
        o << "winding number stride: " << s.winding_number_stride << std::endl;
    }
    o << "use geocuts: " << s.use_geocuts << std::endl;
    o << "contour threshold: " << s.contour_threshold << std::endl;
    o << "support: " << s.support << std::endl;
    o << "min view support: " << s.min_view_support << std::endl;
    o << "max views per cluster: " << s.max_views_per_cluster << std::endl;
    o << "max view angle: " << s.max_view_angle << std::endl;
    o << "outer curve count: " << s.outer_curve_count << std::endl;
    o << "inner curve count: " << s.inner_curve_count << std::endl;
    o << "min knot curvature: " << s.min_knot_angle << std::endl;
    o << "max knots: " << s.max_knots << std::endl;
    o << "curve weight: " << s.curve_weight << std::endl;
    o << "line cost: " << s.line_cost << std::endl;
    o << "curve cost: " << s.curve_cost << std::endl;
    o << "max thickness: " << s.max_thickness << std::endl;
    o << "thickness resolution: " << s.thickness_resolution << std::endl;
    o << "thickness candidates: " << s.thickness_candidates << std::endl;
    o << "thickness clusters: " << s.thickness_clusters << std::endl;
    o << "thickness spatial discounting factor: " << s.thickness_spatial_discount_factor << std::endl;
    o << "normal adjacency threshold: " << s.norm_adjacency_threshold << std::endl;
    o << "normal parallel threshold: " << s.norm_parallel_threshold << std::endl;
    o << "adjacency factor: " << s.adjacency_factor << std::endl;
    o << "sample count: " << s.sample_count << std::endl;
    o << "min samples: " << s.min_samples << std::endl;
    o << "max constraints: " << s.max_constraints << std::endl;
    o << "min part overlap: " << s.min_part_overlap << std::endl;
    //o << "adjacency edge threshold: " << s.adjacency_edge_factor << std::endl;
    o << "points filename: " << s.points_filename << std::endl;
    o << "mesh filename: " << s.mesh_filename << std::endl;
    o << "reconstruction path: " << s.reconstruction_path << std::endl;
    o << "image path: " << s.image_path << std::endl;
    o << "depth path: " << s.depth_path << std::endl;
    o << "image scale: " << s.image_scale << std::endl;
    o << "max image resolution: " << s.max_image_resolution << std::endl;
    o << "globfit export: " << s.globfit_export << std::endl;
    o << "globfit import: " << s.globfit_import << std::endl;
    o << "globfit scale: " << s.globfit_scale << std::endl;
    o << "export location: " << s.result_path << std::endl;
    o << "selection checkpoint: " << s.selection_checkpoint << std::endl;
    o << "connection checkpoint: " << s.connection_checkpoint << std::endl;
    o << "segmentation checkpoint: " << s.segmentation_checkpoint << std::endl;
    o << "old result checkpoint: " << s.oldresult_checkpoint << std::endl;
    o << "curvefit checkpoint: " << s.curvefit_checkpoint << std::endl;
    o << "connector mesh: " << s.connector_mesh << std::endl;
    o << "connector spacing: " << s.connector_spacing << std::endl;
    o << "connector scale: " << s.connector_scale << std::endl;
    o << "correction factor: " << s.correction_factor << std::endl;
    o << "patch radius: " << s.patch_radius << std::endl;
    o << "edge detection threshold: " << s.edge_detection_threshold << std::endl;
    o << "use segmentation: " << s.use_segmentation << std::endl;
    o << "generations: " << s.generations << std::endl;
    o << "max overlap: " << s.max_overlap_ratio << std::endl;
    o << "population size: " << s.population_size << std::endl;
    o << "alignment stride: " << s.alignment_stride << std::endl;
    o << "alignment tolerance: " << s.alignment_tol << std::endl;
    o << "align parallel threshold: " << s.align_parallel_threshold << std::endl;
    //if (s.use_segmentation) {
        o << "segmentation max views per cluster: " << s.segmentation_max_views_per_cluster << std::endl;
        o << "segmentation precision: " << s.segmentation_precision << std::endl;
        o << "segmentation iterations: " << s.segmentation_iters << std::endl;
        o << "segmentation scale: " << s.segmentation_scale << std::endl;
        o << "segmentation data weight: " << s.segmentation_data_weight << std::endl;
        o << "segmentation smoothing weight: " << s.segmentation_smoothing_weight << std::endl;
        o << "segmentation penalty: " << s.segmentation_penalty << std::endl;
        o << "segmentation sigma: " << s.segmentation_sigma << std::endl;
        o << "segmentation levels: " << s.segmentation_levels << std::endl;
        o << "segmentation 8-way connections: " << (s.segmentation_8way ? "true" : "false") << std::endl;
        o << "segmentation gmm components: " << s.segmentation_gmm_components << std::endl;
        o << "segmentation gmm min variance: " << s.segmentation_gmm_min_variance << std::endl;
        o << "segmentation min pixel support: " << s.segmentation_min_pixel_support << std::endl;
        o << "segmentation clean precision threshold: " << s.segmentation_clean_precision_threshold << std::endl;
        o << "segmentation clean recall threshold: " << s.segmentation_clean_recall_threshold << std::endl;
    //}
    o << "visualization? : " << s.visualize << std::endl;
    if (s.visualize) {
        o << "visualization stride: " << s.visualization_stride << std::endl;
    }
    o << "debug visualization: " << s.debug_visualization << std::endl;
    return o;
}

bool Settings::store_line(const std::string &key, const std::string &value) {
    try {
        if (key == "master_resolution") {
            this->master_resolution = std::stof(value);
        } else if (key == "final_scale") {
            this->final_scale = std::stof(value);
        } else if (key == "random_seed") {
            this->random_seed = std::stoi(value);
        } else if (key == "normal_threshold") {
            this->normal_threshold = std::stof(value);
        } else if (key == "clustering_factor") {
            this->clustering_factor = std::stof(value);
        } else if (key == "constraint_factor") {
            this->constraint_factor = std::stof(value);
        } else if (key == "probability") {
            this->probability = std::stof(value);
        } else if (key == "use_cylinders") {
            this->use_cylinders = std::stoi(value);
        } else if (key == "alpha") {
            this->alpha = std::stof(value);
        } else if (key == "voxel_resolution") {
            this->voxel_resolution = std::stof(value);
        } else if (key == "max_contour_hole_ratio") {
            this->max_contour_hole_ratio = std::stof(value);
        } else if (key == "k_n") {
            this->k_n = std::stoi(value);
        } else if (key == "use_winding_number") {
            this->use_winding_number = std::stoi(value);
        } else if (key == "winding_number_stride") {
            this->winding_number_stride = std::stoi(value);
        } else if (key == "use_geocuts") {
            this->use_geocuts = std::stoi(value);
        } else if (key == "contour_threshold") {
            this->contour_threshold = std::stof(value);
        } else if (key == "support") {
            this->support = std::stof(value);
        } else if (key == "min_view_support") {
            this->min_view_support = std::stol(value);
        } else if (key == "max_views_per_cluster") {
            this->max_views_per_cluster = std::stoi(value);
        } else if (key == "max_view_angle") {
            this->max_view_angle = std::stof(value);
        } else if (key == "outer_curve_count") {
            this->outer_curve_count = std::stoi(value);
        } else if (key == "inner_curve_count") {
            this->inner_curve_count = std::stoi(value);
        } else if (key == "min_knot_angle") {
            this->min_knot_angle = std::stof(value);
        } else if (key == "max_knots") {
            this->max_knots = std::stoi(value);
        } else if (key == "line_cost") {
            this->line_cost = std::stof(value);
        } else if (key == "curve_cost") {
            this->curve_cost = std::stof(value);
        } else if (key == "curve_weight") {
            this->curve_weight = std::stof(value);
        }else if (key == "max_thickness") {
            this->max_thickness = std::stof(value);
        } else if (key == "thickness_resolution") {
            this->thickness_resolution = std::stoi(value);
        } else if (key == "thickness_candidates") {
            this->thickness_candidates = std::stoi(value);
        } else if (key == "thickness_clusters") {
            this->thickness_clusters = std::stoi(value);
        } else if (key == "thickness_spatial_discounting_factor") {
            this->thickness_spatial_discount_factor = std::stof(value);
        } else if (key == "norm_adjacency_threshold") {
            this->norm_adjacency_threshold = std::stof(value);
        } else if (key == "norm_parallel_threshold") {
            this->norm_parallel_threshold = std::stof(value);
        } else if (key == "sample_count") {
            this->sample_count = std::stoul(value);
        } else if (key == "min_samples") {
            this->min_samples = std::stoi(value);
        } else if (key == "max_constraints") {
            this->max_constraints = std::stoi(value);
        } else if (key == "min_part_overlap") {
            this->min_part_overlap = std::stoi(value);
        } else if (key == "adjacency_factor") {
            this->adjacency_factor = std::stof(value);
        //} else if (key == "adjacency_edge_factor") {
        //    this->adjacency_edge_factor = std::stof(value);
        } else if (key == "points_filename") {
            this->points_filename = joinpaths(settings_path, value);
        } else if (key == "mesh_filename") {
            this->mesh_filename = joinpaths(settings_path, value);
        } else if (key == "reconstruction_path") {
            this->reconstruction_path = joinpaths(settings_path, value);
        } else if (key == "image_path") {
            this->image_path = joinpaths(settings_path, value);
        } else if (key == "depth_path") {
            this->depth_path = joinpaths(settings_path, value);
        } else if (key == "image_scale") {
            this->image_scale = std::stof(value);
        } else if (key == "max_image_resolution") {
            this->max_image_resolution = std::stoi(value);
            if (this->max_image_resolution < 0) {
                this->max_image_resolution = std::numeric_limits<int>::max();
            }
        } else if (key == "min_pixels_per_voxel") {
            this->min_pixels_per_voxel = std::stoi(value);
        } else if (key == "globfit_export") {
            this->globfit_export = value;
        } else if (key == "globfit_import") {
            this->globfit_import = joinpaths(settings_path, value);
        } else if (key == "globfit_scale") {
            this->globfit_scale=std::stof(value);
        } else if (key == "result_path") {
            this->result_path = value;
        } else if (key == "oldresult_checkpoint") {
            this->oldresult_checkpoint = value;
        } else if (key == "curvefit_checkpoint") {
            this->curvefit_checkpoint = value;
        } else if (key == "segmentation_checkpoint") {
            this->segmentation_checkpoint = value;
        } else if (key == "connection_checkpoint") {
            this->connection_checkpoint = value;
        } else if (key == "selection_checkpoint") {
            this->selection_checkpoint = value;
        } else if (key == "connector_mesh") {
            this->connector_mesh = joinpaths(settings_path, value);
        } else if (key == "connector_spacing") {
            this->connector_spacing = std::stof(value);
        } else if (key == "connector_scale") {
            this->connector_scale = std::stof(value);
        } else if (key == "correction_factor") {
            this->correction_factor = std::stof(value);
        } else if (key == "patch_radius") {
            this->patch_radius = std::stoi(value);
        } else if (key == "edge_detection_threshold") {
            this->edge_detection_threshold = std::stof(value);
        } else if (key == "use_segmentation") {
            this->use_segmentation = std::stoi(value);
        } else if (key == "segmentation_precision") {
            this->segmentation_precision = std::stof(value);
        } else if (key == "segmentation_max_views_per_cluster") {
            this->segmentation_max_views_per_cluster = std::stoi(value);
        } else if (key == "segmentation_iters") {
            this->segmentation_iters = std::stoi(value);
        } else if (key == "segmentation_scale") {
            this->segmentation_scale = std::stof(value);
        } else if (key == "segmentation_data_weight") {
            this->segmentation_data_weight = std::stof(value);
        } else if (key == "segmentation_smoothing_weight") {
            this->segmentation_smoothing_weight = std::stof(value);
        } else if (key == "segmentation_penalty") {
            this->segmentation_penalty = std::stof(value);
        } else if (key == "segmentation_sigma") {
            this->segmentation_sigma = std::stof(value);
        } else if (key == "segmentation_8way") {
            this->segmentation_8way = std::stoi(value);
        } else if (key == "segmentation_levels") {
            this->segmentation_levels = std::stoi(value);
        } else if (key == "segmentation_gmm_components") {
            this->segmentation_gmm_components = std::stoi(value);
        } else if (key == "segmentation_gmm_min_variance") {
            this->segmentation_gmm_min_variance = std::stof(value);
        } else if (key == "segmentation_min_support") {
            this->segmentation_min_pixel_support = std::stoi(value);
        } else if (key == "segmentation_clean_precision_threshold") {
            this->segmentation_clean_precision_threshold = std::stof(value);
        } else if (key == "segmentation_clean_recall_threshold") {
            this->segmentation_clean_recall_threshold = std::stof(value);
        } else if (key == "population_size") {
            this->population_size = std::stoul(value);
        } else if (key == "generations") {
            this->generations = std::stoi(value);
        } else if (key == "max_overlap_ratio") {
            this->max_overlap_ratio = std::stof(value);
        } else if (key == "alignment_stride") {
            this->alignment_stride = std::stoi(value);
        } else if (key == "alignment_tol") {
            this->alignment_tol = std::stod(value);
        } else if (key == "alignment_parallel_threshold") {
            this->align_parallel_threshold = std::stof(value);
        } else if (key == "visualize") {
            this->visualize = std::stoi(value);
        } else if (key == "visualization_stride") {
            this->visualization_stride = std::stoi(value);
        } else if (key == "debug_visualization") {
            this->debug_visualization = std::stoi(value);
        } else {
            return false;
        }
    } catch (std::exception &e) {
        std::cout << "parse error (" << e.what() << "): " << key << "=" << value << std::endl;
        return false;
    }
    return true;
}

bool Settings::parse_file(const std::string &settings_filename) {
    settings_path = std::string(settings_filename);
    size_t delim_pos = settings_path.rfind('/');
    if (delim_pos != std::string::npos) {
        settings_path = settings_path.substr(0, delim_pos + 1);
    } else {
        settings_path = "";
    }
    std::cout << "settings path: " << settings_path << std::endl;
    std::ifstream if_config(settings_filename);
    if (if_config) {
        std::string line;
        while (std::getline(if_config, line)) {
            size_t found = line.find('#');
            if (found != std::string::npos) {
                line = line.substr(0, found);
            }
            std::istringstream is_line(line);
            std::string key;
            if (std::getline(is_line, key, '=')) {
                std::string value;
                if (std::getline(is_line, value)) {
                    if (!store_line(key, value)) {
                        std::cout << "invalid setting: " << line << std::endl;
                        return false;
                    }
                }
            }
        }
        setDerived();
        return true;
    } else {
        std::cout << "failed to load config file " << settings_filename << std::endl;
        return false;
    }
}

void Settings::setDerived() {
    max_view_angle_cos = std::cos(max_view_angle);
}