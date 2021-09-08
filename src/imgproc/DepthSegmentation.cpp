//
// Created by James Noeckel on 2/4/20.
//

#include "DepthSegmentation.h"
#include "multilabel_graph_cut.h"
#include "utils/sorted_data_structures.hpp"

DepthSegmentation::DepthSegmentation(const std::vector<BoundedPlane> &bboxes, const std::vector<Cylinder> &cylinders, const std::vector<int> &bbox_clusters, Image &image, const CameraIntrinsics &camera, PointCloud3::Handle cloud, float scale, float correction_factor, int colorspace) :
        scale_(scale),
        correction_factor_(correction_factor),
        rot_(image.rot_),
        trans_(image.trans_),
        fdist_(camera.params_.data()),
        principal_point_(camera.params_.data() + 2),
        bboxes_(bboxes),
        cylinders_(cylinders),
        visible_clusters_(bbox_clusters),
        cloud_(std::move(cloud))
{
    int bbox_index = 0;
    int cylinder_index = 0;
    int global_index = 0;
    for (size_t bbox_cluster : bbox_clusters) {
        if (bbox_cluster < bboxes.size()) {
            planar_clusters_.push_back(bbox_cluster);
            bbox_indices_[bbox_cluster] = bbox_index++;
        } else {
            cylinder_clusters_.push_back(bbox_cluster);
            cylinder_indices_[bbox_cluster] = cylinder_index++;
        }
        global_indices_[bbox_cluster] = global_index++;
    }
    use_cylinder_exteriors_.resize(cylinder_index);
    cv::Mat image_raw = image.getImage();
    cv::Mat depth_map = image.getDepthGeometric();
    if (depth_map.empty() || image_raw.empty() || depth_map.rows != image_raw.rows || depth_map.cols != image_raw.cols) {
        width_ = 0;
        height_ = 0;
    } else {
        cv::resize(depth_map, depth_map_, cv::Size(), scale, scale, cv::INTER_NEAREST);
        width_ = depth_map_.cols;
        height_ = depth_map_.rows;
        cv::resize(image_raw, image_, cv::Size(width_, height_), 0, 0, cv::INTER_NEAREST);
        cv::cvtColor(image_, image_, cv::COLOR_BGR2Lab, 3);
    }

}

void DepthSegmentation::computeDepthSegmentation(float threshold) {
    if (width_ == 0) return;
    segmentation_ = -cv::Mat::ones(height_,width_, CV_32SC1);
    std::vector<int> cylinder_outside_counts(cylinder_clusters_.size(), 0);
    std::vector<int> cylinder_inside_counts(cylinder_clusters_.size(), 0);

    for (int i=0; i<height_; i++) {
        for (int j=0; j<width_; j++) {
            Eigen::Vector3d pos_cam(static_cast<double>(j)/scale_, static_cast<double>(i)/scale_, depth_map_.at<double>(i, j));
            pos_cam.head(2) -= principal_point_.matrix();
            pos_cam.head(2).array() *= pos_cam.z()/(fdist_);
            Eigen::Vector3d pos_world = rot_.conjugate() * (pos_cam - trans_);
            for (size_t c : visible_clusters_.get()) {
                bool contained = false;
                if (c < bboxes_.get().size()) {
                    contained = bboxes_.get()[c].contains3D(pos_world, threshold);
                } else {
                    const Cylinder& cylinder = cylinders_.get()[c - bboxes_.get().size()];
                    contained = cylinder.contains3D(pos_world, threshold);
                    if (contained) {
                        // check whether this is an outside or inside cylinder point
                        Eigen::Vector3d ray_dir(static_cast<double>(j) / scale_, static_cast<double>(i) / scale_, 1);
                        ray_dir.head(2) -= principal_point_.matrix();
                        ray_dir.head(2).array() /= (fdist_);
                        ray_dir.normalize();
                        ray_dir = rot_.conjugate() * ray_dir;
                        double dotprod = ray_dir.dot(cylinder.normal(pos_world));
                        if (dotprod > 0) {
                            cylinder_inside_counts[cylinder_indices_[c]] += 1;
                        } else {
                            cylinder_outside_counts[cylinder_indices_[c]] += 1;
                        }
                    }
                }
                if (contained) {
                    segmentation_.at<int32_t>(i, j) = static_cast<int32_t>(c);
                    break;
                }
            }
        }
    }

    for (int i=0; i<cylinder_clusters_.size(); i++) {
        use_cylinder_exteriors_[i] = cylinder_outside_counts[i] > cylinder_inside_counts[i];
    }

    //background
    std::vector<cv::Point> allpoints;
    allpoints.reserve(cloud_->P.rows());
    for (size_t p=0; p<cloud_->P.rows(); p++) {
        Eigen::Vector3d pt = cloud_->P.row(p).head(3);
        Eigen::Array3d pt_proj = rot_ * pt + trans_;
        Eigen::Array2d pt_pix = (pt_proj.head(2)/pt_proj.z() * fdist_ * correction_factor_ + principal_point_) * static_cast<double>(scale_);
        allpoints.emplace_back(pt_pix.x(), pt_pix.y());
    }

    std::vector<cv::Point> hull;
    cv::convexHull(allpoints, hull);
    cv::Mat background_mask = cv::Mat::ones(segmentation_.size(), CV_8UC1);
    cv::fillConvexPoly(background_mask, hull.data(), hull.size(), 0);
    int erosion_rad = std::min(segmentation_.rows, segmentation_.cols) / 16;
    int kernel_size = erosion_rad*2+1;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size), cv::Point(erosion_rad, erosion_rad));
    cv::erode(background_mask, background_mask, kernel);
    background_mask.convertTo(background_mask, CV_32SC1);
    segmentation_ = segmentation_.mul(1 - background_mask) + BACKGROUND_VALUE * background_mask;
    dirty_ = true;
}

void DepthSegmentation::outlierRemoval(int min_support) {
    if (width_ == 0) return;
    for (size_t c : visible_clusters_.get()) {
        cv::Mat binary_mask = segmentation_ == c;
        cv::Mat connected_components(binary_mask.size(), CV_32SC1);
        cv::Mat stats;
        cv::Mat centroids;
        int comps = cv::connectedComponentsWithStats(binary_mask, connected_components, stats, centroids, 4, CV_32SC1, cv::CCL_DEFAULT);
        if (comps > 1) {
            for (int i = 0; i < connected_components.rows; i++) {
                for (int j = 0; j < connected_components.cols; j++) {
                    int comp = connected_components.at<int32_t>(i, j);
                    if (comp > 0) {
                        if (stats.at<int32_t>(comp, cv::CC_STAT_AREA) < min_support) {
                            segmentation_.at<int32_t>(i, j) = -1;
                        }
                    }
                }
            }
        }
    }
    dirty_ = true;
}


void DepthSegmentation::energyMinimization(float segmentation_data_weight, float segmentation_smoothing_weight, float segmentation_penalty, float segmentation_sigma, int num_components, int segmentation_iters, int levels) {
    if (width_ == 0) return;
    multilabel_graph_cut(image_, segmentation_, segmentation_data_weight, segmentation_smoothing_weight, segmentation_penalty, segmentation_sigma, num_components, segmentation_iters, levels);
    dirty_ = true;
}

void DepthSegmentation::cleanSegmentation(int iterations, float precision_threshold, float recall_threshold, bool optimal_cluster_reassignment, bool allow_background_conversion) {
    if (width_ == 0) return;
    //filter segmentation mask using bboxes and connected components
    //TODO: discard based on tuneable threshold
    //TODO: weight cluster-view importance based on amount of invalid labelings detected
    Eigen::Vector3d camera_pos = -(rot_.conjugate() * trans_);

    // FIND ALL CONNECTED COMPONENTS WITH COORDINATES (except background)
    std::vector<std::vector<cv::Point>> all_connected_coords;
    cv::Mat total_connected_components = -cv::Mat::ones(segmentation_.size(), CV_32SC1);
    int comp_offset = 0;
    for (size_t c : visible_clusters_.get()) {
        cv::Mat binary_mask = segmentation_ == c;
        cv::Mat connected_components(binary_mask.size(), CV_32SC1);
        int comps = cv::connectedComponents(binary_mask, connected_components, 4);
        if (comps > 1) {
            std::vector <std::vector<cv::Point>> connected_coords(comps-1);
            for (int i = 0; i < connected_components.rows; i++) {
                for (int j = 0; j < connected_components.cols; j++) {
                    cv::Point pt(j, i);
                    int lbl = connected_components.at<int32_t>(pt);
                    if (lbl > 0) {
                        connected_coords[lbl - 1].push_back(pt);
                        total_connected_components.at<int32_t>(pt) = lbl - 1 + comp_offset;
                    }
                }
            }
            std::move(connected_coords.begin(), connected_coords.end(),
                      std::back_insert_iterator < std::vector < std::vector < cv::Point >> >
                              (all_connected_coords));
            comp_offset += comps-1;
        }
    }

    size_t num_components = all_connected_coords.size();

    // FIND NUMBER OF VALID PIXELS IN EACH COMPONENT
    // cluster -> count
    std::vector<int> total_counts(visible_clusters_.get().size(), 0);

    // connected component -> (cluster -> count)
    std::vector<std::vector<int>> valid_counts(num_components, std::vector<int>(visible_clusters_.get().size(), 0));

    for (int i=0; i<total_connected_components.rows; i++) {
        for (int j=0; j<total_connected_components.cols; j++) {
            int comp = total_connected_components.at<int32_t>(i, j);
            Eigen::Vector3d ray_dir(static_cast<double>(j) / scale_, static_cast<double>(i) / scale_, 1);
            ray_dir.head(2) -= principal_point_.matrix();
            ray_dir.head(2).array() /= (fdist_);
            ray_dir.normalize();
            ray_dir = rot_.conjugate() * ray_dir;
            for (auto c : visible_clusters_.get()) {
                double t1, t2;
                bool contained = false;
                if (c < bboxes_.get().size()) {
                    contained = bboxes_.get()[c].intersectRay(camera_pos, ray_dir, t1);
                } else {
                    bool intersected1, intersected2;
                    cylinders_.get()[c - bboxes_.get().size()].intersect3D(camera_pos, ray_dir, t1, t2, intersected1, intersected2);
                    if (use_cylinder_exteriors_[cylinder_indices_[c]]) {
                        contained = intersected1;
                    } else {
                        contained = intersected2;
                    }
                }
                if (contained) {
                    if (recall_threshold <= 1) {
                        total_counts[global_indices_[c]] += 1;
                    }
                    if (comp >= 0) {
                        valid_counts[comp][global_indices_[c]] += 1;
                    }
                }
            }
        }
    }

    // POPULATE NEIGHBORS
    // connected component -> (connected component -> count)
    std::unordered_map<int, std::vector<std::pair<int, int>>> neighbor_counts;
    for (int comp_id=0; comp_id<num_components; comp_id++) {
        auto &coords = all_connected_coords[comp_id];
        int cluster = segmentation_.at<int32_t>(coords[0]);
        int num_valid = valid_counts[comp_id][global_indices_[cluster]];
        // find adjacent component counts for each component below valid threshold
        int precision_count = static_cast<int>(coords.size() * precision_threshold);
        int recall_count = recall_threshold > 1 ? std::numeric_limits<int>::max() : static_cast<int>(total_counts[global_indices_[cluster]] * recall_threshold);
        if (num_valid <= std::min(precision_count, recall_count)) {
            for (const cv::Point &pt : coords) {
                for (int dim=0; dim<=1; dim++) {
                    for (int o=-1; o<=1; o+=2) {
                        Eigen::Vector2i pt_2p(pt.x, pt.y);
                        pt_2p[dim] += o;
                        cv::Point pt2(pt_2p.x(), pt_2p.y());
                        if (pt2.x >= 0 && pt2.y >= 0 && pt2.x < segmentation_.cols && pt2.y < segmentation_.rows) {
                            int comp_id_2 = total_connected_components.at<int32_t>(pt2);
                            if (comp_id != comp_id_2) {
                                if (sorted_contains(neighbor_counts[comp_id], comp_id_2)) {
                                    sorted_get(neighbor_counts[comp_id], comp_id_2) += 1;
                                } else {
                                    sorted_insert(neighbor_counts[comp_id], comp_id_2, 1);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // map invalid connected components to the best matching cluster using neighboring components and membership scores
    if (optimal_cluster_reassignment) {
        std::vector<int> visited;
        //TODO: retry this several times sequentially as well? Still relies on neighbors
        for (const auto &comp_counts : neighbor_counts) {
            // sort clusters by valid scores and look in descending order for a cluster represented by a neighboring component
            // until the cutoff validity threshold
            std::vector<std::pair<int, int>> cluster_rank;
            cluster_rank.reserve(visible_clusters_.get().size());
            for (size_t c : visible_clusters_.get()) {
                cluster_rank.emplace_back(c, valid_counts[comp_counts.first][global_indices_[c]]);
            }
            std::sort(cluster_rank.begin(), cluster_rank.end(), [](std::pair<int, int> pair1, std::pair<int, int> pair2) {return pair1.second < pair2.second;});
            bool found = false;
            for (size_t i=cluster_rank.size()-1; i>=0; i--) {
                if (cluster_rank[i].second < std::max(static_cast<size_t>(1), all_connected_coords[comp_counts.first].size())) break;
                for (auto lbl_count : comp_counts.second) {
                    if (lbl_count.first >= 0 && segmentation_.at<int32_t>(all_connected_coords[lbl_count.first][0]) == cluster_rank[i].first) {
                        for (const auto &pt : all_connected_coords[comp_counts.first]) {
                            segmentation_.at<int32_t>(pt) = cluster_rank[i].first;
                        }
                        found = true;
                        break;
                    }
                }
                if (found) break;
            }
            if (found) {
                visited.push_back(comp_counts.first);
            }
        }
        for (int n : visited) {
            neighbor_counts.erase(n);
        }
    }

    // for all the components that were not fixed in the previous step,
    // map invalid connected components to the color of the valid component with the most neighboring pixels
    for (int i=0; !neighbor_counts.empty() && i<iterations; i++) {
        std::vector<int> visited;
        for (const auto &comp_counts : neighbor_counts) {
            int max_count = 0;
            int max_index;
            bool found = false;
            for (const auto &count : comp_counts.second) {
                if (neighbor_counts.find(count.first) == neighbor_counts.end() && count.second > max_count) {
                    if (!allow_background_conversion && count.first < 0) continue;
                    max_count = count.second;
                    max_index = count.first;
                    found = true;
                }
            }
            if (found) {
                int cluster = max_index < 0 ? BACKGROUND_VALUE : segmentation_.at<int32_t>(all_connected_coords[max_index][0]);
                for (const auto &pt : all_connected_coords[comp_counts.first]) {
                    segmentation_.at<int32_t>(pt) = cluster;
                }
                visited.push_back(comp_counts.first);
            }
        }
        for (int n : visited) {
            neighbor_counts.erase(n);
        }
    }
    dirty_ = true;
}

cv::Mat DepthSegmentation::getSegmentation() const {
    return segmentation_;
}

const std::vector<double> &DepthSegmentation::scores() {
    if (!dirty_) return scores_;
    if (segmentation_.empty()) {
        return scores_;
    }
    Eigen::Vector3d camera_pos = -(rot_.conjugate() * trans_);
    std::vector<size_t> totals(bboxes_.get().size(), 0);
    for (int i=0; i<segmentation_.rows; i++) {
        for (int j=0; j<segmentation_.cols; j++) {
            Eigen::Vector3d ray_dir(static_cast<double>(j) / scale_, static_cast<double>(i) / scale_, 1);
            ray_dir.head(2) -= principal_point_.matrix();
            ray_dir.head(2).array() /= (fdist_);
            ray_dir.normalize();
            ray_dir = rot_.conjugate() * ray_dir;
            int cluster = segmentation_.at<int32_t>(i, j);
            if (cluster >= 0) {
                double t1, t2;
                bool contained = false;
                if (cluster < bboxes_.get().size()) {
                    contained = bboxes_.get()[cluster].intersectRay(camera_pos, ray_dir, t1);
                } else {
                    bool intersected1, intersected2;
                    cylinders_.get()[cluster - bboxes_.get().size()].intersect3D(camera_pos, ray_dir, t1, t2, intersected1, intersected2);
                    contained = intersected1 || intersected2;
                }
                if (contained) {
                    scores_[cluster] = (totals[cluster] * scores_[cluster] + 1) / (scores_[cluster] + 1);
                }
                totals[cluster]++;
            }
        }
    }
    dirty_ = false;
    return scores_;
}

bool DepthSegmentation::isValid() const {
    return width_ > 0 && height_ > 0;
}
