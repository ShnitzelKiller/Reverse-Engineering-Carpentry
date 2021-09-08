//
// Created by James Noeckel on 1/21/20.
//

#include "region_growing.h"
#include <queue>
#include <Eigen/Dense>
#include "math/GaussianMixture.h"
#include <iostream>

#define TRAINING_SIZE 100

typedef std::pair<float, std::pair<int, int>> WeightedPosLabel;

void region_growing(const cv::Mat &image, cv::Mat &seed, float gmm_weight) {
    if (image.empty() || seed.empty() || image.rows != seed.rows || image.cols != seed.cols) {
        return;
    }
    std::priority_queue<WeightedPosLabel, std::vector<WeightedPosLabel>, std::greater<WeightedPosLabel>> pq;
    //cv::Mat costs = cv::Mat::ones(seed.rows, seed.cols, CV_32FC1) * std::numeric_limits<float>::max();
    std::vector<GaussianMixture> gmms;
    std::unordered_map<int, size_t> cluster_indices;
    if (gmm_weight > 0) {
        //initialize GMMs for each cluster
        size_t index = 0;
        for (int i = 0; i < seed.rows; i++) {
            for (int j = 0; j < seed.cols; j++) {
                int cluster = seed.at<int>(i, j);
                auto it = cluster_indices.find(cluster);
                if (it == cluster_indices.end()) {
                    cluster_indices[cluster] = index;
                    index++;
                }
            }
        }
        std::vector<std::vector<Eigen::RowVector3i>> colors(index);
        for (int i = 0; i < seed.rows; i++) {
            for (int j = 0; j < seed.cols; j++) {
                int cluster = seed.at<int>(i, j);
                cv::Vec3b col = image.at<cv::Vec3b>(i, j);
                colors[cluster_indices[cluster]].emplace_back(col[0], col[1], col[2]);
            }
        }
        std::cout << "learning " << index << " gmms" << std::endl;

        gmms.reserve(index);
        for (int i = 0; i < index; i++) {
            gmms.emplace_back(3, 3);
            int n = std::min(TRAINING_SIZE, static_cast<int>(colors[i].size()));
            Eigen::MatrixX3d data(n, 3);
            int full_n = colors[i].size();
            Eigen::VectorXi picks = Eigen::VectorXi::Random(n);
            picks = picks.unaryExpr([&](const int num) { return abs(num) % full_n; });
            for (int j = 0; j < n; j++) {
                data.row(j) = colors[i][picks[j]].cast<double>();
            }
            //TODO: initialize with color range in mind, to prevent close centers
            gmms[i].initialize(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(255, 255, 255));
            int iters = gmms[i].learn(data);
            if (!gmms[i].success()) {
                std::cout << "warning: gmm " << i << " failed to learn from " << n << "points" << std::endl;
            }
        }
    }

    std::cout << "initializing" << std::endl;
    //initialize
    for (int i=0; i<seed.rows; i++) {
        for (int j=0; j<seed.cols; j++) {
            int pos = j + i * seed.cols;
            if (seed.at<int>(i, j) < 0) {
                for (int dim=0; dim<=1; dim++) {
                    for (int dx=-1; dx<=1; dx+=2) {
                        Eigen::Array2i offset(0, 0);
                        offset[dim] += dx;
                        int pi = i + offset.y();
                        int pj = j + offset.x();
                        if (pi < 0 || pj < 0 || pi >= seed.rows || pj >= seed.cols) continue;
                        int neighbor = seed.at<int>(pi, pj);
                        if (neighbor >= 0) {
                            cv::Vec3b col0 = image.at<cv::Vec3b>(i, j);
                            cv::Vec3b col1 = image.at<cv::Vec3b>(pi, pj);
                            cv::Vec<float, 3> col0f(col0[0], col0[1], col0[2]);
                            cv::Vec<float, 3> col1f(col1[0], col1[1], col1[2]);
                            cv::Vec<float, 3> diff = col1f - col0f;
                            float cost = diff.dot(diff);
                            Eigen::RowVector3d col_vec(col0[0], col0[1], col0[2]);
                            if (gmm_weight > 0)
                                cost -= gmm_weight * static_cast<float>(gmms[cluster_indices[neighbor]].getLogLikelihood(col_vec)(0));
                            //if (cost < costs.at<float>(i, j)) {
                            //    costs.at<float>(i, j) = cost;
                                pq.push(std::make_pair(cost, std::make_pair(pos, neighbor)));
                            //}
                        }
                    }
                }
            }
        }
    }
    std::cout << "segmenting" << std::endl;
    //region growing
    while (!pq.empty()) {
        std::pair<int, int> pos_label = pq.top().second;
        pq.pop();
        int i = pos_label.first / seed.cols;
        int j = pos_label.first % seed.cols;
        int curr_label = seed.at<int>(i, j);
        if (curr_label < 0) {
            seed.at<int>(i, j) = pos_label.second;
            cv::Vec3b col0 = image.at<cv::Vec3b>(i, j);
            cv::Vec<float, 3> col0f(col0[0], col0[1], col0[2]);
            for (int dim=0; dim<=1; dim++) {
                for (int dx=-1; dx<=1; dx+=2) {
                    Eigen::Array2i offset(0, 0);
                    offset[dim] += dx;
                    int pi = i + offset.y();
                    int pj = j + offset.x();
                    if (pi < 0 || pj < 0 || pi >= seed.rows || pj >= seed.cols) continue;
                    int neighbor = seed.at<int>(pi, pj);
                    if (neighbor < 0) {
                        cv::Vec3b col1 = image.at<cv::Vec3b>(pi, pj);
                        cv::Vec<float, 3> col1f(col1[0], col1[1], col1[2]);
                        cv::Vec<float, 3> diff = col1f - col0f;
                        Eigen::RowVector3d col_vec(col1[0], col1[1], col1[2]);
                        float cost = diff.dot(diff);
                        if (gmm_weight > 0)
                            cost -= gmm_weight * static_cast<float>(gmms[cluster_indices[pos_label.second]].getLogLikelihood(col_vec)(0));
                        //if (cost < costs.at<float>(pi, pj)) {
                        //    costs.at<float>(pi, pj) = cost;
                            int newpos = pj + pi * seed.cols;
                            pq.push(std::make_pair(cost, std::make_pair(newpos, pos_label.second)));
                        //}
                    }
                }
            }
        }
    }
}