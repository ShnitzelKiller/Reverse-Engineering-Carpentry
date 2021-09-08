//
// Created by James Noeckel on 1/29/20.
//

#include "multilabel_graph_cut.h"
#include "imgproc/multilabel/GCoptimization.h"
#include <Eigen/Dense>
#include <numeric>
#include <random>
#include "math/GaussianMixture.h"

#define TRAINING_SIZE 10000
#define MIN_VARIANCE 4

struct SmoothData {
    float sigma;
    uchar *data;
    float weight;
};

int smooth_cost_fn(int p1, int p2, int l1, int l2, void* data) {
    if (l1 == l2) {
        return 0;
    } else {
        auto myData = (SmoothData *) data;
        float diff0 = static_cast<float>(static_cast<short>(*(myData->data + p1 * 3)) -
                                         static_cast<short>(*(myData->data + p2 * 3)));
        float diff1 = static_cast<float>(static_cast<short>(*(myData->data + p1 * 3 + 1)) -
                                         static_cast<short>(*(myData->data + p2 * 3 + 1)));
        float diff2 = static_cast<float>(static_cast<short>(*(myData->data + p1 * 3 + 2)) -
                                         static_cast<short>(*(myData->data + p2 * 3 + 2)));
        float diff = diff0 * diff0 + diff1 * diff1 + diff2 * diff2;
        return static_cast<int>(std::round(myData->weight * exp(-diff / (2 * myData->sigma * myData->sigma))));
    }
}

void multilabel_graph_cut(const cv::Mat &image, cv::Mat &labels, float data_weight, float smoothness_weight, float label_penalty, float sigma, int components, int iterations, int levels) {

    std::unordered_map<int, int> cluster_to_index;
    std::vector<int> clusters;
    int index=0;
    for (int i=0; i<labels.rows; i++) {
        for (int j=0; j<labels.cols; j++) {
            int cluster = labels.at<int>(i,j);
            if (cluster >= 0) {
                auto it = cluster_to_index.find(cluster);
                if (it == cluster_to_index.end()) {
                    cluster_to_index[cluster] = index;
                    clusters.push_back(cluster);
                    index++;
                }
            }
        }
    }

    int num_labels = clusters.size();
    int num_pixels = labels.total();

    // precompute Gaussian pyramid
    std::vector<cv::Mat> pyramid;
    pyramid.reserve(levels);
    pyramid.push_back(image);
//    if (levels > 1) {
//        std::cout << "building Gaussian pyramid with " << levels << " levels" << std::endl;
//    }
    for (int i=0; i<levels-1; i++) {
        int ksize = (1 << i)*2 + 1;
        cv::Mat blurred;
        cv::GaussianBlur(image, blurred, cv::Size(ksize, ksize), 0);
        pyramid.push_back(blurred);
    }

    std::vector<std::vector<Eigen::RowVectorXi>> colors(num_labels);
    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            int cluster = labels.at<int>(i, j);
            if (cluster >= 0) {
                std::vector<Eigen::RowVectorXi> &color_c = colors[cluster_to_index[cluster]];
                color_c.emplace_back(3*levels);
                for (int l=0; l<levels; l++) {
                    cv::Vec3b col = pyramid[l].at<cv::Vec3b>(i, j);
                    color_c[color_c.size()-1].segment(l*3, 3) = Eigen::RowVector3i(col[0], col[1], col[2]);
                }
            }
        }
    }

    std::vector<GaussianMixture> gmms;
    //std::vector<int> sizes;
    gmms.reserve(num_labels);
    //sizes.reserve(num_labels);
    for (int i = 0; i < num_labels; i++) {
        int full_n = colors[i].size();
        int n = std::min(TRAINING_SIZE, full_n);
        Eigen::MatrixXd data(n, 3*levels);
        if (n < full_n) {
            std::vector<int> indices(full_n);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
            for (int j = 0; j < n; j++) {
                data.row(j) = colors[i][indices[j]].cast<double>();
            }
        } else {
            for (int j=0; j < n; j++) {
                data.row(j) = colors[i][j].cast<double>();
            }
        }

        Eigen::RowVectorXd mean = data.colwise().mean();
        /*Eigen::MatrixXd cov = (data.rowwise()-mean).transpose() * (data.rowwise()-mean);
        auto cholesky = cov.ldlt();*/
        gmms.emplace_back(components, 3*levels, MIN_VARIANCE);
        //sizes.push_back(full_n);
        bool bypass_EM = false;
        /*if (cholesky.info() == Eigen::NumericalIssue || cholesky.vectorD().mean() < MIN_VARIANCE) {
            std::cout << "detected ill-posed data of size "<< n <<" for gmm " << i << " (" << clusters[i] << "); using default single gaussian" << std::endl;
            bypass_EM = true;
        }*/
        //if (!bypass_EM) {
            //gmms[i].initialize(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(255, 255, 255));
            //int init_iters = gmms[i].initialize_k_means(data);
            //if (init_iters == 0) {
            //    std::cout << "warning: gmm " << i << " (" << clusters[i] << ") failed to initialize from " << n << " points; using default single gaussian" << std::endl;
            //} else {
                int iters = gmms[i].learn(data);
                if (!gmms[i].success()) {
                    std::cout << "warning: gmm " << i << " (" << clusters[i] << ") failed to learn from " << n
                              << " points; using default single gaussian" << std::endl;
                    bypass_EM = true;
                }
            //}
        //}
        /*std::cout << "data " << i << " data covariance: " << std::endl << cov << std::endl;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigs(cov);
        std::cout << "eigenvalues: " << eigs.eigenvalues().transpose() << std::endl;*/
        if (bypass_EM) {
            std::vector<Eigen::MatrixXd> cov_default(1, Eigen::MatrixXd::Identity(3*levels, 3*levels) *
                                                        MIN_VARIANCE); //TODO: come up with a default variance
            Eigen::Matrix<double, 1, 1> pi(1);
            gmms[i].initialize(mean, cov_default, pi);
            if (!gmms[i].useCurrentModel()) {
                std::cout << "failed to set model" << std::endl;
                return;
            }
        }
    }

    int *result = new int[num_pixels];   // stores result of optimization

    try {
        auto gc = new GCoptimizationGridGraph(labels.cols, labels.rows, num_labels);
        //set up data costs
        //std::cout << "setting up costs" << std::endl;
        for ( int i = 0; i < num_pixels; i++ ) {
            int col = i % labels.cols;
            int row = i / labels.cols;
            cv::Point point(col, row);
            int cluster = labels.at<int>(point);
            int l_gt = cluster < 0 ? -1 : cluster_to_index[cluster];
            Eigen::RowVectorXd col_vec(3*levels);
            for (int l=0; l<levels; l++) {
                cv::Vec3b color = pyramid[l].at<cv::Vec3b>(point);
                col_vec.segment(3*l, 3) = Eigen::RowVector3d(color[0], color[1], color[2]);
            }

            for (int l = 0; l < num_labels; l++) {

                if (cluster < 0) {
                    float logl = /*(static_cast<float>(sizes[l])/num_pixels) * */ static_cast<float>(gmms[l].logp_data(
                            col_vec)(0));
                    gc->setDataCost(i, l, -data_weight * logl);
                } else {
                    if (l == l_gt) {
                        gc->setDataCost(i, l, 0);
                    } else {
                        gc->setDataCost(i, l, label_penalty);
                    }
                }
            }
        }
        SmoothData data;
        data.data = image.data;
        data.sigma = sigma;
        data.weight = smoothness_weight;
        gc->setSmoothCost(&smooth_cost_fn, &data);
        //printf("\nBefore optimization energy is %lld",gc->compute_energy());
        gc->expansion(iterations);// run expansion for some iterations. For swap use gc->swap(num_iterations);
        //printf("\nAfter optimization energy is %lld\n",gc->compute_energy());
        for ( int  i = 0; i < num_pixels; i++ )
            result[i] = gc->whatLabel(i);
        delete gc;
    } catch (GCException &e) {
        e.Report();
    }

    for ( int i = 0; i < num_pixels; i++ ) {
        int col = i % labels.cols;
        int row = i / labels.cols;
        cv::Point point(col, row);
        labels.at<int>(point) = clusters[result[i]];
    }
    delete[] result;
}