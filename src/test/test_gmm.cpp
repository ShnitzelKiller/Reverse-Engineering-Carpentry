//
// Created by James Noeckel on 1/22/20.
//

#include "math/GaussianMixture.h"
#include <opencv2/opencv.hpp>
#include <ctime>
#include "math/NormalRandomVariable.h"
#include <random>
#include "utils/color_conversion.hpp"

#define trials 10
#define width 500
#define height 500

void display_gmm(const Eigen::Ref<const Eigen::MatrixX2d> &data, const GaussianMixture &gmm, const std::string &filename, std::mt19937 &gen) {
    int k = gmm.getNumComponents();
    //std::uniform_real_distribution<double> hue_dist(0, 360);
    //std::uniform_real_distribution<double> sv_dist(0.5, 1);
    Eigen::MatrixX3d colors(k, 3);
    for (int i=0; i<k; i++) {
        double r, g, b;
        hsv2rgb<double>((360*i)/static_cast<double>(k), 1.0, 1.0, r, g, b);
        colors.row(i) = 255 * Eigen::RowVector3d(r, g, b);
    }

    cv::Mat mask = cv::Mat::zeros(height, width, CV_8UC3);
    Eigen::MatrixX2i coords(width * height, 2);
    for (int i = 0; i < width * height; i++) {
        coords(i, 0) = i / width;
        coords(i, 1) = i % width;
    }
    Eigen::VectorXd total_log_likelihoods = gmm.logp_data(coords.cast<double>());
    double max_log_likelihood = total_log_likelihoods.maxCoeff();

    for (int i = 0; i < width * height; i++) {
        auto color = static_cast<unsigned char>(255 * exp(total_log_likelihoods(i) - max_log_likelihood));
        mask.at<cv::Vec3b>(coords(i, 0), coords(i, 1)) = cv::Vec3b(color, color, color);
    }
    Eigen::MatrixXd log_likelihoods = gmm.logp_z_given_data(data);
    for (int i = 0; i < data.rows(); i++) {
        int j;
        log_likelihoods.row(i).maxCoeff(&j);
        cv::Point2i point(static_cast<int>(data(i, 1)), static_cast<int>(data(i, 0)));
        if (point.x >= 0 && point.x < width && point.y >= 0 && point.y < height) {
            mask.at<cv::Vec3b>(point) = cv::Vec3b(colors(j, 0), colors(j, 1), colors(j, 2));
        }
    }
    cv::imwrite(filename, mask);
}

GaussianMixture test_gmm(bool use_kmeans, Eigen::MatrixX2d &out_data, bool print_comparison, std::mt19937 &gen, int k=4) {
    std::uniform_real_distribution<double> pos_dist(static_cast<double>(width) / 8, static_cast<double>(width) * (7.0 / 8.0));
    std::uniform_real_distribution<double> var_dist(10, 4000);
    std::uniform_real_distribution<double> angdist(0, M_PI_4);
    GaussianMixture gmm;
    gmm.setNumComponents(k);
    int n = 2000;
    std::cout << "generating data..." << std::endl;

    Eigen::MatrixX2d data(n, 2);
    Eigen::MatrixX2d means(k, 2);
    for (int i=0; i<k; i++) {
        means.row(i) = Eigen::RowVector2d(pos_dist(gen), pos_dist(gen));
    }

    std::vector<Eigen::Matrix2d> sigmas(k);
    for (int i=0; i<k; i++) {
        Eigen::Array2d eigenvalues(var_dist(gen), var_dist(gen));
        double angle = angdist(gen);
        Eigen::Matrix2d rot;
        rot << cos(angle), -sin(angle),
               sin(angle), cos(angle);
        sigmas[i] = (rot.array().rowwise() * eigenvalues.transpose()).matrix() * rot.transpose();
    }
    std::vector<NormalRandomVariable> randvs;
    randvs.reserve(k);
    for (int i=0; i<k; i++) {
        randvs.emplace_back(means.row(i).transpose(), sigmas[i]);
    }
    std::uniform_int_distribution<int> cluster_dist(0, k-1);
    for (int i = 0; i < n; i++) {
        int cluster = cluster_dist(gen);
        data.row(i) = randvs[cluster]().transpose();
    }
    std::cout << "generated data " << std::endl;

    std::cout << "GMM parameter estimation..." << std::endl;
    float time_avg = 0.0f;
    int total_iters = 0;
    int total_successes = 0;
    for (int i = 0; i < trials; i++) {
        gmm.clear();
        if (!use_kmeans) {
            gmm.initialize_random_means(data);
        }
        auto start_t = clock();
        int iters = gmm.learn(data);
        total_iters += iters;
        auto total_t = clock() - start_t;
        float time_sec = static_cast<float>(total_t) / CLOCKS_PER_SEC;
        time_avg += time_sec;
        if (gmm.success()) {
            total_successes += 1;
        }
    }
    time_avg /= trials;
    std::cout << "average time: " << time_avg << std::endl;
    std::cout << "average iters: " << static_cast<float>(total_iters) / static_cast<float>(trials) << std::endl;
    std::cout << "total successes: " << total_successes << '/' << trials << std::endl;
    if (print_comparison) {
        std::vector<int> closest(k);
        std::vector<bool> chosen(k, false);
        for (int i=0; i<k; i++) {
            int ind = -1;
            double mindist = std::numeric_limits<double>::max();
            for (int j=0; j<k; j++) {
                if (!chosen[j]) {
                    double d2 = (gmm.means().row(i) - means.row(j)).squaredNorm();
                    if (d2 < mindist) {
                        ind = j;
                        mindist = d2;
                    }
                }
            }
            closest[i] = ind;
            chosen[ind] = true;
        }
        std::cout << "---- parameter comparison ----" << std::endl
                  << "means:                     ground truth:" << std::endl;
        for (int i = 0; i < k; i++) {
            std::cout << '[' << gmm.means().row(i) << "] -------- [" << means.row(closest[i]) << ']' << std::endl;
        }
        std::cout << "covariances:               ground truth:" << std::endl;
        std::cout << "------------------------------------" << std::endl;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < 2; j++) {
                std::cout << '|' << gmm.sigmas()[i].row(j) << "| -------- |" << sigmas[closest[i]].row(j) << '|' << std::endl;
            }
            std::cout << "------------------------------------" << std::endl;
        }
    }
    out_data = std::move(data);
    return gmm;
}

void show_eigenvalues(const GaussianMixture &gmm) {
    for (int i=0; i< gmm.getNumComponents(); i++) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(gmm.sigmas()[i]);
        std::cout << "eigenvalues for cluster " << i << ": [" << eig.eigenvalues().transpose() << "]; mean: [" << gmm.means().row(i) << ']' << std::endl;
        std::cout << "eigenvectors:\n" << eig.eigenvectors() << std::endl;
    }
}

int main(int argc, char **argv) {
    int k = 4;
    if (argc > 1) {
        try {
            k = std::stoi(argv[1]);
        } catch (std::invalid_argument &e) {
            std::cout << "usage: " << argv[0] << " [num_components]" << std::endl;
            return 1;
        }
    }
    auto seed = (unsigned int) time(nullptr);
    srand(seed);
    std::mt19937 gen(std::random_device{}());
    //unsigned int seed = 1579761920;
    std::cout << "seed: " << seed << std::endl;
    {
        std::cout << "=======testing without kmeans=======" << std::endl;
        Eigen::MatrixX2d data;
        test_gmm(false, data, false, gen, k);
        std::cout << "=======testing with kmeans=======" << std::endl;
        GaussianMixture gmm = test_gmm(true, data, true, gen, k);
        display_gmm(data, gmm, "test_gmm_1.png", gen);
    }

    {
        std::cout << "=========== testing 1D subspaces setting min variance to 2 ===========" << std::endl;
        GaussianMixture gmm(3, 2, 2);
        Eigen::MatrixX2d data(20, 2);
        for (int i=0; i<10; i++) {
            data.row(i) = Eigen::RowVector2d(200+i*10, 200+i*15);
            data.row(i+10) = Eigen::RowVector2d(width-i*30, 200 + i*10);
        }
        int iters = gmm.learn(data);
        if (!gmm.success()) {
            std::cout << "learning on disjoint linear subspace data failed" << std::endl;
            return 1;
        } else {
            std::cout << "succeeded in " << iters << " iterations" << std::endl;
        }
        show_eigenvalues(gmm);
        display_gmm(data, gmm, "test_gmm_2.png", gen);
    }

    {
        std::cout << "=========== testing high degeneracy ===========" << std::endl;
        GaussianMixture gmm;
        gmm.setNumComponents(3);
        gmm.setMinEigenvalue(0.001);
        Eigen::MatrixXd data = Eigen::MatrixXd::Zero(50, 2);
        data.row(4) = Eigen::RowVector2d(1, 1);
        Eigen::MatrixXd data_extrapoint(data);
        data_extrapoint.row(19) = Eigen::RowVector2d(.01, -.01);
        data *= 0.01;
        int initialized = gmm.initialize_k_means(data);
        if (initialized != 0) {
            std::cout << "init should have failed" << std::endl;
            return 1;
        }

        initialized = gmm.initialize_k_means(data_extrapoint);
        if (!initialized) {
            std::cout << "initialization failed" << std::endl;
            return 1;
        }
        int iters = gmm.learn(data_extrapoint);
        if (!gmm.success()) {
            std::cout << "learning on degenerate data failed" << std::endl;
            return 1;
        }
        std::cout << "finished; k-means initialization iters: " << initialized << "; learning iters: " << iters << std::endl;
        show_eigenvalues(gmm);
    }
    return 0;
}