//
// Created by James Noeckel on 10/6/20.
//
#include <opencv2/opencv.hpp>
#include "imgproc/graph_cut.h"
#include "math/GaussianMixture.h"
/*#include <boost/filesystem.hpp>
#include <boost/regex.hpp>*/
#include "utils/timingMacros.h"


//#define WIDTH 64
//#define HEIGHT 64
#define MIN_PROB 1e-11f

const std::vector<std::vector<std::pair<int, bool>>> viewIndices = {{{1, false}, {2, false}, {7, false}, {13, false}, {14, false}},
                                                   {{34, false}, {35, false}, {44, true}, {45, true}, {54, false}, {55, false}, {56, false}, {64, true}, {65, true}, {66, true}},
                                                   {{19, false}, {35, false}, {44, true}, {45, true}, {54, false}, {55, false}, {56, false}, {64, true}, {65, true}, {66, true}},
                                                   {{31, true}, {39, false}, {40, false}, {49, true}, {50, true}, {51, true}, {59, false}, {60, false}, {61, false}, {69, true}}};
const std::vector<int> parts = {0, 1, 3, 4};

int main(int argc, char **argv) {
    using namespace Eigen;
    if (argc != 5) {
        std::cout << "usage: " << argv[0] << " smoothness sigma components variance" << std::endl;
        return 1;
    }
    float smoothness = std::stof(argv[1]);
    float sigma = std::stof(argv[2]);
    int components = std::stoi(argv[3]);
    float variance = std::stof(argv[4]);
    std::cout << "smoothness: " << smoothness << std::endl;
    std::cout << "sigma: " << sigma << std::endl;
    std::cout << "components: " << components << std::endl;
    std::cout << "variance: " << variance << std::endl;
    for (int partInd=0; partInd<parts.size(); ++partInd) {
        std::vector<std::string> imageFnames;
        std::vector<std::string> occlusionMaskFnames;
        int partId = parts[partInd];
        for (auto pair : viewIndices[partInd]) {
            imageFnames.push_back("../test_data/segmentation2/part_" + std::to_string(partId) + "_view_" +
                                  std::to_string(pair.first) + (pair.second ? "_backside" : "") + "_warped.png");
            occlusionMaskFnames.push_back("../test_data/segmentation2/part_" + std::to_string(partId) + "_view_" +
                                          std::to_string(pair.first) + (pair.second ? "_backside" : "") +
                                          "_mask_warped_eroded.png");
        }

        /*or (int partId : parts) {
            boost::filesystem::directory_iterator end_itr; // Default ctor yields past-the-end
            const boost::regex img_filter( "part_" + std::to_string(partId) + "_view_\\d+_warped\\.png" );
            const boost::regex mask_filter( "part_" + std::to_string(partId) + "_view_\\d+_mask_warped_eroded\\.png" );
            for (boost::filesystem::directory_iterator i("../test_data/segmentation/"); i != end_itr; ++i) {
                // Skip if not a file
                if (!boost::filesystem::is_regular_file(i->status())) continue;
                if (i->path().extension() != ".png") continue;

                boost::smatch what;
                if (boost::regex_match(i->path().filename().string(), what, img_filter)) {
                    imageFnames.push_back(i->path().string());
                }
                if (boost::regex_match(i->path().filename().string(), what, mask_filter)) {
                    occlusionMaskFnames.push_back(i->path().string());
                }
            }
            if (imageFnames.size() != occlusionMaskFnames.size()) {
                std::cout << "unequal file counts for part " << partId << " (" << imageFnames.size() << "-" << occlusionMaskFnames.size() << std::endl;
                return 1;
            }
            if (imageFnames.empty()) {
                std::cout << "no files matched" << std::endl;
                return 1;
            }
            std::sort(imageFnames.begin(), imageFnames.end());
            std::sort(occlusionMaskFnames.begin(), occlusionMaskFnames.end());*/

        cv::Mat initMask = cv::imread("../test_data/segmentation2/part_" + std::to_string(partId) + "_initMask.png",
                                      cv::IMREAD_GRAYSCALE) / 255;
        if (initMask.empty()) {
            std::cout << "failed to load initMask " << partId << std::endl;
            return 1;
        }

        std::vector<cv::Mat> images;
        std::vector<cv::Mat> occlusionMasks;

        for (int index = 0; index < imageFnames.size(); ++index) {
            images.push_back(cv::imread(imageFnames[index]));
            cv::cvtColor(images.back(), images.back(), cv::COLOR_BGR2Luv, 0);
            occlusionMasks.push_back(cv::imread(occlusionMaskFnames[index], cv::IMREAD_GRAYSCALE) / 255);
            if (images.back().empty() || occlusionMasks.back().empty()) {
                std::cout << "failed to load image or mask " << imageFnames[index] << std::endl;
                return 1;
            }
            if (images.back().rows != initMask.rows || images.back().cols != initMask.cols) {
                std::cout << "image and mask sizes do not match: " << images.back().size << " vs " << initMask.size
                          << std::endl;
                return 1;
            }
            if (occlusionMasks.back().rows != initMask.rows || occlusionMasks.back().cols != initMask.cols) {
                std::cout << "mask sizes do not match: " << occlusionMasks.back().size << " vs " << initMask.size
                          << std::endl;
                return 1;
            }
        }
        for (int iteration = 0; iteration < 2; ++iteration) {
            std::vector<std::vector<GaussianMixture>> gmms(images.size(), std::vector<GaussianMixture>(2));
            for (size_t i = 0; i < images.size(); ++i) {
                for (int cluster = 0; cluster <= 1; ++cluster) {
                    const auto &img = images[i];
                    std::vector<int> indices;
                    indices.reserve(img.total());
                    //add all pixels that are both inside the known subset and unoccluded in this view to the training set
                    for (int p = 0; p < img.total(); ++p) {
                        if (initMask.at<uchar>(p) == cluster && occlusionMasks[i].at<uchar>(p)) {
                            indices.push_back(p);
                        }
                    }
                    size_t n = indices.size();
                    std::cout << "learning view " << i << " cluster " << cluster << " with " << n << " points"
                              << std::endl;
                    MatrixX3d colors(n, 3);
                    for (size_t j = 0; j < n; ++j) {
                        int p = indices[j];
                        const auto &col = img.at<cv::Vec3b>(p);
                        colors.row(j) = RowVector3d(col[0], col[1], col[2]);
                    }
                    gmms[i][cluster] = GaussianMixture(components, 3, variance);
                    int iters = gmms[i][cluster].learn(colors);
                    /*if (!gmms[i][cluster].success()) {
                        std::cout << "warning: gmm " << i << " cluster " << cluster << " failed to learn from " << n
                                  << " points after " << iters << " iterations" << std::endl;
                        cv::imshow("mask", occlusionMasks[i]);
                        cv::waitKey();
                        cv::destroyWindow("mask");
    //                    std::cout << initMask << std::endl;
                        cv::imshow("initMask", initMask == cluster);
                        cv::waitKey();
                        cv::destroyWindow("initMask");
                    }*/
                }
            }
            std::vector<cv::Mat> avgProbImgs(2);
            avgProbImgs[0] = cv::Mat::zeros(initMask.rows, initMask.cols, CV_32FC1);
            avgProbImgs[1] = cv::Mat::zeros(initMask.rows, initMask.cols, CV_32FC1);
            cv::Mat viewCount = cv::Mat::ones(initMask.rows, initMask.cols, CV_32FC1) * 0.0001;
            for (size_t imgInd = 0; imgInd < images.size(); ++imgInd) {
                std::vector<cv::Mat> debugImgs(2);
                debugImgs[0] = cv::Mat::zeros(initMask.rows, initMask.cols, CV_32FC1);
                debugImgs[1] = cv::Mat::zeros(initMask.rows, initMask.cols, CV_32FC1);
                for (int i = 0; i < initMask.rows; ++i) {
                    for (int j = 0; j < initMask.cols; ++j) {
                        if (occlusionMasks[imgInd].at<uchar>(i, j)) {
                            for (int cluster = 0; cluster <= 1; ++cluster) {
                                if (!gmms[imgInd][cluster].success()) continue;
                                const auto &col = images[imgInd].at<cv::Vec3b>(i, j);
                                RowVector3d color(col[0], col[1], col[2]);
                                float logp = static_cast<float>(gmms[imgInd][cluster].logp_data(color)(0));
                                float prob = std::exp(logp);
                                if (prob > 1) {
                                    std::cout << "warning: probability " << prob << " at " << i << ", " << j << ": "
                                              << cluster << std::endl;
                                }
                                debugImgs[cluster].at<float>(i, j) = prob;
                                avgProbImgs[cluster].at<float>(i, j) += prob;
                                viewCount.at<float>(i, j) += 1;
                            }
                        }
                    }
                }
                for (int cluster = 0; cluster <= 1; ++cluster) {
                    if (!gmms[imgInd][cluster].success()) continue;
                    cv::Mat debugImgConverted;
                    double minVal;
                    double maxVal;
                    cv::Point minLoc;
                    cv::Point maxLoc;
                    cv::minMaxLoc(debugImgs[cluster], &minVal, &maxVal, &minLoc, &maxLoc);
                    debugImgs[cluster].convertTo(debugImgConverted, CV_8UC1, 255.0 / maxVal);
                    for (int pix = 0; pix < initMask.total(); ++pix) {
                        if (!occlusionMasks[imgInd].at<uchar>(pix)) {
                            debugImgConverted.at<uchar>(pix) = 127;
                        }
                    }
                    cv::imwrite("part_" + std::to_string(partId) + "_prob_" + std::to_string(cluster) + "_view_" +
                                std::to_string(imgInd) + "_iteration_" + std::to_string(iteration) + ".png", debugImgConverted);
                }
            }
            avgProbImgs[0] /= viewCount;
            avgProbImgs[1] /= viewCount;

            for (int cluster = 0; cluster <= 1; ++cluster) {
                cv::Mat probImgConverted;
                double minVal;
                double maxVal;
                cv::Point minLoc;
                cv::Point maxLoc;
                cv::minMaxLoc(avgProbImgs[cluster], &minVal, &maxVal, &minLoc, &maxLoc);
                std::cout << "minval " << cluster << ": " << minVal << "; maxval: " << maxVal << std::endl;
                std::cout << "view count at maxval: " << viewCount.at<float>(maxLoc) << std::endl;
                avgProbImgs[cluster].convertTo(probImgConverted, CV_8UC1, 255.0 / maxVal);
                cv::imwrite("part_" + std::to_string(partId) + "_prob_" + std::to_string(cluster) + "_iteration_" + std::to_string(iteration) + ".png",
                            probImgConverted);
            }

            cv::Mat energy(initMask.rows, initMask.cols, CV_32FC2);
            for (int row = 0; row < initMask.rows; ++row) {
                for (int col = 0; col < initMask.cols; ++col) {
                    float energy0;
                    float energy1 = -std::log(std::max(MIN_PROB, avgProbImgs[1].at<float>(row, col)));
                    if (row == 0 || col == 0 || row == initMask.rows - 1 || col == initMask.cols - 1) {
                        energy0 = -1000000 * std::log(MIN_PROB);
                    } else {
                        energy0 = -std::log(std::max(MIN_PROB, avgProbImgs[0].at<float>(row, col)));
                    }
                    energy.at<cv::Vec<float, 2>>(row, col) = cv::Vec<float, 2>(energy0, energy1);
                }
            }
            cv::Mat labels(initMask.rows, initMask.cols, CV_8UC1);
            std::cout << "running maxflow" << std::endl;
            DECLARE_TIMING(flow);
            START_TIMING(flow);
//        float maxflow = graph_cut(energy, labels, smoothness, avgProbImgs, sigma);
            float maxflow = graph_cut(energy, labels, smoothness, images, sigma, occlusionMasks);
//        float maxflow = graph_cut(energy, labels, smoothness);
            STOP_TIMING(flow);
            PRINT_TIMING(flow);
            std::cout << "solution with " << maxflow << " flow" << std::endl;
            cv::imwrite("part_" + std::to_string(partId) + "_graphCutOutput_iteration_" + std::to_string(iteration) + ".png", labels * 255);
            initMask = labels;
        }
    }
    return 0;

    /*cv::Mat energy(HEIGHT, WIDTH, CV_32FC2);
    cv::Mat labels(HEIGHT, WIDTH, CV_8UC1);
    for (int i=0; i<HEIGHT; ++i) {
        for (int j=0; j<WIDTH; ++j) {
            float probi = std::max(0.0f, (std::abs(i - HEIGHT/2.0f) - WIDTH / 8.0f)/WIDTH*4);
            float probj = std::max(0.0f, (std::abs(j - HEIGHT/2.0f) - WIDTH / 8.0f)/WIDTH*4);
            float prob = std::max(probi, probj);
            energy.at<cv::Vec<float, 2>>(i, j) = cv::Vec<float, 2>(prob, 1.0f-prob);
        }
    }
    float maxflow = graph_cut(energy, labels, 5);
    std::cout << "solution with " << maxflow << " flow" << std::endl;
    cv::imwrite("graphCutTest.png", labels*255);
    return 0;*/
}