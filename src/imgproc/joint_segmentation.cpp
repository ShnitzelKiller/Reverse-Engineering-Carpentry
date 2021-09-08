//
// Created by James Noeckel on 10/1/20.
//

#include "joint_segmentation.h"
#include "matching_scores.h"
#include <Eigen/Dense>
#include "math/GaussianMixture.h"
#include <numeric>

#define MIN_CORR 0.3
#define MIN_VARIANCE 10.0

uint DEBUG_ID = 0;


void joint_segmentation(const std::vector<cv::Mat> &allImages, const std::vector<cv::Mat> &allMasks, cv::Mat &labels,
                        const Settings &settings, const std::vector<std::vector<int>> &groups,
                        std::mt19937 random_engine) {
    using namespace Eigen;
    int num_pixels = labels.total();
    std::vector<std::vector<int>> indexMap = groups;
    if (indexMap.empty()) {
        indexMap.emplace_back();
        indexMap.back().resize(allImages.size());
        std::iota(indexMap.back().begin(), indexMap.back().end(), 0);
    }
    std::vector<GaussianMixture> gmms(allImages.size());
    for (int imageInd = 0; imageInd < allImages.size(); ++imageInd) {
        const cv::Mat &img = allImages[imageInd];
        std::vector<int> indices;
        indices.reserve(img.total());
        //add all pixels that are both inside the known subset and unoccluded in this view to the training set
        for (int p = 0; p < img.total(); ++p) {
            if (labels.at<uchar>(p) == 0 && allMasks[imageInd].at<uchar>(p)) {
                indices.push_back(p);
            }
        }
        size_t n = indices.size();
        MatrixX3d colors(n, 3);
        for (size_t j = 0; j<n; ++j) {
            int p = indices[j];
            const auto &col = img.at<cv::Vec3b>(p);
            colors.row(j) = RowVector3d(col[0], col[1], col[2]);
        }
        gmms[imageInd] = GaussianMixture(settings.segmentation_gmm_components, 3, MIN_VARIANCE);
        int iters = gmms[imageInd].learn(colors);
        if (!gmms[imageInd].success()) {
            std::cout << "warning: gmm " << imageInd << " failed to learn from " << n
                      << " points after " << iters << " iterations" << std::endl;
        }
    }

    cv::Mat debugAvgProb = cv::Mat::zeros(labels.rows, labels.cols, CV_32FC1);
    cv::Mat debugCount = cv::Mat::ones(labels.rows, labels.cols, CV_32FC1) * 0.0001;
    for (int imgInd = 0; imgInd < allImages.size(); ++imgInd) {
        float maxProb = 0.0f;
        if (gmms[imgInd].success()) {
            cv::Mat debugGMMImg(labels.rows, labels.cols, CV_32FC1);
            for (int i = 0; i < labels.rows; ++i) {
                for (int j = 0; j < labels.cols; ++j) {
                    //RunningAverage avgProb(1);
                    const auto &col = allImages[imgInd].at<cv::Vec3b>(i, j);
                    RowVector3d color(col[0], col[1], col[2]);
                    double logp = gmms[imgInd].logp_data(color)(0);
                    double prob = std::exp(logp);
                    //avgProb.add(prob);
                    maxProb = std::max(maxProb, static_cast<float>(prob));
                    debugGMMImg.at<float>(i, j) = prob;
                    if (allMasks[imgInd].at<uchar>(i, j)) {
                        debugAvgProb.at<float>(i, j) += prob;
                        debugCount.at<float>(i, j) += 1;
                    }
                }
            }
            //convert to viewable image
            float scaleFac = 255.0f / maxProb;
            cv::Mat debugGMMImg2;
            debugGMMImg.convertTo(debugGMMImg2, CV_8UC1, scaleFac);
            cv::imwrite("debug_img_" + std::to_string(DEBUG_ID) + "_view_" + std::to_string(imgInd) + ".png", debugGMMImg2);
        }
    }
    debugAvgProb /= debugCount;
    cv::Mat debugAvgProb2;
    double minVal;
    double maxVal;
    cv::Point minLoc;
    cv::Point maxLoc;
    cv::minMaxLoc( debugAvgProb, &minVal, &maxVal, &minLoc, &maxLoc );

    debugAvgProb.convertTo(debugAvgProb2, CV_8UC1, 255.0 / maxVal);
    cv::imwrite("debug_img_" + std::to_string(DEBUG_ID) + "_avg_scale" + std::to_string(maxVal) + ".png", debugAvgProb2);

    DEBUG_ID++;

    //cv::Mat energies(labels.rows, labels.cols, CV_32FC2);

    /*auto result = std::make_unique<int[]>(num_pixels);   // stores result of optimization
    try {
        auto gc = new GCoptimizationGridGraph(labels.cols, labels.rows, 2);
        std::cout << "setting up energy function" << std::endl;
        for ( int i = 0; i < num_pixels; i++ ) {
            int col = i % labels.cols;
            int row = i / labels.cols;
//            std::cout << "row " << row << ", col " << col << std::endl;
            cv::Point point(col, row);
            uchar initCluster = labels.at<uchar>(point);
            //TODO: set smoothness costs
            //TODO: incorporate initial guess
            float totalNCC = 0.0f;
            size_t count = 0;
//            std::cout << "computing frontal scores" << std::endl;
            for (int img1 = 0; img1 < frontImages.size(); ++img1) {
                if (frontMasks[img1].at<uchar>(row, col) == 0) continue;
//                std::cout << "image1: " << img1 << " (" << frontImages[img1].rows << ", " << frontImages[img1].cols << ") and (" << frontMasks[img1].rows << ", " << frontMasks[img1].cols << ')' << std::endl;
                for (int img2 = img1+1; img2 < frontImages.size(); ++img2) {
                    if (frontMasks[img2].at<uchar>(row, col) == 0) continue;
//                    std::cout << "image2: " << img2 << " (" << frontImages[img2].rows << ", " << frontImages[img2].cols << ") and (" << frontMasks[img2].rows << ", " << frontMasks[img2].cols << ')' << std::endl;
                    float NCC = norm_cross_correlation(frontImages[img1], frontImages[img2], row, col, row, col, radius);
                    totalNCC += NCC;
                    ++count;
                }
            }
//            std::cout << "computing back scores" << std::endl;
            for (int img1 = 0; img1 < backImages.size(); ++img1) {
                if (backMasks[img1].at<uchar>(row, col) == 0) continue;
                for (int img2 = img1+1; img2 < backImages.size(); ++img2) {
                    if (backMasks[img2].at<uchar>(row, col) == 0) continue;
                    float NCC = norm_cross_correlation(backImages[img1], backImages[img2], row, col, row, col, radius);
                    totalNCC += NCC;
                    ++count;
                }
            }
//            std::cout << "setting data score" << std::endl;
            totalNCC /= count;
            totalNCC = (totalNCC + 1)/2.0f;
            //DEBUG: set mask to NCC score
            labels.at<uchar>(row, col) = static_cast<uchar>(std::round(std::max(0.0f, std::min(255.0f, totalNCC * 255))));
            //between 0 and 1
            gc->setDataCost(i, 1, (1.0f-totalNCC) * ENERGY_RESOLUTION);
            gc->setDataCost(i, 0, (1.0f-MIN_CORR) * ENERGY_RESOLUTION);
        }
        cv::imwrite("debug_img" + std::to_string(DEBUG_ID++) + ".png", labels);
        std::cout << "expansion... ";
        gc->expansion(1);
        std::cout << "done" << std::endl;
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
        labels.at<uchar>(point) = result[i];
    }*/
}
