//
// Created by James Noeckel on 10/13/20.
//
#include "reconstruction/ReconstructionData.h"
#include <opencv2/opencv.hpp>
using namespace Eigen;

void saveEpipolarLines(ReconstructionData &reconstruction) {
    for (auto &pair : reconstruction.images) {
        Vector3d origin1 = pair.second.origin();
        double minL2 = std::numeric_limits<double>::max();
        int matchImgInd = -1;
        for (const auto &pair2 : reconstruction.images) {
            if (pair.first != pair2.first) {
                Vector3d origin2 = pair2.second.origin();
                double L2 = (origin1 - origin2).squaredNorm();
                if (L2 < minL2) {
                    minL2 = L2;
                    matchImgInd = pair2.first;
                }
            }
        }
        Vector3d center(0, 0, 0);
        Vector2d midpoint1 = reconstruction.project(center.transpose(), pair.first).transpose();
        Vector2d midpoint2 = reconstruction.project(center.transpose(), matchImgInd).transpose();
//        Vector2d midpoint1 = reconstruction.resolution(pair.first) * 0.5;
//        Vector2d midpoint2 = reconstruction.resolution(matchImgInd) * 0.5;
        Vector2d epipolarLine1 = reconstruction.epipolar_line(midpoint1.y(), midpoint1.x(),
                                                              pair.first, matchImgInd);
        Vector2d epipolarLine2 = reconstruction.epipolar_line(midpoint2.y(), midpoint2.x(),
                                                              matchImgInd, pair.first);
        Vector2d startpoint1 = midpoint1 - epipolarLine1 * midpoint1.x();
        Vector2d otherpoint1 = midpoint1 + epipolarLine1 * midpoint1.x();
        Vector2d startpoint2 = midpoint2 - epipolarLine2 * midpoint2.x();
        Vector2d otherpoint2 = midpoint2 + epipolarLine2 * midpoint2.x();
        cv::Mat img1 = pair.second.getImage().clone();
        cv::line(img1, cv::Point(startpoint1.x(), startpoint1.y()), cv::Point(otherpoint1.x(), otherpoint1.y()), cv::Scalar(255, 100, 255), 2);
        cv::circle(img1, cv::Point(midpoint1.x(), midpoint1.y()), 2, cv::Scalar(0, 0, 255), CV_FILLED);
        cv::imwrite("image_" + std::to_string(pair.first) + "_" + std::to_string(matchImgInd) + "_" + std::to_string(pair.first) + ".png", img1);
        cv::Mat img2 = reconstruction.images[matchImgInd].getImage().clone();
        cv::line(img2, cv::Point(startpoint2.x(), startpoint2.y()), cv::Point(otherpoint2.x(), otherpoint2.y()), cv::Scalar(255, 100, 255), 2);
        cv::circle(img2, cv::Point(midpoint2.x(), midpoint2.y()), 2, cv::Scalar(0, 0, 255), CV_FILLED);
        cv::imwrite("image_" + std::to_string(pair.first) + "_" + std::to_string(matchImgInd) + "_" + std::to_string(matchImgInd) + ".png", img2);

    }
}

int main(int argc, char **argv) {
    ReconstructionData reconstruction;

    if (!reconstruction.load_bundler_file("../data/bench/alignment_complete3/complete3.out")) {
        std::cout << "failed to load reconstruction" << std::endl;
        return 1;
    }
    //saveEpipolarLines(reconstruction);
    reconstruction.setImageScale(0.25);
    saveEpipolarLines(reconstruction);
    return 0;
}