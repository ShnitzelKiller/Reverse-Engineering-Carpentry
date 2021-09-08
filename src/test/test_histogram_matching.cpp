//
// Created by James Noeckel on 10/7/20.
//

#include "imgproc/histogram_matching.h"

int main(int argc, char **argv) {
    cv::Mat img1 = cv::imread("../test_data/matching/part_1_view_53_warped.png");
    cv::Mat img2 = cv::imread("../test_data/matching/part_1_view_54_warped.png");
    cv::Mat mask = cv::imread("../test_data/matching/part_1_initMask_eroded.png", cv::IMREAD_GRAYSCALE);
    if (img1.empty() || img2.empty()) {
        std::cout << "images not found" << std::endl;
        return 1;
    }
    histogram_matching(img2, img1, -1, mask, mask);
    cv::imwrite("matched_result.png", img1);
    return 0;
}