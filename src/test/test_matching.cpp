//
// Created by James Noeckel on 10/2/20.
//

#include "imgproc/matching_scores.h"

void saveFloatImage(const std::string &name, const cv::Mat &img, float maxval=1) {
    cv::Mat converted;
    img.convertTo(converted, CV_8UC1, 255/maxval);
    cv::imwrite(name, converted);
}

cv::Mat corrImage(const cv::Mat &imgBase, const cv::Mat &imgConv, int iconv, int jconv, int radius, bool separate=true, const cv::Mat &mask1=cv::Mat(), const cv::Mat &mask2=cv::Mat()) {
    cv::Mat corrs(imgBase.rows, imgBase.cols, CV_32FC1);
    for (int i=0; i<imgBase.rows; ++i) {
        for (int j=0; j<imgBase.cols; ++j) {
            corrs.at<float>(i, j) = 0.5f * (1 + norm_cross_correlation(imgBase, imgConv, i, j, iconv, jconv, radius, separate, mask1, mask2));
        }
    }
    return corrs;
}

int main(int argc, char** argv) {
    cv::Mat img = cv::imread("../test_data/limmy.png");
    cv::resize(img, img, cv::Size(), 0.25, 0.25);
    //cv::Mat img = cv::Mat::ones(200, 200, CV_8UC3) * 101;
    if (img.empty()) {
        std::cout << "limmy not found" << std::endl;
        return 1;
    }
    double ncc1 = norm_cross_correlation(img, img, 50, 50, 50, 50, 5);
    double ssd1 = ssd(img, img, 50, 2, 50, 2, 5);
    if (std::abs(ncc1 - 1.0) > 1e-7) {
        std::cout << "incorrect ncc; was " << ncc1 << ", should be 1" << std::endl;
        return 1;
    } else {
        std::cout << "error ncc: " << ncc1 - 1 << std::endl;
    }
    if (ssd1 != 0) {
        std::cout << "incorrect ssd; was " << ssd1 << ", should be 0" << std::endl;
    }
    double ncc2 = norm_cross_correlation(img, img, 50, 50, 100, 150, 5);
    double ssd2 = ssd(img, img, 50, 50, 100, 150, 5);
    std::cout << "ncc2: " << ncc2 << std::endl;
    std::cout << "ssd2: " << ssd2 << std::endl;

    {
        cv::Mat corrs = corrImage(img, img, 23, 66, 5, true);
        saveFloatImage("limmy_correlation_separate.png", corrs);
    }
    {
        cv::Mat corrs = corrImage(img, img, 23, 66, 5, false);
        saveFloatImage("limmy_correlation_joint.png", corrs);
    }
    {
        cv::Mat imgf;
        img.convertTo(imgf, CV_32FC3, 1.f/255);
        cv::Mat corrs = corrImage(imgf, imgf, 23, 66, 5);
        for (size_t i=0; i<corrs.total(); ++i) {
            auto &pix = corrs.at<float>(i);
            pix = std::pow(pix, 2.2);
        }
        saveFloatImage("limmy_correlation_separate_float.png", corrs);
    }
    {
        cv::Mat mask = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
        cv::rectangle(mask, cv::Rect(0, 0, img.cols / 2, img.rows), 1, cv::FILLED);
        cv::Mat corrs = corrImage(img, img, 23, 66, 5, true, mask, mask);
        cv::imwrite("limmy_mask.png", mask * 255);
        saveFloatImage("limmy_correlation_masked.png", corrs);
    }
    return 0;
}