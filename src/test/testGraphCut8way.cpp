
//
// Created by James Noeckel on 1/23/21.
//

#include "imgproc/graph_cut.h"

void testEnergy(const cv::Mat &energy, const std::string &name, float smoothness) {
    std::cout << "testing " << name << std::endl;
    cv::Mat labels(energy.rows, energy.cols, CV_8UC1);
    double flow = graph_cut(energy, labels, smoothness);
    std::cout << "flow 4way: " << flow << std::endl;
    cv::imwrite(name + "_4waymask.png", labels * 255);
    labels = cv::Mat(energy.rows, energy.cols, CV_8UC1);
    flow = graph_cut(energy, labels, smoothness, {}, 0, {}, {}, 0, true);
    std::cout << "flow 8way: " << flow << std::endl;
    cv::imwrite(name + "_8waymask.png", labels * 255);
}

int main(int argc, char **argv) {
    float smoothness = 1;
    int width = 50;
    float maxEnergy = 10000000.0f;
    int centerline = width/2;
    {
        cv::Mat energy(3, 3, CV_32FC2);
        for (int i=0; i<3; ++i) {
            for (int j=0; j<3; ++j) {
                float energy0 = 0, energy1 = maxEnergy;
                if (i == 1 && j == 1) {
                    std::cout << "setting center pixel" << std::endl;
                    energy0 = maxEnergy;
                    energy1 = 0;
                }
                energy.at<cv::Vec2f>(i, j) = cv::Vec2f(energy0, energy1);
            }
        }
        testEnergy(energy, "onepixel", 1);
    }

    cv::Mat energy(width, width, CV_32FC2);

    for (int i=0; i<width; ++i) {
        for (int j=0; j<width; ++j) {
            float energy0 = 0, energy1 = 0;
            if (i == centerline || j == centerline){
                energy0 = maxEnergy;
                energy1 = 0;
            }
            if (i == 0 || j == 0 || i == width-1 || j == width-1) {
                energy0 = 0;
                energy1 = maxEnergy;
            }
            energy.at<cv::Vec2f>(i, j) = cv::Vec2f(energy0, energy1);
        }
    }
    testEnergy(energy, "axial", smoothness);

    for (int i=0; i<width; ++i) {
        for (int j=0; j<width; ++j) {
            float energy0 = 0, energy1 = 0;
            if (std::abs(i-centerline) == std::abs(j-centerline)){
                energy0 = maxEnergy;
                energy1 = 0;
            }
            if (i == 0 || j == 0 || i == width-1 || j == width-1) {
                energy0 = 0;
                energy1 = maxEnergy;
            }
            energy.at<cv::Vec2f>(i, j) = cv::Vec2f(energy0, energy1);
        }
    }

    testEnergy(energy, "diag", smoothness);
    return 0;
}