//
// Created by James Noeckel on 1/8/20.
//
#include "math/integration.h"
#include "utils/test_utils.h"

#define Ass(cond) if (!cond) { return 1; }

bool test_integrate(int rows, int cols, const cv::Point2d &a, const cv::Point2d &b, const std::string &name, bool grayscale=false) {
    std::cout << name << std::endl;
    cv::Mat img;
    if (grayscale) {
        img = cv::Mat::ones(rows, cols, CV_8UC1);
    }
    else {
        img = cv::Mat::ones(rows, cols, CV_8UC3);
    }
    Eigen::VectorXd color = integrate_image(img, Eigen::Vector2d(a.x, a.y), Eigen::Vector2d(b.x, b.y));
    std::cout << "result: " << color.transpose() << std::endl;
    bool passed = std::abs(color[0] - 1) < 1e-6;
    if (!grayscale) {
        img = cv::Mat::zeros(rows, cols, CV_8UC3);
        integrate_image(img, Eigen::Vector2d(a.x, a.y), Eigen::Vector2d(b.x, b.y), false, Eigen::Vector2d(),
                        Eigen::Vector2d(), cv::Mat(), true);
        cv::imwrite(name + ".png", img);
        std::cout << name << (passed ? " passed" : " failed") << std::endl;
    }
    return passed;
}

int main(int argc, char **argv) {
    {
        cv::Mat img = cv::Mat::zeros(5, 5, CV_8UC1);
        img.at<uchar>(1, 1) = 255;
        img.at<uchar>(1, 2) = 255;
        img.at<uchar>(1, 3) = 255;
        img.at<uchar>(2, 1) = 255;
        img.at<uchar>(2, 2) = 255;
        img.at<uchar>(2, 3) = 255;
        img.at<uchar>(3, 1) = 255;
        img.at<uchar>(3, 2) = 255;
        img.at<uchar>(3, 3) = 255;
        img.at<uchar>(4, 1) = 255;
        img.at<uchar>(4, 2) = 255;
        img.at<uchar>(4, 3) = 255;
        cv::Mat derivativeX;
        cv::Mat derivativeY;
        cv::Scharr(img, derivativeX, CV_16S, 1, 0);
        cv::Scharr(img, derivativeY, CV_16S, 0, 1);
        std::cout << img << std::endl;
        std::cout << "x: " << std::endl << derivativeX << std::endl;
        std::cout << "y: " << std::endl << derivativeY << std::endl;

        Eigen::VectorXd color_1 = integrate_image(img, Eigen::Vector2d(1, 0), Eigen::Vector2d(1, 4));
        Eigen::VectorXd color_tenth = integrate_image(img, Eigen::Vector2d(.1, 0), Eigen::Vector2d(.1, 4));
        Ass(assertEquals(color_1.size(), 1, "color 1 size"))
        Ass(assertApproxEquals(color_tenth(0)*10, color_1(0), "vertical col_tenth times 10"))
        color_1 = integrate_image(img, Eigen::Vector2d(0, 1), Eigen::Vector2d(4, 1));
        color_tenth = integrate_image(img, Eigen::Vector2d(0, .1), Eigen::Vector2d(4, .1));
        Ass(assertApproxEquals(color_tenth(0)*10, color_1(0), "horizontal col_tenth times 10"))
        Eigen::VectorXd color = integrate_image(img, Eigen::Vector2d(0.5, 1), Eigen::Vector2d(0.5, 2));
        Ass(assertApproxEquals(color(0), 127.5, "vertical pixel midpoint color"))
        color = integrate_image(img, Eigen::Vector2d(1, 1), Eigen::Vector2d(1, 2));
        Ass(assertApproxEquals(color(0), 255, "vertical full pixel edge color"))
        color = integrate_image(img, Eigen::Vector2d(1, 1), Eigen::Vector2d(1, 3));
        Ass(assertApproxEquals(color(0), 255, "vertical full two pixel edge color"))
        color = integrate_image(img, Eigen::Vector2d(0, 0), Eigen::Vector2d(0, 1));
        Ass(assertApproxEquals(color(0), 0, "vertical empty pixel edge color"))
        color = integrate_image(img, Eigen::Vector2d(2, 0), Eigen::Vector2d(2, 1));
        Ass(assertApproxEquals(color(0), 127.5, "vertical pixel linear interp color"))
        {
            double derivA = integrate_image(derivativeX, Eigen::Vector2d(1, 0), Eigen::Vector2d(1, 1), true, Eigen::Vector2d(1, 0), Eigen::Vector2d(0, 0), derivativeY)(0) / 16;
            double derivB = integrate_image(derivativeX, Eigen::Vector2d(1, 0), Eigen::Vector2d(1, 1), true, Eigen::Vector2d(0, 0), Eigen::Vector2d(1, 0), derivativeY)(0) / 16;
            Ass(assertPrint(derivA < derivB, "derivative moving the top point should be smaller than moving the bottom point"))
        }
        {
            double derivA = integrate_image(derivativeX, Eigen::Vector2d(2, 2), Eigen::Vector2d(2, 3), true, Eigen::Vector2d(0, 1), Eigen::Vector2d(0, 0), derivativeY)(0) / 16;
            Ass(assertApproxEquals(derivA, 0, "derivative of contracting endpoints is not 0"))
        }
        {
            double derivA = integrate_image(derivativeX, Eigen::Vector2d(2, 1), Eigen::Vector2d(2, 3), true, Eigen::Vector2d(0, 1), Eigen::Vector2d(0, 0), derivativeY)(0) / 16;
            double derivB = integrate_image(derivativeX, Eigen::Vector2d(2, 1), Eigen::Vector2d(2, 3), true, Eigen::Vector2d(0, 0), Eigen::Vector2d(0, 1), derivativeY)(0) / 16;
            //std::cout << "derivA: " << derivA << ", " << "derivB: " << derivB << std::endl;
            Ass(assertPrint(derivA > 0 && derivB > 0, "shifting both endpoints should down should lead to positive change"))
        }
        {
            double derivA = integrate_image(derivativeX, Eigen::Vector2d(2, 2), Eigen::Vector2d(3, 2), true, Eigen::Vector2d(-1, 0), Eigen::Vector2d(0, 0), derivativeY)(0) / 16;
            double derivB = integrate_image(derivativeX, Eigen::Vector2d(2, 2), Eigen::Vector2d(3, 2), true, Eigen::Vector2d(0, 0), Eigen::Vector2d(-1, 0), derivativeY)(0) / 16;
            Ass(assertPrint(derivA > 0 && derivB > 0, "shifting both endpoints should left should lead to positive change"))
        }
    }
    bool passed =
    test_integrate(100, 200, cv::Point(25, 25), cv::Point(160, 90), "test1") &&
    test_integrate(100, 200, cv::Point(25, 25), cv::Point(160, 25), "test2") &&
    test_integrate(100, 200, cv::Point(25, 25), cv::Point(25, 90), "test3") &&
    test_integrate(100, 200, cv::Point(25, 25), cv::Point(30, 250), "test4") &&
    test_integrate(100, 200, cv::Point(30, 90), cv::Point(160, 25), "test5") &&
    test_integrate(100, 200, cv::Point(160, 90), cv::Point(25, 25), "test6") &&
    test_integrate(100, 200, cv::Point(160, 25), cv::Point(25, 90), "test7") &&
    test_integrate(100, 200, cv::Point(160, 25), cv::Point(25, 25), "test8") &&
    test_integrate(100, 200, cv::Point(25, 90), cv::Point(25, 25), "test9") &&
    test_integrate(1024, 1024, cv::Point(960, 0), cv::Point(1, 1023), "test10") &&
    test_integrate(256, 256, cv::Point(-10, -50), cv::Point(128, 128), "test11") &&
    test_integrate(4, 4, cv::Point(0, 0), cv::Point(2, 2), "test12") &&
    test_integrate(3, 3, cv::Point(0, 0), cv::Point(2, 2), "test13") &&
            test_integrate(100, 200, cv::Point(25, 25), cv::Point(160, 90), "test14_grayscale", true);
    return !passed;
}