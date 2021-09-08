//
// Created by James Noeckel on 4/6/20.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <utility>
#include "geometry/shapes2/VoxelGrid.hpp"
#include "math/fields/PointDensityField.h"
#include <random>

#define cellSize 32

using namespace Eigen;

struct CircleField : public ScalarField<2> {
    CircleField(Vector2d center, double r) : center_(std::move(center)), r_(r) {}
    VectorXd operator()(const Ref<const Matrix<double, -1, 2>> &Q) const override {
        size_t n = Q.rows();
        VectorXd values(n);
        for (unsigned i=0; i<n; ++i) {
            double d1 = r_*r_ - (Q.row(i) - center_.transpose()).squaredNorm();
            double d2 = 0.25 * r_*r_ - (Q.row(i).transpose() - center_ - Vector2d(0., r_)).squaredNorm();
            double d = -log(exp(-d1/10.) + exp(d2/10.))*10.;
            values(i) = d > 0.5 ? 1.0 : -1.0;
        }
        return values;
    }
private:
    Vector2d center_;
    double r_;
};

struct ThresholdField : public ScalarField<2> {
    ThresholdField(const ScalarField<2> &field, double threshold) : field_(field), threshold_(threshold) {}
    VectorXd operator()(const Ref<const Matrix<double, -1, 2>> &Q) const override {
        VectorXd result = field_(Q);
        for (size_t i=0; i<result.size(); ++i) {
            result[i] = result[i] > threshold_ ? 1.0 : -1.0;
        }
        return result;
    }
private:
    const ScalarField<2> &field_;
    double threshold_;
};

void draw_contours(const std::string &name, const std::vector<std::vector<Eigen::Vector2d>> &contours, int width, int height, double offsetX, double offsetY, double spacing, std::shared_ptr<ScalarField<2>> field=std::shared_ptr<ScalarField<2>>(), double threshold=NAN, double avgDensity=1.0) {
    cv::Mat image(height * cellSize, width * cellSize, CV_8UC3);
    image = cv::Scalar(255, 255, 255);
    if (field) {
        for (size_t i=0; i<image.cols; ++i) {
            for (size_t j=0; j<image.rows; ++j) {
                double x = static_cast<double>(i)/cellSize * spacing + offsetX;
                double y = static_cast<double>(j)/cellSize * spacing + offsetY;
                double fieldVal = (*field)(RowVector2d(x, y))(0);
                bool highlight = std::isfinite(threshold) && std::abs(fieldVal - threshold) < threshold * 0.1;
                double col = std::min(255, 50 + static_cast<int>(fieldVal * 255 / avgDensity));
                image.at<cv::Vec3b>(j, i) = cv::Vec3b(col, col, highlight ? 255 : col);
            }
        }
    }
    for (size_t i=0; i<width; ++i) {
        cv::line(image, cv::Point2d((i+0.5) * cellSize, 0), cv::Point2d((i+0.5) * cellSize, image.rows), cv::Scalar(100, 100, 255));
    }
    for (size_t i=0; i<height; ++i) {
        cv::line(image, cv::Point2d(0, (i+0.5) * cellSize), cv::Point2d(image.cols, (i+0.5) * cellSize), cv::Scalar(100, 100, 255));
    }
    for (const auto &contour : contours) {
        for (size_t i=0; i<contour.size(); ++i) {
            const auto &ptA = contour[i];
            const auto &ptB = contour[(i+1)%contour.size()];
            Eigen::Vector2d ptA2 = (ptA - Vector2d(offsetX, offsetY))/spacing * cellSize;
            Eigen::Vector2d ptB2 = (ptB - Vector2d(offsetX, offsetY))/spacing * cellSize;
            cv::line(image, cv::Point2d(ptA2(0), ptA2(1)), cv::Point2d(ptB2(0), ptB2(1)), cv::Scalar(0, 255, 0), 2);
        }
    }
    cv::imshow(name, image);
    cv::waitKey();
    cv::destroyWindow(name);
}

void printContours(const std::vector<std::vector<Eigen::Vector2d>> &contours, const std::vector<std::vector<int>> &hierarchy) {
    std::cout << "found " << contours.size() << " contours" << std::endl;
    for (int i=0; i<contours.size(); i++) {
        std::cout << i << ": " << contours[i].size() << std::endl;
        if (!hierarchy[i].empty()) {
            std::cout << "children " << i << ": ";
            for (int j : hierarchy[i]) {
                std::cout << j << ", ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << "outer contours: ";
    for (int j : hierarchy.back()) {
        std::cout << j << ", ";
    }
    std::cout << std::endl;
}

bool test(std::vector<double> &data, int width, int height, const std::string &name, int expected_outer_contours, int outer_children=-1) {
    std::cout << "testing " << name << std::endl;
    VoxelGrid2D grid(std::move(data), width, height, 0, 0, 1);
    std::vector<std::vector<int>> hierarchy;
    std::vector<std::vector<Eigen::Vector2d>> contours = grid.marching_squares(hierarchy, 0.5f);
    printContours(contours, hierarchy);
    draw_contours(name, contours, width, height, 0, 0, 1);
    if (expected_outer_contours == hierarchy.back().size() && (outer_children < 0 || outer_children == hierarchy[hierarchy.back()[0]].size())) {
        return true;
    } else {
        std::cout << "test " << name << " failed\n";
        return false;
    }
}

int main(int argc, char **argv) {
    {
        //test point density field
        std::mt19937 randomEngine(std::random_device{}());
        int width = 20, height = 20;
        int numPoints = 2000;
        double threshold = 2;
        MatrixX2d points(numPoints, 2);
        size_t numUsed = 0;
        cv::Mat image = cv::Mat::zeros(height * cellSize, width * cellSize, CV_8UC3);
        for (size_t i=0; i<numPoints; ++i) {
            RowVector2d pt(std::uniform_real_distribution<double>(0, width)(randomEngine), std::uniform_real_distribution<double>(0, height)(randomEngine));
            if ((pt - RowVector2d(width/2, height/2)).squaredNorm() < width*width/5 ) {
                points.row(numUsed++) = pt;
                cv::circle(image, cv::Point(pt.x() * cellSize, pt.y() * cellSize), 3, cv::Scalar(0, 0, 255), cv::FILLED);
            }
        }
        cv::imshow("points", image);
        cv::waitKey();
        cv::destroyWindow("points");
        PointCloud2::Handle points2( new PointCloud2);
        points2->P = points.block(0, 0, numUsed, 2);
        {
            VoxelGrid2D grid(points2->P, 1);
            auto contours = grid.marching_squares(0.1, false);
            draw_contours("occupancy grid", contours, width, height, 0, 0, 1);
        }
        std::shared_ptr<ScalarField<2>> field = std::make_shared<PointDensityField>(std::move(points2), 0.5);
        std::shared_ptr<ScalarField<2>> tfield = std::make_shared<ThresholdField>(*field, threshold);
        VoxelGrid2D grid(tfield, 0, 0, width, height, 1);
        std::vector<std::vector<int>> hierarchy;
        auto contours = grid.marching_squares(hierarchy, 0, false);
        printContours(contours, hierarchy);
        draw_contours("scalar field without bisection", contours, width, height, 0, 0, 1, field, threshold, 20);
        hierarchy.clear();
        contours = grid.marching_squares(hierarchy, 0);
        printContours(contours, hierarchy);
        draw_contours("scalar field with bisection", contours, width, height, 0, 0, 1, field, threshold, 20);
    }
    {
        //test a scalar field
        int width = 20, height = 20;
        double threshold = 0.0;
        std::shared_ptr<ScalarField<2>> field = std::make_shared<CircleField>(Vector2d(width/2, height/2), width*0.4);
        VoxelGrid2D grid(field, 0, 0, width, height, 1);
        std::vector<std::vector<int>> hierarchy;
        auto contours = grid.marching_squares(hierarchy, threshold, false);
        printContours(contours, hierarchy);
        draw_contours("scalar field without bisection", contours, width, height, 0, 0, 1, field, threshold);
        hierarchy.clear();
        contours = grid.marching_squares(hierarchy, threshold);
        printContours(contours, hierarchy);
        draw_contours("scalar field with bisection", contours, width, height, 0, 0, 1, field, threshold);
    }
    bool passed = true;
    {
        std::vector<double> data(9, 1.0f);
        data[4] = 0.0f;
        passed = passed && test(data, 3, 3, "donut", 1, 1);
    }
    {
        std::vector<double> data(9, 1.0f);
        data[3] = data[4] = data[5] = 0.0f;
        passed = passed && test(data, 3, 3, "two horizontal", 2, 0);
    }
    {
        std::vector<double> data(9, 1.0f);
        data[1] = data[4] = data[7] = 0.0f;
        passed = passed && test(data, 3, 3, "two vertical", 2, 0);
    }
    {
        std::vector<double> data(25, 1.0f);
        for (int i=1; i<4; i++) {
            for (int j=1; j<4; j++) {
                data[i*5+j] = 0.0f;
            }
        }
        data[12] = 1.0f;
        passed = passed && test(data, 5, 5, "double donut", 1, 1);
    }
    {
        std::vector<double> data(25, 1.0f);
        data[6] = 0.0f;
        data[8] = 0.0f;
        data[16] = 0.0f;
        passed = passed && test(data, 5, 5, "holes", 1, 3);
    }





    return !passed;
}