//
// Created by James Noeckel on 10/13/20.
//

#include "imgproc/dda_foreach.h"
#include "../../../libigl/include/igl/copyleft/marching_cubes.h"
#include "../../../libigl/include/igl/opengl/glfw/Viewer.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

int main(int argc, char ** argv) {
    {
        cv::Mat img = cv::Mat::zeros(64, 64, CV_8UC1);
        auto lambda = [&](int i, int j) {
            img.at<uchar>(i, j) = 255;
        };
        dda_foreach(lambda, 5.0, 12.0, 32.0, 22.0);
        cv::imwrite("test1.png", img);

        img = 0;
        dda_foreach(lambda, 32.0, 22.0, 5.0, 12.0);
        cv::imwrite("test2.png", img);

        img = 0;
        dda_foreach(lambda, 5., 5., 5.001, 5.);
        cv::imwrite("test3.png", img);

        img = 0;
        dda_foreach(lambda, 10., 50., 50., 10.);
        cv::imwrite("test4.png", img);
        img = 0;
        dda_foreach(lambda, 50., 10., 10., 50.);
        cv::imwrite("test5.png", img);


        auto lambda2 = [&](int i, int j) {
            std::cout << "(" << i << ", " << j << ")" << std::endl;
        };
        dda_foreach(lambda2, 497.663, 618.022, 498.337, 613.978);
    }

    using namespace Eigen;
    Array3i resolution(20, 20, 20);
    int N = resolution.prod();
    VectorXd field = VectorXd::Ones(N);
    MatrixX3d gridPoints(resolution.prod(), 3);
    int gridInd = 0;
    for (int i=0; i<resolution.x(); ++i) {
        for (int j=0; j<resolution.y(); ++j) {
            for (int k=0; k<resolution.z(); ++k, ++gridInd) {
                Vector3d point(i, j, k);
                gridPoints.row(gridInd) = point.transpose();
            }
        }
    }
    auto lambda = [&](int i, int j, int k) {
        field[i * resolution.y()*resolution.z() + j * resolution.z() + k] = -1;
    };
    //axes
    dda_foreach(lambda, 0., 10., 10., 19., 10., 10.);
    dda_foreach(lambda, 10., 0., 10., 10., 19., 10.);
    dda_foreach(lambda, 10., 10., 0., 10., 10., 19.);

    dda_foreach(lambda, 0., 0., 0., 19., 19., 19.);
//    dda_foreach(lambda, 10., 10., 0., 19., 19., 19.);
    dda_foreach(lambda, 19., 19., 19., 10., 10., 0.);
    MatrixX3d V(0, 3);
    MatrixX3i F(0, 3);
    igl::copyleft::marching_cubes(field, gridPoints, resolution.z(), resolution.y(), resolution.x(),
                                  0.0, V, F);
    for (int i=0; i<F.rows(); ++i) {
        std::swap(F(i, 0), F(i, 1));
    }
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V, F);
    viewer.data().point_size=5;
    for (int i=0; i<2; ++i) {
        for (int j=0; j<2; ++j) {
            for (int k=0; k<2; ++k) {
                viewer.data().add_points(RowVector3d(i, j, k) * 20, RowVector3d(i, j, k));
            }
        }
    }
    viewer.launch();
    return 0;
}