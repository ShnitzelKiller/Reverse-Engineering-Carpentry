//
// Created by James Noeckel on 4/8/20.
//

#include <iostream>
#include "math/robust_derivative.hpp"
#include "math/nonmax_suppression.h"
#include "geometry/find_affine.h"
#include "geometry/csg1d.h"

int main(int argc, char **argv) {
    {
        std::vector<std::pair<double, double>> i1 = {{0, 1}, {2, 2.3}};
        std::vector<std::pair<double, double>> i2 = {{0.5, 1.5}, {1.7, 10}};
        auto i3 = csg1d(i1, i2);
        for (const auto &interval : i3) {
            std::cout << "[" << interval.first << "-" << interval.second << "], ";
        }
        std::cout << std::endl;
    }

    {
        Eigen::MatrixX3d P(10, 3);
        P.setRandom();
        std::cout << "P: \n" << P << std::endl;
        Eigen::Vector3d trans;
        trans.setRandom();
        Eigen::Vector3d axis;
        axis.setRandom();
        axis.normalize();
        Eigen::Quaterniond rot(Eigen::AngleAxis(1.0, axis));
        double scale = 5;
        Eigen::MatrixX3d Q(P.rows(), 3);
        for (int i=0; i<P.rows(); ++i) {
            Q.row(i) = (scale * (rot * P.row(i).transpose()) + trans).transpose();
        }
        std::cout << "Q: \n" << Q << std::endl;

        double scale2;
        Eigen::Vector3d trans2;
        Eigen::Quaterniond rot2;
        find_affine(P, Q, scale2, trans2, rot2);
        std::cout << "scale: " << scale << "; scale2: " << scale2 << std::endl;
        std::cout << "rot: " << rot.w() << ", " << rot.x() << ", " << rot.y() << ", " << rot.z() << "; rot2: " << rot2.w() << ", " << rot2.x() << ", " << rot2.y() << ", " << rot2.z() << std::endl;
        std::cout << "trans: " << trans.transpose() << "; trans2: " << trans2.transpose() << std::endl;
        Eigen::MatrixX3d Ptrans(P.rows(), 3);
        for (int i=0; i<P.rows(); ++i) {
            Ptrans.row(i) = (scale2 * (rot2 * P.row(i).transpose()) + trans2).transpose();
        }
        std::cout << "error: \n" << (Ptrans - Q).rowwise().squaredNorm() << std::endl;
    }
    {
        std::vector<int> values = {0, 1, 2, 1, 0};
        std::vector<int> suppressed(5);
        nonmax_suppression<int>(values.begin(), values.end(), suppressed.begin());
        if (suppressed[2] != 2 || suppressed[0] != 0 || suppressed[1] != 0 || suppressed[3] != 0 || suppressed[4] != 0) {
            std::cout << "values: ";
            for (auto val : suppressed) {
                std::cout << val << ", ";
            }
            std::cout << std::endl;
            return 1;
        }
    }
    {
        std::vector<int> values = {0, 1, 2, 1, 0};
        nonmax_suppression<int>(values.begin(), values.end(), values.begin());
        if (values[2] != 2 || values[0] != 0 || values[1] != 0 || values[3] != 0 || values[4] != 0) {
            std::cout << "values: ";
            for (auto val : values) {
                std::cout << val << ", ";
            }
            std::cout << std::endl;
            return 1;
        }
    }
    {
        std::vector<int> values = {0, 1, 2, 2, 1};
        nonmax_suppression<int>(values.begin(), values.end(), values.begin());
        if (values[2] != 0 || values[0] != 0 || values[1] != 0 || values[3] != 2 || values[4] != 0) {
            std::cout << "values: ";
            for (auto val : values) {
                std::cout << val << ", ";
            }
            std::cout << std::endl;
            return 1;
        }
    }
    {
        std::vector<int> values = {0, 1, 2, 1, 2};
        nonmax_suppression<int>(values.begin(), values.end(), values.begin());
        if (values[0] != 0 || values[1] != 0 || values[2] != 2 || values[3] != 0 || values[4] != 0) {
            std::cout << "values: ";
            for (auto val : values) {
                std::cout << val << ", ";
            }
            std::cout << std::endl;
            return 1;
        }
    }
    {
        std::vector<int> values = {3, 1, 2, 1, 2};
        nonmax_suppression<int>(values.begin(), values.end(), values.begin());
        if (values[0] != 0 || values[1] != 0 || values[2] != 2 || values[3] != 0 || values[4] != 0) {
            std::cout << "values: ";
            for (auto val : values) {
                std::cout << val << ", ";
            }
            std::cout << std::endl;
            return 1;
        }
    }
    {
        std::vector<int> values = {1, 1, 2, 1, 1};
        nonmax_suppression<int>(values.begin(), values.end(), values.begin());
        if (values[0] != 0 || values[1] != 0 || values[2] != 2 || values[3] != 0 || values[4] != 0) {
            std::cout << "values: ";
            for (auto val : values) {
                std::cout << val << ", ";
            }
            std::cout << std::endl;
            return 1;
        }
    }
    {
        std::vector<int> values = {0, 1, 2, 3, 4};
        nonmax_suppression<int>(values.begin(), values.end(), values.begin());
        if (values[0] != 0 || values[1] != 0 || values[2] != 0 || values[3] != 0 || values[4] != 0) {
            std::cout << "values: ";
            for (auto val : values) {
                std::cout << val << ", ";
            }
            std::cout << std::endl;
            return 1;
        }
    }
    {
        std::vector<float> values = {0.11f, 0.1f, 0.15f, 0.155f, 0.154f, -10.0f};
        nonmax_suppression<float>(values.begin(), values.end(), values.begin());
        if (values[0] != 0 || values[1] != 0 || values[2] != 0 || values[3] != 0.155f || values[4] != 0 || values[5] != 0) {
            std::cout << "values: ";
            for (auto val : values) {
                std::cout << val << ", ";
            }
            std::cout << std::endl;
            return 1;
        }
    }
    {
        std::vector<double> values = {0.0, 0.0, 0.0, 0.0, 10., 0.0, 0.0, 0.0, 0.0};
        std::vector<double> derivs = robust_derivative(values, 1.0);
        std::cout << "derivatives: ";
        for (double d : derivs) {
            std::cout << d << ", ";
        }
        std::cout << std::endl;
    }
    return 0;
}