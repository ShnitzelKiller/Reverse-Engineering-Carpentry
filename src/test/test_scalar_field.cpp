//
// Created by James Noeckel on 9/2/20.
//

#include "math/fields/FieldSlice.h"
#include <iostream>

using namespace Eigen;

class CircleField : public ScalarField<3> {
public:
    explicit CircleField(double radius=8.0) : radius_(radius) {}
    VectorXd operator()(const Ref<const Matrix<double, -1, 3>> &Q) const override {
        size_t n = Q.rows();
        VectorXd w(n);
        for (size_t i=0; i<n; ++i) {
            w(i) = Q.row(i).squaredNorm() - radius_*radius_;
            w(i) = w(i) > 0 ? 1.0 : 0.0;
        }
        return w;
    }
private:
    double radius_;
};

int main(int argc, char** argv) {
    Quaterniond rot = Quaterniond::Identity();
    ScalarField<3>::Handle field = std::make_shared<CircleField>();
    {
        FieldSlice slice(field, rot, 0, 16, 100);
        std::cout << "values: " << std::endl;
        for (size_t i = 0; i < 9; ++i) {
            for (size_t j = 0; j < 9; ++j) {
                double val = slice((RowVector2d(i, j).array() - 4) * 2)(0);
                std::cout << val << ", ";
            }
            std::cout << std::endl;
        }
    }
    {
        FieldSlice slice(field, rot, 6);
        std::cout << "values: " << std::endl;
        for (size_t i = 0; i < 9; ++i) {
            for (size_t j = 0; j < 9; ++j) {
                double val = slice((RowVector2d(i, j).array() - 4) * 2)(0);
                std::cout << val << ", ";
            }
            std::cout << std::endl;
        }
    }
    return 0;
}