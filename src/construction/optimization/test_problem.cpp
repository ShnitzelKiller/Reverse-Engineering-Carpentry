//
// Created by James Noeckel on 9/10/20.
//

#include "test_problem.h"
#include "utils/printvec.h"
#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;
struct Callable {
    explicit Callable(const problem &p, bool gradient=false) : p_(p), gradient_(gradient) {}
    vector_double operator()(const vector_double &x) {
        if (gradient_) return p_.gradient(x);
        else return p_.fitness(x);
    }
private:
    const problem &p_;
    bool gradient_;
};

double testGradient(const problem &p, const vector_double &x, double dx) {
    vector_double estimate = estimate_gradient(Callable(p), x, dx);
    std::cout << "estimate: " << std::endl << estimate << std::endl;
    vector_double actual = p.gradient(x);
    std::cout << "actual: " << std::endl << actual << std::endl;
    vector_double diff(estimate.size());
    double maxError = 0.0;
    for (size_t i=0; i<diff.size(); ++i) {
        diff[i] = estimate[i]-actual[i];
        maxError = std::max(maxError, std::abs(diff[i]));
    }
    std::cout << "diff: " << diff << std::endl;
    return maxError;
}

double testHessian(const problem &p, const vector_double &x, double dx) {
    auto sparsity = p.hessians_sparsity();
    auto hessians = p.hessians(x);
    std::vector<MatrixXd> denseHessians(sparsity.size(), MatrixXd::Zero(x.size(), x.size()));
    for (size_t i=0; i<sparsity.size(); ++i) {
        for (size_t l=0; l<sparsity[i].size(); ++l) {
            size_t row = sparsity[i][l].first;
            size_t col = sparsity[i][l].second;
            denseHessians[i](row, col) = hessians[i][l];
            denseHessians[i](col, row) = hessians[i][l];
        }
    }
    auto gradOfGrad = estimate_gradient(Callable(p, true), x, dx);
    std::vector<MatrixXd> denseApproxHessians(sparsity.size(), MatrixXd(x.size(), x.size()));
    for (size_t f=0; f<sparsity.size(); ++f) {
        for (size_t i=0; i < x.size(); ++i) {
            for (size_t j=0; j < x.size(); ++j) {
                denseApproxHessians[f](i, j) = gradOfGrad[f * x.size() * x.size() + i*x.size() + j];
            }
        }
    }
    std::vector<MatrixXd> diffs(sparsity.size());
    double maxError = 0.0;
    for (size_t f=0; f<sparsity.size(); ++f) {
        std::cout << "hessian " << f << ": " << std::endl;
        std::cout << denseHessians[f] << std::endl;
        std::cout << "approx hessian " << f << ": " << std::endl;
        std::cout << denseApproxHessians[f] << std::endl;
        diffs[f] = denseApproxHessians[f] - denseHessians[f];
        std::cout << "diff " << f << ": " << std::endl;
        std::cout << diffs[f] << std::endl;
        maxError = std::max(maxError, diffs[f].array().abs().maxCoeff());
    }
    return maxError;
}
