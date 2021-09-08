//
// Adapted from user davidhigh's StackOverflow post at https://stackoverflow.com/questions/6142576/sample-from-multivariate-normal-gaussian-distribution-in-c
//

#include "NormalRandomVariable.h"
#include <random>

NormalRandomVariable::NormalRandomVariable(Eigen::Ref<const Eigen::MatrixXd> const &covar)
        : NormalRandomVariable(Eigen::VectorXd::Zero(covar.rows()), covar)
{}


NormalRandomVariable::NormalRandomVariable(Eigen::Ref<const Eigen::VectorXd> const &mean,
                                           Eigen::Ref<const Eigen::MatrixXd> const &covar) : mean(mean)
{
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
    transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
}


Eigen::VectorXd NormalRandomVariable::operator()() const
{
    static std::mt19937 gen{ std::random_device{}() };
    static std::normal_distribution<> dist;
    return mean + transform * Eigen::VectorXd{ mean.size() }.unaryExpr([&](auto x) { return dist(gen); });
}