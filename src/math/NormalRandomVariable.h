//
// Adapted from user davidhigh's StackOverflow post at https://stackoverflow.com/questions/6142576/sample-from-multivariate-normal-gaussian-distribution-in-c
//

#pragma once
#include <Eigen/Dense>

struct NormalRandomVariable
{
    explicit NormalRandomVariable(Eigen::Ref<const Eigen::MatrixXd> const& covar);

    NormalRandomVariable(Eigen::Ref<const Eigen::VectorXd> const& mean, Eigen::Ref<const Eigen::MatrixXd> const& covar);

    Eigen::VectorXd mean;
    Eigen::MatrixXd transform;

    Eigen::VectorXd operator()() const;
};