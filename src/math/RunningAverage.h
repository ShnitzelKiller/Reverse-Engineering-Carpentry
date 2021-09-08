//
// Created by James Noeckel on 9/9/20.
//

#pragma once
#include <Eigen/Dense>

class RunningAverage {
public:
    explicit RunningAverage(int dim=1);
    void add(const Eigen::Ref<const Eigen::VectorXd> &value, double weight=1.0);
    void add(double value, double weight=1.0);
    Eigen::VectorXd get() const;
    double getScalar() const;
private:
    double w = 0.0;
    Eigen::VectorXd avg;
};

