//
// Created by James Noeckel on 9/9/20.
//

#include "RunningAverage.h"

RunningAverage::RunningAverage(int dim) : avg(Eigen::VectorXd::Zero(dim)) {}

void RunningAverage::add(double value, double weight) {
    if (w == 0) {
        avg = Eigen::VectorXd::Zero(1);
    }
    double newW = w + weight;
    avg(0) = (avg(0) * w + value * weight) / newW;
    w = newW;
}

void RunningAverage::add(const Eigen::Ref<const Eigen::VectorXd> &value, double weight) {
    if (w == 0) {
        avg = Eigen::VectorXd::Zero(value.size());
    }
    double newW = w + weight;
    avg = (avg * w + value * weight) / newW;
    w = newW;
}

double RunningAverage::getScalar() const {
    return get()(0);
}

Eigen::VectorXd RunningAverage::get() const {
    return avg;
}
