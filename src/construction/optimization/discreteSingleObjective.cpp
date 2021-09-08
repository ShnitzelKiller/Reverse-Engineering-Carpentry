//
// Created by James Noeckel on 9/15/20.
//

#include "discreteProblems.hpp"

using namespace Eigen;

CarpentryOptimizationSOProblem::CarpentryOptimizationSOProblem(const MatrixXd &overlaps, const MatrixXd &allDistances, double maxOverlap, std::vector<int> indexMap, std::vector<int> known)
        : dimension_(indexMap.size()), overlaps_(&overlaps), allDistances_(&allDistances), maxOverlap_(maxOverlap), indexMap_(std::move(indexMap)), known_(std::move(known))
{

}

vector_double::size_type CarpentryOptimizationSOProblem::get_nobj() const {
    return 1;
}

vector_double::size_type CarpentryOptimizationSOProblem::get_nix() const {
    return dimension_;
}

vector_double::size_type CarpentryOptimizationSOProblem::get_nec() const {
    return 0;
}

vector_double::size_type CarpentryOptimizationSOProblem::get_nic() const {
    return 1;
}

vector_double CarpentryOptimizationSOProblem::fitness(const vector_double &dv) const {
    std::vector<bool> w(known_.size());
    for (size_t i=0; i<w.size(); ++i) {
        if (known_[i] == 1) {
            w[i] = true;
        } else if (known_[i] == 2) {
            w[i] = false;
        }
    }
    for (size_t i=0; i<dimension_; ++i) {
        w[indexMap_[i]] = dv[i] > 0.5;
    }
    double maxOverlapRatio = maxOverlap(*overlaps_, w);
//    auto start_t = clock();
    VectorXd minDistances = VectorXd::Constant(allDistances_->rows(), std::numeric_limits<double>::max());
    for (size_t k=0; k<w.size(); ++k) {
        if (w[k]) {
            minDistances = minDistances.cwiseMin(allDistances_->col(k));
        }
    }
    double avgPointDistance = minDistances.mean();
    //auto pd_t = clock();
    //auto total_ct = clock() - pd_t;
    //auto total_pt = pd_t - start_t;
    //float time_sec_ct = static_cast<float>(total_ct) / CLOCKS_PER_SEC;
    //float time_sec_pt = static_cast<float>(total_pt) / CLOCKS_PER_SEC;
    /*std::cout << "call " << numCalls++ << ": ";
    std::cout << "solution: [";
    for (size_t j=0; j < dv.size(); ++j) {
        std::cout << dv[j] << ", ";
    }
    std::cout << "]; Ed=" << avgPointDistance << " Eo=" << maxOverlapRatio << std::endl;*/
    return {avgPointDistance, maxOverlapRatio - maxOverlap_};
}

std::pair<vector_double, vector_double> CarpentryOptimizationSOProblem::get_bounds() const {
    return {std::vector<double>(dimension_, 0.), std::vector<double>(dimension_, 1.)};
}

