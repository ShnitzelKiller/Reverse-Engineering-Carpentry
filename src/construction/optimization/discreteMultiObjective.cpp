//
// Created by James Noeckel on 9/15/20.
//

#include "discreteProblems.hpp"

using namespace Eigen;

PAGMOCarpentryOptimizationMOProblem::PAGMOCarpentryOptimizationMOProblem(Construction &construction, double a, const MatrixXd &overlaps, const MatrixXd &allDistances)
: construction_(&construction), dimension_(construction.partData.size()), a_(a), overlaps_(&overlaps), allDistances_(&allDistances)
{

}

vector_double::size_type PAGMOCarpentryOptimizationMOProblem::get_nobj() const {
    return 3;
}

vector_double::size_type PAGMOCarpentryOptimizationMOProblem::get_nix() const {
    return dimension_;
}

vector_double::size_type PAGMOCarpentryOptimizationMOProblem::get_nec() const {
    return 0;
}

vector_double::size_type PAGMOCarpentryOptimizationMOProblem::get_nic() const {
    return 0;
}

vector_double PAGMOCarpentryOptimizationMOProblem::fitness(const vector_double &dv) const {
    std::vector<bool> w(dimension_);
    for (size_t i=0; i<w.size(); ++i) {
        w[i] = dv[i] > 0.5;
    }

    double maxOverlapRatio = maxOverlap(*overlaps_, w);
    construction_->setW(w);
    auto start_t = clock();
    VectorXd minDistances = VectorXd::Constant(allDistances_->rows(), std::numeric_limits<double>::max());
    for (size_t k=0; k<w.size(); ++k) {
        if (w[k]) {
            minDistances = minDistances.cwiseMin(allDistances_->col(k));
        }
    }
    double avgPointDistance = minDistances.mean();
    auto pd_t = clock();
    double connectivity = construction_->connectivityEnergy(a_);
    auto total_ct = clock() - pd_t;
    auto total_pt = pd_t - start_t;
    float time_sec_ct = static_cast<float>(total_ct) / CLOCKS_PER_SEC;
    float time_sec_pt = static_cast<float>(total_pt) / CLOCKS_PER_SEC;
    std::cout << "call " << numCalls++ << ": Ed=" << avgPointDistance << " (" << time_sec_pt << "s); " << "Ec=" << connectivity << " (" << time_sec_ct << "s); " << "Eo=" << maxOverlapRatio << std::endl;
    return {avgPointDistance, connectivity,
            maxOverlapRatio};
}

std::pair<vector_double, vector_double> PAGMOCarpentryOptimizationMOProblem::get_bounds() const {
    return {std::vector<double>(dimension_, 0.), std::vector<double>(dimension_, 1.)};
}