//
// Created by James Noeckel on 9/15/20.
//

#include "discreteProblems.hpp"

double maxOverlap(const Eigen::MatrixXd &overlaps, const std::vector<bool> &w) {
    double maxOverlapRatio = 0.0;
    for (size_t i=0; i<w.size(); ++i) {
        if (w[i]) {
            double totalOverlap = 0.0;
            for (size_t j = 0; j < w.size(); ++j) {
                if (i == j) continue;
                if (w[j]) {
                    totalOverlap += overlaps(i, j);
                }
            }
            maxOverlapRatio = std::max(maxOverlapRatio, totalOverlap);
        }
    }
    return maxOverlapRatio;
}