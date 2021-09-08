#pragma once
#include "construction/Construction.h"
#include "utils/typedefs.hpp"
#include <pagmo/pagmo.hpp>

using namespace pagmo;

double maxOverlap(const Eigen::MatrixXd &overlaps, const std::vector<bool> &w);

struct PAGMOCarpentryOptimizationMOProblem {
    PAGMOCarpentryOptimizationMOProblem() = default;
    PAGMOCarpentryOptimizationMOProblem(Construction &construction, double a, const Eigen::MatrixXd &overlaps, const Eigen::MatrixXd &allDistances);

    vector_double::size_type get_nobj() const;

    vector_double::size_type get_nix() const;

    // Number of equality constraints.
    vector_double::size_type get_nec() const;
    // Number of inequality constraints.
    vector_double::size_type get_nic() const;

    // Implementation of the objective function.
    vector_double fitness(const vector_double &dv) const;
    // Implementation of the box bounds.
    std::pair<vector_double, vector_double> get_bounds() const;
private:
    vector_double::size_type dimension_;
    Construction *construction_;
    mutable int numCalls = 0;
    const Eigen::MatrixXd *overlaps_;
    const Eigen::MatrixXd *allDistances_;
    double a_;
};

struct CarpentryOptimizationSOProblem {
    CarpentryOptimizationSOProblem() = default;
    CarpentryOptimizationSOProblem(const Eigen::MatrixXd &overlaps, const Eigen::MatrixXd &allDistances, double maxOverlap, std::vector<int> indexMap, std::vector<int> known);

    vector_double::size_type get_nobj() const;

    vector_double::size_type get_nix() const;

    // Number of equality constraints.
    vector_double::size_type get_nec() const;
    // Number of inequality constraints.
    vector_double::size_type get_nic() const;

    // Implementation of the objective function.
    vector_double fitness(const vector_double &dv) const;

    // Implementation of the box bounds.
    std::pair<vector_double, vector_double> get_bounds() const;

private:
    vector_double::size_type dimension_;
    mutable int numCalls = 0;
    const Eigen::MatrixXd *overlaps_;
    const Eigen::MatrixXd *allDistances_;
    double maxOverlap_;
    /** map from reduced indices to full indices */
    std::vector<int> indexMap_;
    /** mask on full indices, 0 = unknown, 1 = true, 2 = false */
    std::vector<int> known_;
};