//because pagmo2 simulated_annealing implementation only works with continuous variables, we need a simler one
#pragma once

#include <pagmo/pagmo.hpp>
#include <pagmo/population.hpp>
#include <pagmo/algorithms/not_population_based.hpp>
#include <Eigen/Dense>
using namespace pagmo;
class PAGMO_DLL_PUBLIC HypercubeSimulatedAnnealing : public not_population_based {
public:
    HypercubeSimulatedAnnealing(unsigned iters=1000, const Eigen::MatrixXd *overlaps=nullptr, double minOverlap=0.5, double Ts = 10., double Tf = .1, unsigned n_T_adj = 10u, unsigned seed = pagmo::random_device::next());

    population evolve(population pop) const;

    /*void set_seed(unsigned seed) {
        m_seed = seed;
        m_e.
    }*/

    std::string get_name() const;

private:
    unsigned m_iters;
    // Starting temperature
    double m_Ts;
    // Final temperature
    double m_Tf;
    // Number of temperature adjustments during the annealing procedure
    unsigned m_n_T_adj;

    mutable pagmo::detail::random_engine_type m_e;
    unsigned m_seed;
    const Eigen::MatrixXd *m_overlaps;
    double m_minOverlap;
};
