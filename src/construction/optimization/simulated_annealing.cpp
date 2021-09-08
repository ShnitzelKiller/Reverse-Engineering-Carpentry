//
// Created by James Noeckel on 9/15/20.
//

#include "simulated_annealing.hpp"

HypercubeSimulatedAnnealing::HypercubeSimulatedAnnealing(unsigned int iters, const Eigen::MatrixXd *overlaps,
                                                         double minOverlap, double Ts, double Tf, unsigned int n_T_adj,
                                                         unsigned int seed)
                                                         : m_Ts(Ts), m_Tf(Tf), m_n_T_adj(n_T_adj), m_iters(iters), m_e(seed), m_seed(seed), m_overlaps(overlaps), m_minOverlap(minOverlap) {
    if (Ts <= 0. || !std::isfinite(Ts)) {
        pagmo_throw(std::invalid_argument, "The starting temperature must be finite and positive, while a value of "
                                           + std::to_string(Ts) + " was detected.");
    }
    if (Tf <= 0. || !std::isfinite(Tf)) {
        pagmo_throw(std::invalid_argument, "The final temperature must be finite and positive, while a value of "
                                           + std::to_string(Tf) + " was detected.");
    }
    if (Tf > Ts) {
        pagmo_throw(std::invalid_argument,
                    "The final temperature must be smaller than the initial temperature, while a value of "
                    + std::to_string(Tf) + " >= " + std::to_string(Ts) + " was detected.");
    }
    if (n_T_adj == 0u) {
        pagmo_throw(std::invalid_argument,
                    "The number of temperature adjustments must be strictly positive, while a value of "
                    + std::to_string(n_T_adj) + " was detected.");
    }
}

population HypercubeSimulatedAnnealing::evolve(population pop) const {
    const auto &prob = pop.get_problem();
    const auto &lb = prob.get_lb();
    const auto &ub = prob.get_ub();
    const auto dim = prob.get_nx();             // not const as used type for counters
    const auto nic = prob.get_nic();
    const auto nec = prob.get_nec();
    const auto nobj = prob.get_nobj();
    if (prob.get_nix() != prob.get_nx()) {
        pagmo_throw(std::invalid_argument, "non-integer parameters detected in " + prob.get_name() + " instance. "
                                           + get_name() + " cannot deal with them");
    }
    /*if (prob.get_nc() != 0u) {
        pagmo_throw(std::invalid_argument, "Non linear constraints detected in " + prob.get_name() + " instance. "
                                           + get_name() + " cannot deal with them");
    }*/
    if (nobj != 1u) {
        pagmo_throw(std::invalid_argument, "Multiple objectives detected in " + prob.get_name() + " instance. "
                                           + get_name() + " cannot deal with them");
    }
    if (prob.is_stochastic()) {
        pagmo_throw(std::invalid_argument,
                    "The problem appears to be stochastic " + get_name() + " cannot deal with it");
    }
    if (!pop.size()) {
        pagmo_throw(std::invalid_argument, get_name() + " does not work on an empty population");
    }
    for (size_t i=0; i<dim; ++i) {
        if (lb[i] != 0.0 || ub[i] != 1.0) {
            pagmo_throw(std::invalid_argument, prob.get_name() + " must have bounds from 0 to 1 in all dimension");
        }
    }
    std::uniform_real_distribution<double> drng(0., 1.); // to generate a number in [0, 1)
    auto sel_xf = select_individual(pop);
    vector_double x0(std::move(sel_xf.first)), fit0(std::move(sel_xf.second));
    // Determines the coefficient to decrease the temperature
    const double Tcoeff = std::pow(m_Tf / m_Ts, 1.0 / static_cast<double>(m_n_T_adj));
    // Stores the current and new points
    std::vector<unsigned> choices(dim);
    auto xNEW = x0;
    auto xOLD = xNEW;
    auto best_x = xNEW;
    auto fNEW = fit0;
    auto fOLD = fit0;
    auto best_f = fit0;
    auto iters = m_iters / (m_n_T_adj * dim);
    for (decltype(m_n_T_adj) jter = 0u; jter < m_n_T_adj; ++jter) {
        // Stores the number of accepted points for each component
        std::vector<int> acp(dim, 0u);
        double currentT = m_Ts;

        for (decltype(iters) iter = 0u; iter < iters; ++iter) {
            auto nter = std::uniform_int_distribution<vector_double::size_type>(0u, dim - 1u)(m_e);
            for (size_t numb = 0u; numb < dim; ++numb) {
                nter = (nter + 1u) % dim;
                xNEW[nter] = 1.0 - xOLD[nter];
                double a2 = 1.0; //metropolis-hastings correction term
                //if x component is 1, and overlap table is present, look for lateral moves
                if (m_overlaps != nullptr) {
                    /** number of extra choices beyond simply flipping this component */
                    size_t numChoices = 0;
                    for (size_t j = 0; j < dim; ++j) {
                        if (j != nter && xOLD[j] < 0.5) {
                            double ratio1 = (*m_overlaps)(nter, j);
                            double ratio2 = (*m_overlaps)(j, nter);
                            if (std::max(ratio1, ratio2) > m_minOverlap) {
                                choices[numChoices++] = j;
                            }
                        }
                    }
                    // if this is the newly disabled component, choose an overlapping component to enable
                    if (xNEW[nter] < 0.5 && numChoices > 0) {
                        /** [0,numChoices-1]: perform swap move
                         *  numChoices: only turn off component */
                        unsigned choice = std::uniform_int_distribution<unsigned>(0, numChoices)(m_e);
                        if (choice < numChoices) {
                            xNEW[choices[choice]] = 1.0;
                            //compute the number of outgoing choices from target component
                            size_t numReverseChoices = 0;
                            for (size_t j = 0; j < dim; ++j) {
                                if (j != choices[choice] && xNEW[j] < 0.5) {
                                    double ratio1 = (*m_overlaps)(choices[choice], j);
                                    double ratio2 = (*m_overlaps)(j, choices[choice]);
                                    if (std::max(ratio1, ratio2) > m_minOverlap) {
                                        ++numReverseChoices;
                                    }
                                }
                            }
                            a2 = static_cast<double>(numChoices+1)/static_cast<double>(numReverseChoices+1);
                        } else {
                            // if component is only disabled, enabling is that component's sole reverse operation
                            a2 = static_cast<double>(numChoices+1);
                        }
                    } else {
                        // if enabling a component, numChoices is the number of choices among which disabling it must be chosen for this component
                        a2 = 1.0/static_cast<double>(numChoices+1);
                    }
                }
                fNEW = prob.fitness(xNEW);
                bool acceptable = true;
                if (nec > 0) {
                    for (size_t c=nobj; c<nobj+nec; ++c) {
                        if (fNEW[c] != 0) {
                            acceptable = false;
                            break;
                        }
                    }
                }
                if (acceptable && nic > 0) {
                    for (size_t c=nobj+nec; c<nobj+nec+nic; ++c) {
                        if (fNEW[c] > 0) {
                            acceptable = false;
                            break;
                        }
                    }
                }
//                    std::cout << "acceptable: " << acceptable << "; fNEW: " << fNEW[0] << "; " << "fOLD: " << fOLD[0] << std::endl;
                if (!acceptable) {
                    xNEW = xOLD;
                } else {
                    double a = std::exp((fOLD[0] - fNEW[0]) / currentT) * a2;
                    if (a >= 1) {
                        // accept
                        xOLD = xNEW;
                        fOLD = fNEW;
                        acp[nter]++; // Increase the number of accepted values
                        // We update the best
                        if (fNEW[0] <= best_f[0]) {
                            best_f = fNEW;
                            best_x = xNEW;
                        }
                    } else {
                        if (a > drng(m_e)) {
                            xOLD = xNEW;
                            fOLD = fNEW;
                            acp[nter]++; // Increase the number of accepted values
                            if (fNEW[0] <= best_f[0]) {
                                best_f = fNEW;
                                best_x = xNEW;
                            }
                        } else {
                            xNEW = xOLD;
                        }
                    }
                }
            }
        }

        // Cooling schedule
        currentT *= Tcoeff;
    }
    if (best_f[0] <= fit0[0]) {
        replace_individual(pop, best_x, best_f);
    }
    return pop;
}

std::string HypercubeSimulatedAnnealing::get_name() const {
    return "hypercube MCMC";
}

