//
// Created by James Noeckel on 1/13/20.
//

#pragma once

#include <vector>
#include <cmath>

/**
 * Compute the derivative of signal assuming spacing h
 * @param abs take the absolute value of the derivative
 */
template <typename T>
std::vector<double> robust_derivative(const std::vector<T> &signal, bool abs=false, double h=1.0);


template <typename T>
std::vector<double> robust_derivative(const std::vector<T> &signal, bool abs, double h) {
    int N = signal.size();
    std::vector<double> deriv(signal.size());
    for (int i=0; i<signal.size(); i++) {
        int i_n2 = std::max(0, i-2);
        int i_n1 = std::max(0, i-1);
        int i_1 = std::min(N-1, i+1);
        int i_2 = std::min(N-2, i+2);
        deriv[i] = (2*(static_cast<double>(signal[i_1])-static_cast<double>(signal[i_n1])) + static_cast<double>(signal[i_2]) - static_cast<double>(signal[i_n2]))/(8.0*h);
        if (abs) deriv[i] = std::fabs(deriv[i]);
    }
    return deriv;
}