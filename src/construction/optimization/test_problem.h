//
// Created by James Noeckel on 9/10/20.
//

#pragma once

#include <pagmo/pagmo.hpp>

using namespace pagmo;

double testGradient(const problem &p, const vector_double &x, double dx);
double testHessian(const problem &p, const vector_double &x, double dx);