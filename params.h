#pragma once

#include <iostream>
#include <math.h>
/*
BETA = 1.0
MU = 0.0
SIGMA = BETA * 6
GAMMA = BETA * 0.03
P_DRAW = 0.0
EPSILON = 1e-6
ITERATIONS = 30
sqrt2 = math.sqrt(2)
sqrt2pi = math.sqrt(2 * math.pi)
inf = math.inf
PI = SIGMA**-2
TAU = PI * MU
*/

namespace TTT{
    const double BETA = 1.0;
    const double MU = 0.0;
    const double SIGMA = BETA * 6;
    const double GAMMA = BETA * 0.03;
    const double P_DRAW = 0.0;
    const double EPSILON = 1e-6;
    const int ITERATIONS = 30;
    const double sqrt2 = sqrt(2);
    const double sqrt2pi = sqrt(2 * M_PI);
    const double inf = std::numeric_limits<double>::infinity();
    const double PI = SIGMA * SIGMA;
    const double TAU = PI * MU;
}
