#pragma once
#include <cmath>

namespace TTT {
    using team_weight = std::vector<double>;
    using race_weight = std::vector<team_weight>;
    using team_keys = std::vector<std::string>;
    using race_keys = std::vector<team_keys>;
    using team_weight = std::vector<double>;
    using race_weight = std::vector<team_weight>;

    const double BETA = 1.0;
    const double MU = 0.0;
    const double SIGMA = BETA * 6;
    const double GAMMA = BETA * 0.03;
    const double P_DRAW = 0.0;
    const double EPSILON = 1e-6;
    const int ITERATIONS = 30;
    const double sqrt2 = std::sqrt(2);
    const double sqrt2pi = std::sqrt(2 * M_PI);
    const double inf = std::numeric_limits<double>::infinity();
    const double PI = std::pow(SIGMA, -2);
    const double TAU = MU * PI;
}