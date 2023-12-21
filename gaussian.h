#pragma once

#include <iostream>
#include <cmath>

#include "params.h"


namespace TTT
{

    /*

    def mu_sigma(tau_: float, pi_: float) -> Tuple[float, float]:
        if pi_ > 0.0:
            sigma = math.sqrt(1 / pi_)
            mu = tau_ / pi_
        elif pi_ + 1e-5 < 0.0:
            raise ValueError(" sigma should be greater than 0 ")
        else:
            sigma = inf
            mu = 0.0
        return mu, sigma

    */

    std::pair<double, double> mu_sigma(const double tau_, const double pi_)
    {
        if (pi_ > 0.0)
        {
            double sigma = std::sqrt(1 / pi_);
            double mu = tau_ / pi_;
            return std::make_pair(mu, sigma);
        }
        else if (pi_ < 0.0)
        {
            throw std::invalid_argument(" sigma should be greater than 0 ");
        }
        else
        {
            double sigma = inf;
            double mu = 0.0;
            return std::make_pair(mu, sigma);
        }
    }

    /*
    class Gaussian(object):
        def __init__(self, mu: float = MU, sigma: float = SIGMA) -> None:
            if sigma >= 0.0:
                self.mu, self.sigma = mu, sigma
            else:
                raise ValueError(" sigma should be greater than 0 ")

        @property
        def tau(self) -> float:
            if self.sigma > 0.0:
                return self.mu * (self.sigma**-2)
            else:
                return inf

        @property
        def pi(self) -> float:
            if self.sigma > 0.0:
                return self.sigma**-2
            else:
                return inf

        def __add__(self, M: Self) -> Self:
            return Gaussian(self.mu + M.mu, math.sqrt(self.sigma**2 + M.sigma**2))

        def __sub__(self, M: Self) -> Self:
            return Gaussian(self.mu - M.mu, math.sqrt(self.sigma**2 + M.sigma**2))

        def __mul__(self, M: Union[float, Self]):
            if isinstance(M, float):
                if M == inf:
                    return Ninf
                else:
                    return Gaussian(M * self.mu, abs(M) * self.sigma)
            else:
                if self.sigma == 0.0:
                    mu = self.mu / ((self.sigma**2 / M.sigma**2) + 1)
                    sigma = 0.0
                elif M.sigma == 0.0:
                    mu = M.mu / ((M.sigma**2 / self.sigma**2) + 1)
                    sigma = 0.0
                else:
                    _tau, _pi = self.tau + M.tau, self.pi + M.pi
                    mu, sigma = mu_sigma(_tau, _pi)
                return Gaussian(mu, sigma)

        def __rmul__(self, other: Self):
            return self.__mul__(other)

        def __truediv__(self, M: Self) -> Self:
            _tau = self.tau - M.tau
            _pi = self.pi - M.pi
            mu, sigma = mu_sigma(_tau, _pi)
            return Gaussian(mu, sigma)

        def forget(self, gamma: float, t: float) -> Self:
            return Gaussian(self.mu, math.sqrt(self.sigma**2 + t * gamma**2))

        def delta(self, M: Self) -> Tuple[float, float]:
            return abs(self.mu - M.mu), abs(self.sigma - M.sigma)

        def exclude(self, M: Self) -> Self:
            return Gaussian(self.mu - M.mu, math.sqrt(self.sigma**2 - M.sigma**2))

        def isapprox(self, M: Self, tol: float = 1e-4) -> bool:
            return (abs(self.mu - M.mu) < tol) and (abs(self.sigma - M.sigma) < tol)
    */

    class Gaussian
    {
    public:
        Gaussian() = default; // Default constructor
        Gaussian(const double _mu, const double _sigma)
        {
            if (_sigma >= 0.0)
            {
                this->mu = _mu;
                this->sigma = _sigma;
            }
            else
            {
                throw std::invalid_argument("sigma should be greater than 0");
            }
        }
        double tau() const
        {
            if (this->sigma == 0.0)
            {
                return inf;
            }
            else if (std::isinf(this->sigma))
            {
                return 0.0;
            }
            else
            {
                return this->mu * std::pow(this->sigma * this->sigma, -2);
            }
        }
        double pi() const
        {
            if (this->sigma == 0.0)
            {
                return inf;
            }
            else if (std::isinf(this->sigma))
            {
                return 0.0;
            }
            else
            {
                return std::pow(this->sigma, -2);
            }
        }
        Gaussian operator+(const Gaussian M) const
        {
            return Gaussian(this->mu + M.mu, std::sqrt(this->sigma * this->sigma + M.sigma * M.sigma));
        }
        Gaussian operator-(const Gaussian& M) const
        {
            return Gaussian(this->mu - M.mu, std::sqrt(this->sigma * this->sigma + M.sigma * M.sigma));
        }

        Gaussian operator*(const double M) const
        {
            if (M == inf)
            {
                return Gaussian(0, inf);
            }
            else
            {
                return Gaussian(M * this->mu, std::abs(M) * this->sigma);
            }
        }

        Gaussian operator*(const Gaussian M) const
        {
            if (this->sigma == 0.0)
            {
                double _mu = this->mu / ((this->sigma * this->sigma / (M.sigma * M.sigma)) + 1);
                double _sigma = 0.0;
                return Gaussian(_mu, _sigma);
            }
            else if (M.sigma == 0.0)
            {
                double _mu = M.mu / ((M.sigma * M.sigma / (this->sigma * this->sigma)) + 1);
                double _sigma = 0.0;
                return Gaussian(_mu, _sigma);
            }
            else
            {
                double _tau = this->tau() + M.tau();
                double _pi = this->pi() + M.pi();
                auto[_mu, _sigma] = mu_sigma(_tau, _pi);
                return Gaussian(_mu, _sigma);
            }
        }

        Gaussian operator/(const Gaussian M)
        {
            double _tau = this->tau() - M.tau();
            double _pi = this->pi() - M.pi();
            double _mu, _sigma;
            _mu = _tau / _pi;
            _sigma = std::sqrt(1 / _pi);
            return Gaussian(_mu, _sigma);
        }

        Gaussian operator=(const Gaussian M)
        {
            this->mu = M.mu;
            this->sigma = M.sigma;
            return *this;
        }

        bool operator==(const Gaussian M) const
        {
            return this->mu == M.mu && this->sigma == M.sigma;
        }
        bool operator!=(const Gaussian M) const
        {
            return this->mu != M.mu || this->sigma != M.sigma;
        }
        Gaussian operator+=(const Gaussian M)
        {
            this->mu += M.mu;
            this->sigma = std::sqrt(this->sigma * this->sigma + M.sigma * M.sigma);
            return *this;
        }
        Gaussian forget(const double gamma,  const double t) const
        {
            return Gaussian(this->mu, std::sqrt(this->sigma * this->sigma + t * gamma * gamma));
        }
        Gaussian exclude(const Gaussian M) const
        {
            return Gaussian(this->mu - M.mu, std::sqrt(this->sigma * this->sigma - M.sigma * M.sigma));
        }
        bool isapprox(const Gaussian M, const double tol = 1e-4) const 
        {
            return (std::abs(this->mu - M.mu) < tol) && (std::abs(this->sigma - M.sigma) < tol);
        }
        std::pair<double, double> delta(const Gaussian M) const
        {
            return std::make_pair(std::abs(this->mu - M.mu), std::abs(this->sigma - M.sigma));
        }

        double mu;
        double sigma;
    };

    Gaussian N01 = Gaussian(0, 1);
    Gaussian N00 = Gaussian(0, 0);
    Gaussian Ninf = Gaussian(0, inf);
    Gaussian Nms = Gaussian(MU, SIGMA);
}