#pragma once

#include <iostream>
#include <cmath>
#include "gaussian.h"
#include "ttt_params.h"

namespace TTT
{
    std::pair<double, double> mu_sigma(double tau_, double pi_)
    {
        if (pi_ > 0.0)
        {
            double sigma = sqrt(1 / pi_);
            double mu = tau_ / pi_;
            return std::make_pair(mu, sigma);
        }
        else if (pi_ + 1e-5 < 0.0)
        {
            throw std::invalid_argument(" sigma should be greater than 0 ");
        }
        else
        {
            return std::make_pair(0.0, INFINITY);
        }
    }

    class Gaussian
    {
    private:
        double mu;
        double sigma;

    public:
        Gaussian(const double _mu = TTT::MU, const double _sigma = TTT::SIGMA)
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

        double getTau() const
        {
            if (sigma > 0.0)
            {
                return mu * std::pow(sigma, -2);
            }
            else
            {
                return INFINITY;
            }
        }

        double getPi() const
        {
            if (sigma > 0.0)
            {
                return std::pow(sigma, -2);
            }
            else
            {
                return INFINITY;
            }
        }

        double getMu() const { return this->mu; }

        double getSigma() const { return this->sigma; }

        std::pair<double, double> delta (const Gaussian M) const
        {
            return std::make_pair(std::abs(mu - M.getMu()), std::abs(sigma - M.getSigma()));
        }

        bool isApprox(const Gaussian M, const double tol = 1e-4) const
        {
            return (std::abs(this->getMu() - M.getMu()) < tol) && (std::abs(this->getSigma() - M.getSigma()) < tol);
        }

        Gaussian operator+(const Gaussian &M) const
        {
            return Gaussian(this->getMu() + M.getMu(), std::sqrt(std::pow(this->getSigma(), 2) + std::pow(M.getSigma(), 2)));
        }

        Gaussian operator-(const Gaussian &M) const
        {
            return Gaussian(this->getMu() - M.getMu(), std::sqrt(std::pow(this->getSigma(), 2) + std::pow(M.getSigma(), 2)));
        }

        Gaussian operator*(const double M) const
        {
            if (std::isinf(M))
            {
                // Return Ninf
                return Gaussian(0, TTT::inf);
            }
            else
            {
                return Gaussian(M * this->getMu(), std::abs(M) * this->getSigma());
            }
        }

        Gaussian operator*(const Gaussian &M) const
        {
            double new_mu;
            double new_sigma;
            if (this->getSigma() == 0.0)
            {
                new_mu = this->getMu() / ((std::pow(this->getSigma(), 2) / std::pow(M.getSigma(), 2)) + 1);
                new_sigma = 0.0;
            }
            else if (M.getSigma() == 0.0)
            {
                new_mu = M.getMu() / ((std::pow(M.getSigma(), 2) / std::pow(this->getSigma(), 2)) + 1);
                new_sigma = 0.0;
            }
            else
            {
                double new_tau = this->getTau() + M.getTau();
                double new_pi = this->getPi() + M.getPi();
                auto [mu, sigma] = TTT::mu_sigma(new_tau, new_pi);
            }
            return Gaussian(new_mu, new_sigma);
        }

        Gaussian operator/(const Gaussian &M) const
        {
            double _tau = this->getTau() - M.getTau();
            double _pi = this->getPi() - M.getPi();
            auto [mu, sigma] = TTT::mu_sigma(_tau, _pi);
            return Gaussian(mu, sigma);
        }

        bool operator==(const Gaussian &M) const
        {
            return (this->getMu() == M.getMu()) && (this->getSigma() == M.getSigma());
        }

        bool operator!=(const Gaussian &M) const
        {
            return !((this->getMu() == M.getMu()) && (this->getSigma() == M.getSigma()));
        }

        Gaussian forget(const double gamma, const double t) const
        {
            return Gaussian(this->getMu(), std::sqrt(std::pow(this->getSigma(), 2) + t * std::pow(gamma, 2)));
        }

        Gaussian exclude(const Gaussian &M) const
        {
            return Gaussian(this->getMu() - M.getMu(), std::sqrt(std::pow(this->getSigma(), 2) - std::pow(M.getSigma(), 2)));
        }
    };

    // const Gaussian N01 = Gaussian(0, 1);
    const Gaussian N00 = Gaussian(0, 0);
    const Gaussian Ninf = Gaussian(0, TTT::inf);
    // const Gaussian Nms = Gaussian(TTT::MU, TTT::SIGMA);
}