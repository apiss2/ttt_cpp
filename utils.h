#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <map>

#include "ttt_params.h"
#include "gaussian.h"
#include "team.h"

namespace TTT
{
    double erfc(double x)
    {
        double z = std::fabs(x);
        double t = 1.0 / (1.0 + z / 2.0);
        double a = -0.82215223 + t * 0.17087277;
        double b = 1.48851587 + t * a;
        double c = -1.13520398 + t * b;
        double d = 0.27886807 + t * c;
        double e = -0.18628806 + t * d;
        double f = 0.09678418 + t * e;
        double g = 0.37409196 + t * f;
        double h = 1.00002368 + t * g;
        double r = t * std::exp(-z * z - 1.26551223 + t * h);
        if (x < 0){
            return 2.0 - r;
        }
        else
        {
            return r;
        }
    }

    double erfcinv(double y)
    {
        if (y >= 2)
        {
            return -TTT::inf;
        }
        if (y < 0)
        {
            throw std::invalid_argument("argument must be nonnegative");
        }
        if (y == 0)
        {
            return TTT::inf;
        }
        if (!(y < 1))
        {
            y = 2 - y;
        }
        double t = std::sqrt(-2 * std::log(y / 2.0));
        double x = -0.70711 * ((2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t);
        for (int i = 0; i < 3; i++)
        {
            double err = erfc(x) - y;
            x += err / (1.12837916709551257 * std::exp(-(x * x)) - x * err);
        }
        if (y < 1){
            return x;
        }
        else
        {
            return -x;
        }
    }

    std::pair<double, double> tau_pi(double mu, double sigma)
    {
        if (sigma > 0.0)
        {
            double pi_ = pow(sigma, -2);
            double tau_ = pi_ * mu;
            return std::make_pair(tau_, pi_);
        }
        else if ((sigma + 1e-5) < 0.0)
        {
            throw std::invalid_argument(" sigma should be greater than 0 ");
        }
        else
        {
            return std::make_pair(TTT::inf, TTT::inf);
        }
    }

    double cdf(double x, double mu = 0, double sigma = 1)
    {
        double z = -(x - mu) / (sigma * TTT::sqrt2);
        return 0.5 * erfc(z);
    }

    double pdf(double x, double mu = 0, double sigma = 1)
    {
        double normalizer = std::pow(TTT::sqrt2pi * sigma, -1);
        double functional = std::exp(-((x - mu) * (x - mu)) / (2 * sigma * sigma));
        return normalizer * functional;
    }

    double ppf(double p, double mu = 0, double sigma = 1)
    {
        return mu - sigma * TTT::sqrt2 * erfcinv(2 * p);
    }

    std::pair<double, double> v_w(double mu, double sigma, double margin, bool tie)
    {
        if (!tie)
        {
            double _alpha = (margin - mu) / sigma;
            double v = pdf(-_alpha, 0, 1) / cdf(-_alpha, 0, 1);
            double w = v * (v + (-_alpha));
            return std::make_pair(v, w);
        }
        else
        {
            double _alpha = (-margin - mu) / sigma;
            double _beta = (margin - mu) / sigma;
            double v = (pdf(_alpha, 0, 1) - pdf(_beta, 0, 1)) / (cdf(_beta, 0, 1) - cdf(_alpha, 0, 1));
            double u = (_alpha * pdf(_alpha, 0, 1) - _beta * pdf(_beta, 0, 1)) / (cdf(_beta, 0, 1) - cdf(_alpha, 0, 1));
            double w = -(u - v * v);
            return std::make_pair(v, w);
        }
    }

    std::pair<double, double> trunc(double mu, double sigma, double margin, bool tie)
    {
        auto[v, w] = v_w(mu, sigma, margin, tie);
        double mu_trunc = mu + sigma * v;
        double sigma_trunc = sigma * std::sqrt(1 - w);
        return std::make_pair(mu_trunc, sigma_trunc);
    }

    Gaussian approx(Gaussian &N, double margin, bool tie)
    {
        auto[mu, sigma] = trunc(N.getMu(), N.getSigma(), margin, tie);
        return Gaussian(mu, sigma);
    }

    double compute_margin(double p_draw, double sd)
    {
        return std::fabs(ppf(0.5 - p_draw / 2, 0.0, sd));
    }

    std::pair<double, double> max_tuple(std::pair<double, double> &t1, std::pair<double, double> &t2)
    {
        return std::make_pair(std::max(t1.first, t2.first), std::max(t1.second, t2.second));
    }

    bool gr_tuple(std::pair<double, double> &tup, double threshold)
    {
        return (tup.first > threshold) || (tup.second > threshold);
    }

    std::vector<int> sortperm(const std::vector<double> &xs, bool reverse = false)
    {
        std::vector<std::pair<double, int>> pairs;
        for (int i = 0; i < xs.size(); i++)
        {
            pairs.push_back(std::make_pair(xs[i], i));
        }
        std::sort(
            pairs.begin(), pairs.end(),
            [reverse](const std::pair<double, int>& a, const std::pair<double, int>& b)
        {
            if (reverse) {
                return a.first > b.first;
            } else {
                return a.first < b.first;
            }
        });
        std::vector<int> sorted_indices;
        for (const auto& pair : pairs) {
            sorted_indices.push_back(pair.second);
        }
        return sorted_indices;
    }

    std::pair<double, double> dict_diff(std::map<std::string, Gaussian> &old_dict, std::map<std::string, Gaussian> &new_dict)
    {
        std::pair<double, double> step = std::make_pair(0.0, 0.0);
        for (auto it = old_dict.begin(); it != old_dict.end(); it++)
        {
            Gaussian old_gaussian = it->second;
            Gaussian new_gaussian = new_dict[it->first];
            std::pair<double, double> delta = old_gaussian.delta(new_gaussian);
            step = max_tuple(step, delta);
        }
        return step;
    }

    class DiffMessage
    {
        Gaussian prior;
        Gaussian likelihood;
    public:
        DiffMessage(Gaussian p, Gaussian l) : prior(p), likelihood(l) {}
        Gaussian p()
        {
            return prior * likelihood;
        }
        
        Gaussian getPrior() const {
            return this->prior;
        }
        
        void setPrior(Gaussian p){
            this->prior = p;
        }

        Gaussian getLikelihood() const {
            return this->likelihood;
        }

        void setLikelihood(Gaussian l){
            this->likelihood = l;
        }
    };

    void clean(std::map<std::string, Agent> &agents, bool last_time = false)
    {
        for (auto &agent_pair : agents)
        {
            std::string a = agent_pair.first;
            agents[a].setMessage(Ninf);
            if (last_time)
            {
                agents[a].setLasttime(-TTT::inf);
            }
        }
    }

    double compute_elapsed(double last_time, double actual_time)
    {
        if (last_time == -TTT::inf)
        {
            return 0;
        }
        else if (last_time == TTT::inf)
        {
            return 1;
        }
        else{
            return actual_time - last_time;
        }
    }
}