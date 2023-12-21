#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <map>

#include "params.h"
#include "gaussian.h"

namespace TTT
{

    /*

    def erfc(x: float) -> float:
        # """(http://bit.ly/zOLqbc)"""
        z = abs(x)
        t = 1.0 / (1.0 + z / 2.0)
        a = -0.82215223 + t * 0.17087277
        b = 1.48851587 + t * a
        c = -1.13520398 + t * b
        d = 0.27886807 + t * c
        e = -0.18628806 + t * d
        f = 0.09678418 + t * e
        g = 0.37409196 + t * f
        h = 1.00002368 + t * g
        r = t * math.exp(-z * z - 1.26551223 + t * h)
        return r if not (x < 0) else 2.0 - r

    */

    double erfc(const double x)
    {
        double z = std::abs(x);
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
        if (x < 0)
        {
            return 2.0 - r;
        }
        else
        {
            return r;
        }
    }

    /*
    def erfcinv(y: float) -> float:
        if y >= 2:
            return -inf
        if y < 0:
            raise ValueError("argument must be nonnegative")
        if y == 0:
            return inf
        if not (y < 1):
            y = 2 - y
        t = math.sqrt(-2 * math.log(y / 2.0))
        x = -0.70711 * ((2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t)
        for _ in [0, 1, 2]:
            err = erfc(x) - y
            x += err / (1.12837916709551257 * math.exp(-(x**2)) - x * err)
        return x if (y < 1) else -x
    */

    double erfcinv(const double y)
    {
        if (y >= 2)
        {
            return -inf;
        }
        if (y < 0)
        {
            throw std::invalid_argument("argument must be nonnegative");
        }
        if (y == 0)
        {
            return inf;
        }
        double _y;
        if (y < 1)
        {
            _y = y;
        }
        else
        {
            _y = 2 - y;
        }
        double t = sqrt(-2 * log(_y / 2.0));
        double x = -0.70711 * ((2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t);
        for (int i = 0; i < 3; i++)
        {
            double err = erfc(x) - _y;
            x += err / (1.12837916709551257 * exp(-(x * x)) - x * err);
        }
        if (_y < 1)
        {
            return x;
        }
        else
        {
            return -x;
        }
    }

    /*

    def tau_pi(mu: float, sigma: float) -> Tuple[float, float]:
        if sigma > 0.0:
            pi_ = sigma**-2
            tau_ = pi_ * mu
        elif (sigma + 1e-5) < 0.0:
            raise ValueError(" sigma should be greater than 0 ")
        else:
            pi_ = inf
            tau_ = inf
        return tau_, pi_

    */
    std::pair<double, double> tau_pi(const double mu, const double sigma)
    {
        if (sigma > 0.0)
        {
            double pi_ = sigma * sigma;
            double tau_ = pi_ * mu;
            return std::make_pair(tau_, pi_);
        }
        else if ((sigma + 1e-5) < 0.0)
        {
            throw std::invalid_argument(" sigma should be greater than 0 ");
        }
        else
        {
            double pi_ = inf;
            double tau_ = inf;
            return std::make_pair(tau_, pi_);
        }
    }

    /*

    def cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
        z = -(x - mu) / (sigma * sqrt2)
        return 0.5 * erfc(z)

    */

    double cdf(const double x, const double mu, const double sigma)
    {
        double z = -(x - mu) / (sigma * sqrt2);
        return 0.5 * erfc(z);
    }

    /*

    def pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
        normalizer = (sqrt2pi * sigma) ** -1
        functional = math.exp(-((x - mu) ** 2) / (2 * sigma**2))
        return normalizer * functional

    */
    double pdf(const double x, const double mu, const double sigma)
    {
        double normalizer = std::pow(sqrt2pi * sigma, -1);
        double functional = std::exp(-std::pow(x - mu, 2) / (2 * sigma * sigma));
        return normalizer * functional;
    }

    /*

    def ppf(p: float, mu: float = 0, sigma: float = 1) -> float:
        return mu - sigma * sqrt2 * erfcinv(2 * p)

    */

    double ppf(const double p, const double mu, const double sigma)
    {
        return mu - sigma * sqrt2 * erfcinv(2 * p);
    }

    /*

    def v_w(mu: float, sigma: float, margin: float, tie: bool) -> Tuple[float, float]:
        if not tie:
            _alpha = (margin - mu) / sigma
            v = pdf(-_alpha, 0, 1) / cdf(-_alpha, 0, 1)
            w = v * (v + (-_alpha))
        else:
            _alpha = (-margin - mu) / sigma
            _beta = (margin - mu) / sigma
            v = (pdf(_alpha, 0, 1) - pdf(_beta, 0, 1)) / (
                cdf(_beta, 0, 1) - cdf(_alpha, 0, 1)
            )
            u = (_alpha * pdf(_alpha, 0, 1) - _beta * pdf(_beta, 0, 1)) / (
                cdf(_beta, 0, 1) - cdf(_alpha, 0, 1)
            )
            w = -(u - v**2)
        return v, w

    */
    std::pair<double, double> v_w(const double mu, const double sigma, const double margin, const bool tie)
    {
        if (tie)
        {
            double _alpha = (-margin - mu) / sigma;
            double _beta = (margin - mu) / sigma;
            double v = (pdf(_alpha, 0, 1) - pdf(_beta, 0, 1)) / (cdf(_beta, 0, 1) - cdf(_alpha, 0, 1));
            double u = (_alpha * pdf(_alpha, 0, 1) - _beta * pdf(_beta, 0, 1)) / (cdf(_beta, 0, 1) - cdf(_alpha, 0, 1));
            double w = -(u - v * v);
            return std::make_pair(v, w);
        }
        else
        {
            double _alpha = (margin - mu) / sigma;
            double v = pdf(-_alpha, 0, 1) / cdf(-_alpha, 0, 1);
            double w = v * (v + (-_alpha));
            return std::make_pair(v, w);
        }
    }

    /*

    def trunc(mu: float, sigma: float, margin: float, tie: bool) -> Tuple[float, float]:
        v, w = v_w(mu, sigma, margin, tie)
        mu_trunc = mu + sigma * v
        sigma_trunc = sigma * math.sqrt(1 - w)
        return mu_trunc, sigma_trunc

    */
    std::pair<double, double> trunc(const double mu, const double sigma, const double margin, const bool tie)
    {
        auto [v, w] = v_w(mu, sigma, margin, tie);
        double mu_trunc = mu + sigma * v;
        double sigma_trunc = sigma * std::sqrt(1 - w);
        return std::make_pair(mu_trunc, sigma_trunc);
    }

    /*

    def approx(N: Gaussian, margin: float, tie: bool) -> Gaussian:
        mu, sigma = trunc(N.mu, N.sigma, margin, tie)
        return Gaussian(mu, sigma)

    */

    Gaussian approx(const Gaussian N, const double margin, const bool tie)
    {
        auto [mu, sigma] = trunc(N.mu, N.sigma, margin, tie);
        return Gaussian(mu, sigma);
    }

    /*

    def compute_margin(p_draw: float, sd: float) -> float:
        return abs(ppf(0.5 - p_draw / 2, 0.0, sd))

    */
    double compute_margin(const double p_draw, const double sd)
    {
        return std::abs(ppf(0.5 - p_draw / 2, 0.0, sd));
    }

    /*

    def max_tuple(t1: List[float], t2: List[float]) -> Tuple[float, float]:
        return max(t1[0], t2[0]), max(t1[1], t2[1])

    */

    std::pair<double, double> max_tuple(const std::pair<double, double> t1, const std::pair<double, double> t2)
    {
        return std::make_pair(std::max(t1.first, t2.first), std::max(t1.second, t2.second));
    }

    /*

    def gr_tuple(tup: List[float], threshold: float) -> bool:
        return (tup[0] > threshold) or (tup[1] > threshold)

    */

    bool gr_tuple(const std::pair<double, double> tup, const double threshold)
    {
        return (tup.first > threshold) || (tup.second > threshold);
    }

    /*

    def sortperm(xs: List[float], reverse: bool = False) -> List[int]:
        sorted_list = sorted(((v, i) for i, v in enumerate(xs)), key=lambda t: t[0], reverse=reverse)
        return [i for v, i in sorted_list]

    */

    std::vector<int> sortperm(const std::vector<double> xs, const bool reverse = false)
    {
        std::vector<std::pair<double, int>> sorted_list;
        for (int i = 0; i < xs.size(); i++)
        {
            sorted_list.push_back(std::make_pair(xs[i], i));
        }
        std::sort(sorted_list.begin(), sorted_list.end(), [](std::pair<double, int> t1, std::pair<double, int> t2)
                  { return t1.first < t2.first; });
        std::vector<int> result;
        for (int i = 0; i < sorted_list.size(); i++)
        {
            result.push_back(sorted_list[i].second);
        }
        return result;
    }

    /*

    def podium(xs: List[float]) -> List[int]:
        return sortperm(xs)

    */
    std::vector<int> podium(const std::vector<double> xs)
    {
        return sortperm(xs);
    }

    /*

    def dict_diff(old: Dict[str, Gaussian], new: Dict[str, Gaussian]):
        step = (0.0, 0.0)
        for a in old:
            step = max_tuple(step, old[a].delta(new[a]))
        return step

    */

    std::pair<double, double> dict_diff(const std::map<std::string, Gaussian> &old_map, const std::map<std::string, Gaussian> &new_map)
    {
        std::pair<double, double> step = std::make_pair(0.0, 0.0);
        for (auto &a : old_map)
        {
            std::string key = a.first;
            step = max_tuple(step, old_map.at(key).delta(new_map.at(key)));
        }
        return step;
    }
}