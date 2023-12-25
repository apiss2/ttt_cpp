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
    class Player(object):
        def __init__(
            self,
            prior: Gaussian = Gaussian(MU, SIGMA),
            beta: float = BETA,
            gamma: float = GAMMA,
            prior_draw: Gaussian = Ninf,
        ):
            self.prior = prior
            self.beta = beta
            self.gamma = gamma
            self.prior_draw = prior_draw

        def performance(self) -> Gaussian:
            return Gaussian(self.prior.mu, math.sqrt(self.prior.sigma**2 + self.beta**2))

    */
    class Player
    {
    public:
        Player(Gaussian prior = Gaussian(MU, SIGMA), double beta = BETA, double gamma = GAMMA, Gaussian prior_draw = Ninf)
            : prior(prior), beta(beta), gamma(gamma), prior_draw(prior_draw) {}

        Gaussian performance() const
        {
            return Gaussian(prior.mu, std::sqrt(prior.sigma * prior.sigma + beta * beta));
        }

        Gaussian prior;
        double beta;
        double gamma;
        Gaussian prior_draw;
    };

    /*
    class TeamVariable(object):
        def __init__(
            self,
            prior: Gaussian = Ninf,
            likelihood_lose: Gaussian = Ninf,
            likelihood_win: Gaussian = Ninf,
            likelihood_draw: Gaussian = Ninf,
        ) -> None:
            self.prior = prior
            self.likelihood_lose = likelihood_lose
            self.likelihood_win = likelihood_win
            self.likelihood_draw = likelihood_draw

        @property
        def p(self) -> Gaussian:
            return (
                self.prior
                * self.likelihood_lose
                * self.likelihood_win
                * self.likelihood_draw
            )

        @property
        def posterior_win(self) -> Gaussian:
            return self.prior * self.likelihood_lose * self.likelihood_draw

        @property
        def posterior_lose(self) -> Gaussian:
            return self.prior * self.likelihood_win * self.likelihood_draw

        @property
        def likelihood(self) -> Gaussian:
            return self.likelihood_win * self.likelihood_lose * self.likelihood_draw

    */
    class TeamVariable
    {
    public:
        TeamVariable(const Gaussian prior = Ninf, const Gaussian likelihood_lose = Ninf, Gaussian likelihood_win = Ninf, const Gaussian likelihood_draw = Ninf)
            : prior(prior), likelihood_lose(likelihood_lose), likelihood_win(likelihood_win), likelihood_draw(likelihood_draw) {}

        Gaussian p() const
        {
            return prior * likelihood_lose * likelihood_win * likelihood_draw;
        }

        Gaussian posterior_win() const
        {
            return prior * likelihood_lose * likelihood_draw;
        }

        Gaussian posterior_lose() const
        {
            return prior * likelihood_win * likelihood_draw;
        }

        Gaussian likelihood() const
        {
            return likelihood_win * likelihood_lose * likelihood_draw;
        }

        Gaussian prior;
        Gaussian likelihood_lose;
        Gaussian likelihood_win;
        Gaussian likelihood_draw;
    };
    /*
    def team_performance(team: List[Player], weights: List[float]) -> Gaussian:
        res = N00
        for player, w in zip(team, weights):
            res += player.performance() * w
        return res
    */
    Gaussian team_performance(const std::vector<Player> &team, const std::vector<double> &weights)
    {
        Gaussian res = N00;
        for (int i = 0; i < team.size(); i++)
        {
            res += team[i].performance() * weights[i];
        }
        return res;
    }
    /*
    class DiffMessage(object):
        def __init__(self, prior: Gaussian = Ninf, likelihood: Gaussian = Ninf):
            self.prior = prior
            self.likelihood = likelihood

        @property
        def p(self) -> Gaussian:
            return self.prior * self.likelihood
    */
    class DiffMessage
    {
    public:
        DiffMessage(const Gaussian prior = Ninf, const Gaussian likelihood = Ninf)
            : prior(prior), likelihood(likelihood) {}

        Gaussian p() const
        {
            return prior * likelihood;
        }

        Gaussian prior;
        Gaussian likelihood;
    };
    
}