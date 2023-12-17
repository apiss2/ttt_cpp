# -*- coding: utf-8 -*-
"""
   TrueskillThroughTime.py
   ~~~~~~~~~~~~~~~~~~~~~~~~~~
   :copyright: (c) 2019-2023 by Gustavo Landfried.
   :license: BSD, see LICENSE for more details.
"""

import math
from typing import Tuple, Union, Self, List, Dict

__all__ = [
    "MU",
    "SIGMA",
    "BETA",
    "GAMMA",
    "P_DRAW",
    "EPSILON",
    "ITERATIONS",
    "Gaussian",
    "Player",
    "Game",
    "History",
]


#: The default standar deviation of the performances. Is the scale of estimates.
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


class Gaussian(object):
    """
    The `Gaussian` class is used to define the prior beliefs of the agents' skills

    Attributes
    ----------
    mu : float
        the mean of the `Gaussian` distribution.
    sigma :
        the standar deviation of the `Gaussian` distribution.

    """

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

    def __iter__(self) -> float:
        return iter((self.mu, self.sigma))

    def __repr__(self) -> str:
        return "N(mu={:.3f}, sigma={:.3f})".format(self.mu, self.sigma)

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
            if self.sigma == 0.0 or M.sigma == 0.0:
                mu = (
                    self.mu / ((self.sigma**2 / M.sigma**2) + 1)
                    if self.sigma == 0.0
                    else M.mu / ((M.sigma**2 / self.sigma**2) + 1)
                )
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


N01 = Gaussian(0, 1)
N00 = Gaussian(0, 0)
Ninf = Gaussian(0, inf)
Nms = Gaussian(MU, SIGMA)


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


def cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    z = -(x - mu) / (sigma * sqrt2)
    return 0.5 * erfc(z)


def pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    normalizer = (sqrt2pi * sigma) ** -1
    functional = math.exp(-((x - mu) ** 2) / (2 * sigma**2))
    return normalizer * functional


def ppf(p: float, mu: float = 0, sigma: float = 1) -> float:
    return mu - sigma * sqrt2 * erfcinv(2 * p)


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


def trunc(mu: float, sigma: float, margin: float, tie: bool) -> Tuple[float, float]:
    v, w = v_w(mu, sigma, margin, tie)
    mu_trunc = mu + sigma * v
    sigma_trunc = sigma * math.sqrt(1 - w)
    return mu_trunc, sigma_trunc


def approx(N: Gaussian, margin: float, tie: bool) -> Gaussian:
    mu, sigma = trunc(N.mu, N.sigma, margin, tie)
    return Gaussian(mu, sigma)


def compute_margin(p_draw: float, sd: float) -> float:
    return abs(ppf(0.5 - p_draw / 2, 0.0, sd))


def max_tuple(t1: List[float], t2: List[float]) -> Tuple[float, float]:
    return max(t1[0], t2[0]), max(t1[1], t2[1])


def gr_tuple(tup: List[float], threshold: float) -> bool:
    return (tup[0] > threshold) or (tup[1] > threshold)


def sortperm(xs: List[float], reverse: bool = False) -> List[int]:
    return [
        i
        for (v, i) in sorted(
            ((v, i) for (i, v) in enumerate(xs)), key=lambda t: t[0], reverse=reverse
        )
    ]


def podium(xs: List[float]) -> List[int]:
    return sortperm(xs)


def dict_diff(old: Dict[str, Gaussian], new: Dict[str, Gaussian]):
    step = (0.0, 0.0)
    for a in old:
        step = max_tuple(step, old[a].delta(new[a]))
    return step


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

    def __repr__(self) -> str:
        return "Player(Gaussian(mu=%.3f, sigma=%.3f), beta=%.3f, gamma=%.3f)" % (
            self.prior.mu,
            self.prior.sigma,
            self.beta,
            self.gamma,
        )


class team_variable(object):
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


def performance(team: List[Player], weights: List[float]) -> Gaussian:
    res = N00
    for player, w in zip(team, weights):
        res += player.performance() * w
    return res


class draw_messages(object):
    def __init__(
        self,
        prior: Gaussian = Ninf,
        prior_team: Gaussian = Ninf,
        likelihood_lose: Gaussian = Ninf,
        likelihood_win: Gaussian = Ninf,
    ):
        self.prior = prior
        self.prior_team = prior_team
        self.likelihood_lose = likelihood_lose
        self.likelihood_win = likelihood_win

    @property
    def p(self):
        return self.prior_team * self.likelihood_lose * self.likelihood_win

    @property
    def posterior_win(self):
        return self.prior_team * self.likelihood_lose

    @property
    def posterior_lose(self):
        return self.prior_team * self.likelihood_win

    @property
    def likelihood(self):
        return self.likelihood_win * self.likelihood_lose


class diff_messages(object):
    def __init__(self, prior: Gaussian = Ninf, likelihood: Gaussian = Ninf):
        self.prior = prior
        self.likelihood = likelihood

    @property
    def p(self) -> Gaussian:
        return self.prior * self.likelihood


class Game(object):
    def __init__(
        self,
        teams: List[List[Player]],
        result: List[List[Union[float, int]]] = [],
        p_draw: float = 0.0,
        weights: List[List[float]] = [],
    ):
        if len(result) and (len(teams) != len(result)):
            raise ValueError("len(result) and (len(teams) != len(result))")
        if (0.0 > p_draw) or (1.0 <= p_draw):
            raise ValueError("0.0 <= proba < 1.0")
        if (p_draw == 0.0) and (len(result) > 0) and (len(set(result)) != len(result)):
            raise ValueError(
                "(p_draw == 0.0) and (len(result)>0) and (len(set(result))!=len(result))"
            )
        if (len(weights) > 0) and (len(teams) != len(weights)):
            raise ValueError("(len(weights)>0) & (len(teams)!= len(weights))")
        if (len(weights) > 0) and (
            any([len(team) != len(weight) for (team, weight) in zip(teams, weights)])
        ):
            ValueError("(len(weights)>0) & exists i (len(teams[i]) != len(weights[i])")

        self.teams = teams
        self.result = result
        self.p_draw = p_draw
        if not weights:
            weights = [[1.0 for p in t] for t in teams]
        self.weights = weights
        self.likelihoods = []
        self.evidence = 0.0
        self.compute_likelihoods()

    def __len__(self) -> int:
        return len(self.teams)

    def size(self) -> List[int]:
        return [len(team) for team in self.teams]

    def performance(self, i: int) -> Gaussian:
        return performance(self.teams[i], self.weights[i])

    def partial_evidence(
        self, d: List[diff_messages], margin: Dict[float], tie: List[bool], e: int
    ) -> None:
        mu, sigma = d[e].prior.mu, d[e].prior.sigma
        self.evidence *= (
            cdf(margin[e], mu, sigma) - cdf(-margin[e], mu, sigma)
            if tie[e]
            else 1 - cdf(margin[e], mu, sigma)
        )

    def graphical_model(
        self,
    ) -> Tuple[
        List[int], List[team_variable], List[diff_messages], List[bool], List[float]
    ]:
        g = self
        r = (
            g.result
            if len(g.result) > 0
            else [i for i in range(len(g.teams) - 1, -1, -1)]
        )
        o = sortperm(r, reverse=True)
        t = [
            team_variable(g.performance(o[e]), Ninf, Ninf, Ninf) for e in range(len(g))
        ]
        d = [
            diff_messages(t[e].prior - t[e + 1].prior, Ninf) for e in range(len(g) - 1)
        ]

        tie = [r[o[e]] == r[o[e + 1]] for e in range(len(d))]
        margin = [
            0.0
            if g.p_draw == 0.0
            else compute_margin(
                g.p_draw,
                math.sqrt(
                    sum([a.beta**2 for a in g.teams[o[e]]])
                    + sum([a.beta**2 for a in g.teams[o[e + 1]]])
                ),
            )
            for e in range(len(d))
        ]
        g.evidence = 1.0
        return o, t, d, tie, margin

    def likelihood_analitico(self) -> Tuple[List[Gaussian], List[Gaussian]]:
        g = self
        o, t, d, tie, margin = g.graphical_model()
        g.partial_evidence(d, margin, tie, 0)
        d = d[0].prior
        
        mu_trunc, sigma_trunc = trunc(d.mu, d.sigma, margin[0], tie[0])
        
        if d.sigma == sigma_trunc:
            delta_div = d.sigma**2 * mu_trunc - sigma_trunc**2 * d.mu
            theta_div_pow2 = inf
        else:
            delta_div = (d.sigma**2 * mu_trunc - sigma_trunc**2 * d.mu) / (d.sigma**2 - sigma_trunc**2)
            theta_div_pow2 = (sigma_trunc**2 * d.sigma**2) / (d.sigma**2 - sigma_trunc**2)
        
        res = []
        for i in range(len(t)):
            team = []
            for j in range(len(g.teams[o[i]])):
                if d.sigma == sigma_trunc:
                    mu = 0.0
                else:
                    if i == 1:
                        mu = g.teams[o[i]][j].prior.mu - (delta_div - d.mu)
                    else:
                        mu = g.teams[o[i]][j].prior.mu + (delta_div - d.mu)
                sigma_analitico = math.sqrt(theta_div_pow2 + d.sigma**2 - g.teams[o[i]][j].prior.sigma ** 2)
                team.append(Gaussian(mu, sigma_analitico))
            res.append(team)
            
        if o[0] < o[1]:
            return (res[0], res[1])
        else:
            return (res[1], res[0])

    def likelihood_teams(self) -> List[Gaussian]:
        g = self
        o, t, d, tie, margin = g.graphical_model()
        step = (inf, inf)
        i = 0
        while gr_tuple(step, 1e-6) and (i < 10):
            step = (0.0, 0.0)
            for e in range(len(d) - 1):
                d[e].prior = t[e].posterior_win - t[e + 1].posterior_lose
                if i == 0:
                    g.partial_evidence(d, margin, tie, e)
                d[e].likelihood = approx(d[e].prior, margin[e], tie[e]) / d[e].prior
                likelihood_lose = t[e].posterior_win - d[e].likelihood
                step = max_tuple(step, t[e + 1].likelihood_lose.delta(likelihood_lose))
                t[e + 1].likelihood_lose = likelihood_lose
            for e in range(len(d) - 1, 0, -1):
                d[e].prior = t[e].posterior_win - t[e + 1].posterior_lose
                if (i == 0) and (e == len(d) - 1):
                    g.partial_evidence(d, margin, tie, e)
                d[e].likelihood = approx(d[e].prior, margin[e], tie[e]) / d[e].prior
                likelihood_win = t[e + 1].posterior_lose + d[e].likelihood
                step = max_tuple(step, t[e].likelihood_win.delta(likelihood_win))
                t[e].likelihood_win = likelihood_win
            i += 1
        if len(d) == 1:
            g.partial_evidence(d, margin, tie, 0)
            d[0].prior = t[0].posterior_win - t[1].posterior_lose
            d[0].likelihood = approx(d[0].prior, margin[0], tie[0]) / d[0].prior
        t[0].likelihood_win = t[1].posterior_lose + d[0].likelihood
        t[-1].likelihood_lose = t[-2].posterior_win - d[-1].likelihood
        return [t[o[e]].likelihood for e in range(len(t))]

    def hasNotOneWeights(self) -> bool:
        for t in self.weights:
            for w in t:
                if w != 1.0:
                    return True
        return False

    def compute_likelihoods(self) -> None:
        if ((len(self.teams) > 2) or self.hasNotOneWeights()):
            m_t_ft = self.likelihood_teams()
            self.likelihoods = list()
            for e in range(len(self)):
                for i in range(len(self.teams[e])):
                    if self.weights[e][i] != 0.0:
                        lh = (1 / self.weights[e][i])
                    else:
                        lh = inf
                    lh = lh * (m_t_ft[e] - self.performance(e).exclude(self.teams[e][i].prior * self.weights[e][i]))
                    self.likelihoods.append(lh)
        else:
            self.likelihoods = self.likelihood_analitico()

    def posteriors(self) -> List[List[Gaussian]]:
        return [
            [
                self.likelihoods[e][i] * self.teams[e][i].prior
                for i in range(len(self.teams[e]))
            ]
            for e in range(len(self))
        ]


class Skill(object):
    def __init__(
        self,
        forward: Gaussian = Ninf,
        backward: Gaussian = Ninf,
        likelihood: Gaussian = Ninf,
        elapsed: float = 0,
    ):
        self.forward = forward
        self.backward = backward
        self.likelihood = likelihood
        self.elapsed = elapsed


class Agent(object):
    def __init__(self, player: Player, message: Gaussian, last_time: bool) -> None:
        self.player = player
        self.message = message
        self.last_time = last_time

    def receive(self, elapsed: float) -> None:
        if self.message != Ninf:
            res = self.message.forget(self.player.gamma, elapsed)
        else:
            res = self.player.prior
        return res


def clean(agents: List[Agent], last_time: bool = False):
    for a in agents:
        agents[a].message = Ninf
        if last_time:
            agents[a].last_time = -inf


class Item(object):
    def __init__(self, name: str, likelihood: Gaussian):
        self.name = name
        self.likelihood = likelihood


class Team(object):
    def __init__(self, items: List[Item], output: Union[int, float]):
        self.items = items
        self.output = output


class Event(object):
    def __init__(self, teams: List[Team], evidence: float, weights: List[float]):
        self.teams = teams
        self.evidence = evidence
        self.weights = weights

    def __repr__(self) -> str:
        return "Event({}, {})".format(self.names, self.result)

    @property
    def names(self) -> List[str]:
        return [[item.name for item in team.items] for team in self.teams]

    @property
    def result(self) -> List[Union[int, float]]:
        return [team.output for team in self.teams]


def get_composition(events: List[Event]) -> List[List[List[str]]]:
    return [[[it.name for it in t.items] for t in e.teams] for e in events]


def get_results(events: List[Event]) -> List[List[Union[int, float]]]:
    return [[t.output for t in e.teams] for e in events]


def compute_elapsed(last_time: float, actual_time: float) -> Union[int, float]:
    return (
        0
        if last_time == -inf
        else (1 if last_time == inf else (actual_time - last_time))
    )


class Batch(object):
    def __init__(
        self,
        composition: List[List[List[str]]],
        results: List[List[Union[int, float]]] = [],
        time: float = 0,
        agents: Dict[str, Agent] = dict(),
        p_draw: float = 0.0,
        weights: List[float] = [],
    ):
        if (len(results) > 0) and (len(composition) != len(results)):
            raise ValueError("(len(results)>0) and (len(composition)!= len(results))")
        if (len(weights) > 0) and (len(composition) != len(weights)):
            raise ValueError("(len(weights)>0) & (len(composition)!= len(weights))")

        this_agents = set([a for teams in composition for team in teams for a in team])
        elapsed = dict(
            [(a, compute_elapsed(agents[a].last_time, time)) for a in this_agents]
        )

        self.skills = dict(
            [
                (a, Skill(agents[a].receive(elapsed[a]), Ninf, Ninf, elapsed[a]))
                for a in this_agents
            ]
        )
        self.events = [
            Event(
                [
                    Team(
                        [
                            Item(composition[e][t][a], Ninf)
                            for a in range(len(composition[e][t]))
                        ],
                        results[e][t]
                        if len(results) > 0
                        else len(composition[e]) - t - 1,
                    )
                    for t in range(len(composition[e]))
                ],
                0.0,
                weights if not weights else weights[e],
            )
            for e in range(len(composition))
        ]
        self.time = time
        self.agents = agents
        self.p_draw = p_draw
        self.iteration()

    def __repr__(self) -> str:
        return "Batch(time={}, events={})".format(self.time, self.events)

    def __len__(self) -> int:
        return len(self.events)

    def posterior(self, agent: Agent) -> Gaussian:
        return (
            self.skills[agent].likelihood
            * self.skills[agent].backward
            * self.skills[agent].forward
        )

    def posteriors(self) -> Dict[str, Gaussian]:
        res = dict()
        for a in self.skills:
            res[a] = self.posterior(a)
        return res

    def within_prior(self, item: Item) -> Player:
        r = self.agents[item.name].player
        mu, sigma = self.posterior(item.name) / item.likelihood
        res = Player(Gaussian(mu, sigma), r.beta, r.gamma)
        return res

    def within_priors(self, event: int) -> List[List[Player]]:  # event=0
        return [
            [self.within_prior(item) for item in team.items]
            for team in self.events[event].teams
        ]

    def iteration(self, _from: int = 0):  # self=b
        for e in range(_from, len(self)):  # e=0
            teams = self.within_priors(e)
            result = self.events[e].result
            weights = self.events[e].weights
            g = Game(teams, result, self.p_draw, weights)
            for t, team in enumerate(self.events[e].teams):
                for i, item in enumerate(team.items):
                    self.skills[item.name].likelihood = (
                        self.skills[item.name].likelihood / item.likelihood
                    ) * g.likelihoods[t][i]
                    item.likelihood = g.likelihoods[t][i]
            self.events[e].evidence = g.evidence

    def convergence(self, epsilon: float = 1e-6, iterations: int = 20) -> int:
        step, i = (inf, inf), 0
        while gr_tuple(step, epsilon) and (i < iterations):
            old = self.posteriors().copy()
            self.iteration()
            step = dict_diff(old, self.posteriors())
            i += 1
        return i

    def forward_prior_out(self, agent: Agent) -> Gaussian:
        return self.skills[agent].forward * self.skills[agent].likelihood

    def backward_prior_out(self, agent: Agent) -> Gaussian:
        N = self.skills[agent].likelihood * self.skills[agent].backward
        return N.forget(self.agents[agent].player.gamma, self.skills[agent].elapsed)

    def new_backward_info(self):
        for a in self.skills:
            self.skills[a].backward = self.agents[a].message
        return self.iteration()

    def new_forward_info(self):
        for a in self.skills:
            self.skills[a].forward = self.agents[a].receive(self.skills[a].elapsed)
        return self.iteration()


class History(object):
    def __init__(
        self,
        composition: List[List[List[str]]],
        results: List[List[Union[int, float]]] = [],
        times: List[float] = [],
        priors: Dict[str, Player] = dict(),
        mu: float = MU,
        sigma: float = SIGMA,
        beta: float = BETA,
        gamma: float = GAMMA,
        p_draw: float = P_DRAW,
        weights: List[float] = [],
    ):
        if (len(results) > 0) and (len(composition) != len(results)):
            raise ValueError("len(composition) != len(results)")
        if (len(times) > 0) and (len(composition) != len(times)):
            raise ValueError(" len(times) error ")
        if (len(weights) > 0) and (len(composition) != len(weights)):
            raise ValueError(
                "(length(weights) > 0) & (length(composition) != length(weights))"
            )

        self.size = len(composition)
        self.batches: List[Batch] = []
        self.agents = dict(
            [
                (
                    a,
                    Agent(
                        priors[a]
                        if a in priors
                        else Player(Gaussian(mu, sigma), beta, gamma),
                        Ninf,
                        -inf,
                    ),
                )
                for a in set(
                    [a for teams in composition for team in teams for a in team]
                )
            ]
        )
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma
        self.p_draw = p_draw
        self.time = len(times) > 0
        self.trueskill(composition, results, times, weights)

    def __repr__(self) -> str:
        return "History(Events={}, Batches={}, Agents={})".format(
            self.size, len(self.batches), len(self.agents)
        )

    def __len__(self) -> int:
        return self.size

    def trueskill(
        self,
        composition: List[List[List[str]]],
        results: List[List[Union[int, float]]],
        times: List[float],
        weights: List[float],
    ):
        o = sortperm(times) if len(times) > 0 else [i for i in range(len(composition))]
        i = 0
        while i < len(self):
            j, t = i + 1, i + 1 if len(times) == 0 else times[o[i]]
            while (len(times) > 0) and (j < len(self)) and (times[o[j]] == t):
                j += 1
            if len(results) > 0:
                b = Batch(
                    [composition[k] for k in o[i:j]],
                    [results[k] for k in o[i:j]],
                    t,
                    self.agents,
                    self.p_draw,
                    weights if not weights else [weights[k] for k in o[i:j]],
                )
            else:
                b = Batch(
                    [composition[k] for k in o[i:j]],
                    [],
                    t,
                    self.agents,
                    self.p_draw,
                    weights if not weights else [weights[k] for k in o[i:j]],
                )
            self.batches.append(b)
            for a in b.skills:
                self.agents[a].last_time = t if self.time else inf
                self.agents[a].message = b.forward_prior_out(a)
            i = j

    def iteration(self) -> Tuple[float, float]:
        step = (0.0, 0.0)
        clean(self.agents)
        for j in reversed(range(len(self.batches) - 1)):
            for a in self.batches[j + 1].skills:
                self.agents[a].message = self.batches[j + 1].backward_prior_out(a)
            old = self.batches[j].posteriors().copy()
            self.batches[j].new_backward_info()
            step = max_tuple(step, dict_diff(old, self.batches[j].posteriors()))
        clean(self.agents)
        for j in range(1, len(self.batches)):
            for a in self.batches[j - 1].skills:
                self.agents[a].message = self.batches[j - 1].forward_prior_out(a)
            old = self.batches[j].posteriors().copy()
            self.batches[j].new_forward_info()
            step = max_tuple(step, dict_diff(old, self.batches[j].posteriors()))

        if len(self.batches) == 1:
            old = self.batches[0].posteriors().copy()
            self.batches[0].convergence()
            step = max_tuple(step, dict_diff(old, self.batches[0].posteriors()))

        return step

    def convergence(
        self,
        epsilon: float = EPSILON,
        iterations: int = ITERATIONS,
        verbose: bool = True,
    ) -> Tuple[Tuple[float, float], int]:
        step = (inf, inf)
        i = 0
        while gr_tuple(step, epsilon) and (i < iterations):
            if verbose:
                print("Iteration = ", i, end=" ")
            step = self.iteration()
            i += 1
            if verbose:
                print(", step = ", step)
        if verbose:
            print("End")
        return step, i

    def learning_curves(self) -> Dict[Tuple[float, Gaussian]]:
        res = dict()
        for b in self.batches:
            for a in b.skills:
                t_p = (b.time, b.posterior(a))
                if a in res:
                    res[a].append(t_p)
                else:
                    res[a] = [t_p]
        return res

    def log_evidence(self) -> float:
        return sum(
            [math.log(event.evidence) for b in self.batches for event in b.events]
        )
