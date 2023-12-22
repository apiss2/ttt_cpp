# -*- coding: utf-8 -*-
"""
   TrueskillThroughTime.py
   ~~~~~~~~~~~~~~~~~~~~~~~~~~
   :copyright: (c) 2019-2023 by Gustavo Landfried.
   :license: BSD, see LICENSE for more details.
"""

import math
from typing import Tuple, Union, Self, List, Dict, Set

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
    sorted_list = sorted(((v, i) for i, v in enumerate(xs)), key=lambda t: t[0], reverse=reverse)
    return [i for v, i in sorted_list]


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


def team_performance(team: List[Player], weights: List[float]) -> Gaussian:
    res = N00
    for player, w in zip(team, weights):
        res += player.performance() * w
    return res


class DiffMessage(object):
    def __init__(self, prior: Gaussian = Ninf, likelihood: Gaussian = Ninf):
        self.prior = prior
        self.likelihood = likelihood

    @property
    def p(self) -> Gaussian:
        return self.prior * self.likelihood


class GraphicalModel:
    def __init__(self, teams: List[List[Player]], _result: List[float], weights: List[float], p_draw: float) -> None:
        self.evidence: float = 1.0
        self.result: List[float] = self.init_result(_result, teams)
        self.order: List[int] = sortperm(self.result, reverse=True)
        self.team_variables: List[TeamVariable] = self.init_team_variables(teams, weights)
        self.diff_messages: List[DiffMessage] = self.init_diff_messages()
        self.tie: List[bool] = self.init_tie()
        self.margins: List[float] = self.init_margin(p_draw, teams)

    def partial_evidence(self, e: int) -> None:
        mu, sigma = self.diff_messages[e].prior.mu, self.diff_messages[e].prior.sigma
        if self.tie[e]:
            _mul = cdf(self.margins[e], mu, sigma) - cdf(-self.margins[e], mu, sigma)
        else:
            _mul = 1 - cdf(self.margins[e], mu, sigma)
        self.evidence = self.evidence * _mul

    def update_from_front(self, step: Tuple[float, float], i: int):
        for e in range(len(self.diff_messages) - 1):
            # 更新_1
            self.diff_messages[e].prior = self.team_variables[e].posterior_win - self.team_variables[e + 1].posterior_lose
            if i == 0:
                # このクラスのevidence値を更新
                self.partial_evidence(self.diff_messages, self.margins, self.tie, e)
            # 更新_2
            self.diff_messages[e].likelihood = approx(self.diff_messages[e].prior, self.margins[e], self.tie[e]) / self.diff_messages[e].prior
            likelihood_lose = self.team_variables[e].posterior_win - self.diff_messages[e].likelihood
            step = max_tuple(step, self.team_variables[e + 1].likelihood_lose.delta(likelihood_lose))
            # 更新_3
            self.team_variables[e + 1].likelihood_lose = likelihood_lose

    def update_from_back(self, step: Tuple[float, float], i: int):
        # 勝利側の値を更新
        for e in range(len(self.diff_messages) - 1, 0, -1):
            # 更新_1
            self.diff_messages[e].prior = self.team_variables[e].posterior_win - self.team_variables[e + 1].posterior_lose
            if (i == 0) and (e == len(self.diff_messages) - 1):
                # このクラスのevidence値を更新
                self.partial_evidence(self.diff_messages, self.margins, self.tie, e)
            # 更新_2
            self.diff_messages[e].likelihood = approx(self.diff_messages[e].prior, self.margins[e], self.tie[e]) / self.diff_messages[e].prior
            likelihood_win = self.team_variables[e + 1].posterior_lose + self.diff_messages[e].likelihood
            step = max_tuple(step, self.team_variables[e].likelihood_win.delta(likelihood_win))
            # 更新_3
            self.team_variables[e].likelihood_win = likelihood_win
   
    def init_result(self, result: List[float], teams: List[List[Player]]):
        if len(result) > 0:
            return result
        else:
            return [i for i in range(len(teams) -1, -1, -1)]

    def init_team_variables(self, teams: List[List[Player]], weights) -> List[TeamVariable]:
        ret: List[TeamVariable] = list()
        for team_idx in range(len(teams)):
            idx = self.order[team_idx]
            _p = team_performance(teams[idx], weights)
            ret.append(TeamVariable(_p, Ninf, Ninf, Ninf))
        return ret
   
    def init_diff_messages(self) -> List[DiffMessage]:
        ret: List[DiffMessage] = list()
        for idx in range(len(self.team_variables) - 1):
            ret.append(DiffMessage(self.team_variables[idx].prior - self.team_variables[idx + 1].prior, Ninf))
        return ret
   
    def init_tie(self) -> List[bool]:
        ret: List[bool] = list()
        for e in range(len(self.diff_messages)):
            ret.append(self.result[self.order[e]] == self.result[self.order[e + 1]])
        return ret
   
    def init_margin(self, p_draw, teams: List[List[Player]]):
        ret: List[float] = list()
        for idx in range(len(self.diff_messages)):
            if p_draw == 0.0:
                ret.append(0.0)
            else:
                _sum1 = sum([a.beta**2 for a in teams[self.order[idx]]])
                _sum2 = sum([a.beta**2 for a in teams[self.order[idx + 1]]])
                _m = compute_margin(p_draw, math.sqrt(_sum1 + _sum2))
                ret.append(_m)
        return ret


class Game(object):
    def __init__(
        self,
        teams: List[List[Player]],
        result: List[float] = [],
        p_draw: float = 0.0,
        weights: List[float] = [],
    ):
        self.check_inputs(result, teams, p_draw, weights)
        self.teams: List[List[Player]] = teams
        self.result: List[float] = result
        self.p_draw: float = p_draw
        self.weights: List[float] = weights
        self.likelihoods: List[List[Gaussian]] = []
        self.evidence: float = 0.0
        self.compute_likelihoods()

    def check_inputs(self, result, teams, p_draw, weights):
        if len(result):
            assert len(teams) == len(result)
        assert 0 < p_draw < 1
        for team in teams:
            assert len(team) == len(weights)

    def performance(self, i: int) -> Gaussian:
        return team_performance(self.teams[i], self.weights)

    def likelihood_analitico(self) -> List[List[Gaussian]]:
        grm = GraphicalModel(self.teams, self.result, self.weights, self.p_draw)
        grm.partial_evidence(0)
        diffmsg = grm.diff_messages[0].prior
        margin = grm.margins[0]
        tie = grm.tie[0]

        mu_trunc, sigma_trunc = trunc(diffmsg.mu, diffmsg.sigma, margin, tie)
        delta_div = diffmsg.sigma**2 * mu_trunc - sigma_trunc**2 * diffmsg.mu
        if diffmsg.sigma == sigma_trunc:
            theta_div_pow2 = inf
        else:
            _div = diffmsg.sigma**2 - sigma_trunc**2
            delta_div = delta_div / _div
            theta_div_pow2 = (sigma_trunc**2 * diffmsg.sigma**2) / _div

        # チームごとに計算
        res = []
        for team_idx in range(len(grm.team_variables)):
            team = []
            for j in range(len(self.teams[grm.order[team_idx]])):
                if diffmsg.sigma == sigma_trunc:
                    mu = 0.0
                else:
                    _mu = self.teams[grm.order[team_idx]][j].prior.mu
                    _d = delta_div - diffmsg.mu
                    if team_idx == 1:
                        mu = _mu - _d
                    else:
                        mu = _mu + _d
                _team_sigma = self.teams[grm.order[team_idx]][j].prior.sigma ** 2
                sigma_analitico = math.sqrt(theta_div_pow2 + diffmsg.sigma**2 - _team_sigma)
                team.append(Gaussian(mu, sigma_analitico))
            res.append(team)
        # evidence更新
        self.evidence = grm.evidence
        # 大小関係確認して返却
        if grm.order[0] < grm.order[1]:
            return [res[0], res[1]]
        else:
            return [res[1], res[0]]

    def likelihood_teams(self) -> List[Gaussian]:
        grm = GraphicalModel(self.teams, self.result, self.weights)
        step = (inf, inf)
        i = 0
        while gr_tuple(step, 1e-6) and (i < 10):
            step = (0.0, 0.0)
            grm.update_from_front(step, i)
            grm.update_from_back(step, i)
            i += 1

        if len(grm.diff_messages) == 1:
            # evidence値を更新
            grm.partial_evidence(0)
            # diff_messages更新
            grm.diff_messages[0].prior = grm.team_variables[0].posterior_win - grm.team_variables[1].posterior_lose
            grm.diff_messages[0].likelihood = approx(grm.diff_messages[0].prior, grm.margins[0], grm.tie[0]) / grm.diff_messages[0].prior

        # Gameクラスのevidenceを更新
        self.evidence = grm.evidence
        grm.team_variables[0].likelihood_win = grm.team_variables[1].posterior_lose + grm.diff_messages[0].likelihood
        grm.team_variables[-1].likelihood_lose = grm.team_variables[-2].posterior_win - grm.diff_messages[-1].likelihood
        return [grm.team_variables[grm.order[e]].likelihood for e in range(len(grm.team_variables))]

    def hasNotOneWeights(self) -> bool:
        for w in self.weights:
            if w != 1.0:
                return True
        return False

    def compute_likelihoods(self) -> None:
        if (len(self.teams) > 2) or self.hasNotOneWeights():
            m_t_ft = self.likelihood_teams()
            self.likelihoods: List[List[Gaussian]] = list()
            for e in range(len(self.teams)):
                likelihood_e = list()
                for i in range(len(self.teams[e])):
                    if self.weights[i] != 0.0:
                        _lh = 1 / self.weights[i]
                    else:
                        _lh = inf
                    lh: Gaussian = _lh * (m_t_ft[e] - self.performance(e).exclude(self.teams[e][i].prior * self.weights[i]))
                    likelihood_e.append(lh)
                self.likelihoods.append(likelihood_e)
        else:
            self.likelihoods = self.likelihood_analitico()


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


def clean(agents: Dict[str, Agent], last_time: bool = False):
    for a in agents:
        agents[a].message = Ninf
        if last_time:
            agents[a].last_time = -inf


class Item(object):
    def __init__(self, name: str, likelihood: Gaussian):
        self.name = name
        self.likelihood = likelihood


class Team(object):
    def __init__(self, items: List[Item], output: float):
        self.items = items
        self.output = output


class Event(object):
    def __init__(self, teams: List[Team], evidence: float, weights: List[float]):
        self.teams = teams
        self.evidence = evidence
        self.weights = weights

    @property
    def result(self) -> List[float]:
        return [team.output for team in self.teams]


def compute_elapsed(last_time: float, actual_time: float) -> float:
    if last_time == -inf:
        return 0
    else:
        if last_time == inf:
            return 1
        else:
            return actual_time - last_time


class Batch(object):
    def __init__(
        self,
        games: List[List[List[str]]],
        results: List[List[float]],
        time: float,
        agents: Dict[str, Agent],
        p_draw: float,
        weights: List[float],
    ):
        self.agents = agents
        self.time = time
        self.p_draw = p_draw
        self.weights = weights
        batch_unique_agents = self._get_agent_set(games)
        self._init_skills(batch_unique_agents)
        self._init_events(games, results)
        self.iteration()

    def _get_agent_set(self, games: List[List[List[str]]]):
        this_batch_agents = set()
        for teams in games:
            for team in teams:
                for agent in team:
                    this_batch_agents.add(agent)
        return this_batch_agents

    def _init_skills(self, agents: Set[str]):
        self.skills: Dict[str, Skill] = dict()
        for a in agents:
            elapsed = compute_elapsed(self.agents[a].last_time, self.time)
            self.skills[a] = Skill(self.agents[a].receive(elapsed), Ninf, Ninf, elapsed)

    def _init_events(self, games, results):
        self.events: List[Event] = list()
        for event_idx in range(len(games)):
            event_teams = []
            for team_idx in range(len(games[event_idx])):
                team_items = []
                for a in range(len(games[event_idx][team_idx])):
                    team_items.append(Item(games[event_idx][team_idx][a], Ninf))
                if len(results) > 0:
                    team_result = results[event_idx][team_idx]
                else:
                    team_result = len(games[event_idx]) - team_idx - 1
                team = Team(team_items, team_result)
                event_teams.append(team)
            event = Event(event_teams, 0.0, self.weights)
            self.events.append(event)

    def posterior(self, agent: Agent) -> Gaussian:
        return (
            self.skills[agent].likelihood
            * self.skills[agent].backward
            * self.skills[agent].forward
        )

    def within_prior(self, item: Item) -> Player:
        r = self.agents[item.name].player
        mu, sigma = self.posterior(item.name) / item.likelihood
        res = Player(Gaussian(mu, sigma), r.beta, r.gamma)
        return res

    def within_priors(self, event_idx: int) -> List[List[Player]]:  # event=0
        result = list()
        for team in self.events[event_idx].teams:
            inner_result = list()
            for item in team.items:
                inner_result.append(self.within_prior(item))
            result.append(inner_result)
        return result

    def iteration(self, _from: int = 0):
        for event_idx in range(_from, len(self.events)):
            teams = self.within_priors(event_idx)
            result = self.events[event_idx].result
            game = Game(teams, result, self.p_draw, self.weights)
            for team_idx, team in enumerate(self.events[event_idx].teams):
                for item_idx, item in enumerate(team.items):
                    self.skills[item.name].likelihood = (
                        self.skills[item.name].likelihood / item.likelihood
                    ) * game.likelihoods[team_idx][item_idx]
                    item.likelihood = game.likelihoods[team_idx][item_idx]
            self.events[event_idx].evidence = game.evidence

    def forward_prior_out(self, agent: Agent) -> Gaussian:
        return self.skills[agent].forward * self.skills[agent].likelihood


class History(object):
    def __init__(
        self,
        games: List[List[List[str]]],
        results: List[List[float]] = [],
        weights: List[float] = [],
        times: List[float] = [],
        mu: float = MU,
        sigma: float = SIGMA,
        beta: float = BETA,
        gamma: float = GAMMA,
        p_draw: float = P_DRAW,
    ):
        self.batches: List[Batch] = list()
        self.agents: Dict[str, Agent] = dict()
        self.mu = mu
        self.sigma = sigma
        self.beta = beta
        self.gamma = gamma
        self.p_draw = p_draw
        self.results = list()
        assert 0 < p_draw < 1
        self.use_specific_time = len(times) > 0

        self._init_games(games)
        self._init_results(games, results)
        self._init_times(games, times)
        self._init_weights(games, weights)
        self._init_agents(games)

    def _init_games(self, games: List[List[List[str]]]):
        # 一番内側のリスト(team)が全て同じ長さであることを確認
        team_len = len(games[0][0])
        for game in games:
            for team in game:
                if len(team) != team_len:
                    raise ValueError("len(team) != len(game[0])")
        self.games = games

    def _init_results(self, games: List[List[List[str]]], results: List[List[float]]):
        # resultのチェックと初期化
        if (len(results) > 0):
            assert len(games) == len(results)
        else:
            # 結果がない場合には、ゲーム数分の結果を降順で作成
            results = [[i for i in range(len(g))][::-1] for g in games]
        self.results = results
        
    def _init_times(self, games: List[List[List[str]]], times: List[float]):
        if len(times) > 0:
            assert len(games) == len(times)
        else:
            times = [i for i in range(len(games))]
        self.times = times
        
    def _init_weights(self, games: List[List[List[str]]], weights: List[float]):
        team_len = len(games[0][0])
        if (len(weights) > 0):
            assert len(weights) == team_len
        else:
            weights = [1.0 for _ in range(team_len)]
        self.weights = weights

    def _get_unique_agents_from_games(self, games: List[List[List[str]]]) -> Set[str]:
        result_set = set()
        for inner_list in games:
            for sub_list in inner_list:
                result_set.update(sub_list)
        return result_set
    
    def _init_agents(self, games: List[List[List[str]]]):
        unique_agents = self._get_unique_agents_from_games(games)
        for agent_name in unique_agents:
            _p = Player(Gaussian(self.mu, self.sigma), self.beta, self.gamma)
            self.agents[agent_name] = Agent(_p, Ninf, -inf,)

    def run(self):
        if self.use_specific_time:
            # 時間の指定がある場合には時間順
            order = sortperm(times)
        else:
            # そうでない場合には単純にリスト順
            order = [i for i in range(len(self.games))]

        idx_from = 0
        while idx_from < len(self.games):
            if self.use_specific_time:
                idx_to, time = idx_from + 1, times[order[idx_from]]
            else:
                idx_to, time = idx_from + 1, idx_from+ 1

            # このイテレーションでどの範囲まで見るか決定する
            while (len(times) > 0) and (idx_to < len(self.games)) and (times[order[idx_to]] == time):
                idx_to += 1

            # 上記で決定した範囲の結果を取ってくる
            _games = [self.games[k] for k in order[idx_from:idx_to]]
            _results = [self.results[k] for k in order[idx_from:idx_to]]

            # バッチを作成
            batch = Batch(
                _games,
                _results,
                time,
                self.agents,
                self.p_draw,
                self.weights
            )
            # agentsの更新
            self._update_agents(batch, time)
            # バッチを保存
            self.batches.append(batch)
            idx_from = idx_to

    def _update_agents(self, batch: Batch, time: float):
        for a in batch.skills:
            self.agents[a].last_time = time if self.use_specific_time else inf
            self.agents[a].message = batch.forward_prior_out(a)

    def get_results(self) -> Dict[str, List[Tuple[float, Gaussian]]]:
        res: Dict[str, List[Tuple[float, Gaussian]]] = dict()
        for b in self.batches:
            for a in b.skills:
                t_p = (b.time, b.posterior(a))
                if a in res:
                    res[a].append(t_p)
                else:
                    res[a] = [t_p]
        return res


if __name__ == "__main__":
    import pandas as pd
    from datetime import datetime

    df = pd.read_csv("history.csv", low_memory=False)

    columns = zip(df.w1_id, df.w2_id, df.l1_id, df.l2_id, df.double)
    _games = [
        [[w1, w2], [l1, l2]] if d == "t" else [[w1], [l1]]
        for w1, w2, l1, l2, d in columns
    ]
    _times = [
        datetime.strptime(t, "%Y-%m-%d").timestamp() / (60 * 60 * 24)
        for t in df.time_start
    ]
    games, times = [], []
    for g, t in zip(_games, _times):
        if len(g[0]) != 1:
            continue
        games.append(g)
        times.append(t)

    import time
    s = time.time()
    h = History(games, times=times, sigma=1.6, gamma=0.036, p_draw=0.001)
    h.run()
    e = time.time()
    print(e - s)

    # result = h.get_results()

    # for a, history in result.items():
    #     print(a, history[-1][1].mu)
