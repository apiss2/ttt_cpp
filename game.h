#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <map>

#include "params.h"
#include "gaussian.h"
#include "utils.h"
#include "team.h"

namespace TTT
{
    /*
    class GraphicalModel:
        def __init__(self, teams: List[List[Player]], _result: List[float], weights: List[float], p_draw: float) -> None:
            self.evidence: float = 1.0
            self.result: List[float] = self.init_result(_result, teams)
            self.order: List[int] = sortperm(self.result, reverse=True)
            self.team_variables: List[TeamVariable] = self.init_team_variables(self.order, teams, weights)
            self.diff_messages: List[DiffMessage] = self.init_diff_messages(teams)
            self.tie: List[bool] = self.init_tie()
            self.margins: List[float] = self.init_margin(p_draw)

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
                result = [i for i in range(len(teams) -1, -1, -1)]

        def init_team_variables(self, teams: List[List[Player]], weights) -> List[TeamVariable]:
            ret: List[TeamVariable] = list()
            for team_idx in range(len(teams)):
                idx = self.order[team_idx]
                _p = team_performance(teams[idx], weights)
                ret.append(TeamVariable(_p, Ninf, Ninf, Ninf))
            return ret

        def init_diff_messages(self, teams: List[List[Player]]) -> List[DiffMessage]:
            ret: List[DiffMessage] = list()
            for idx in range(len(teams) > 1):
                ret.append(DiffMessage(teams[idx].prior - teams[idx + 1].prior, Ninf))
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
    */
    class GraphicalModel
    {
    public:
        GraphicalModel(
            const std::vector<std::vector<Player>> &_teams,
            const std::vector<double> &_result,
            const std::vector<double> &_weights,
            const double p_draw)
        {
            evidence = 1.0;
            result = _result;
            order = sortperm(result);
            init_team_variables(order, _teams, _weights);
            init_diff_messages();
            init_tie();
            init_margin(p_draw, _teams);
        }
        void init_team_variables(const std::vector<int> &order, const std::vector<std::vector<Player>> &teams, const std::vector<double> &weights)
        {
            team_variables.resize(teams.size());
            for (int team_idx = 0; team_idx < teams.size(); team_idx++)
            {
                int idx = order[team_idx];
                Gaussian _p = team_performance(teams[idx], weights);
                team_variables[team_idx] = TeamVariable(_p, Ninf, Ninf, Ninf);
            }
        }
        void init_diff_messages()
        {
            diff_messages.resize(team_variables.size() - 1);
            for (int idx = 0; idx < team_variables.size() - 1; idx++)
            {
                diff_messages[idx] = DiffMessage(team_variables[idx].prior - team_variables[idx + 1].prior, Ninf);
            }
        }
        void init_tie()
        {
            tie.resize(diff_messages.size());
            for (int e = 0; e < diff_messages.size(); e++)
            {
                tie[e] = (result[order[e]] == result[order[e + 1]]);
            }
        }
        void init_margin(const double p_draw, const std::vector<std::vector<Player>> &teams)
        {
            margins.resize(diff_messages.size());
            for (int idx = 0; idx < diff_messages.size(); idx++)
            {
                if (p_draw == 0.0)
                {
                    margins[idx] = 0.0;
                }
                else
                {
                    double _sum1 = 0.0;
                    double _sum2 = 0.0;
                    for (int i = 0; i < teams[order[idx]].size(); i++)
                    {
                        _sum1 += std::pow(teams[order[idx]][i].beta, 2);
                    }
                    for (int i = 0; i < teams[order[idx + 1]].size(); i++)
                    {
                        _sum2 += std::pow(teams[order[idx + 1]][i].beta, 2);
                    }
                    double _m = compute_margin(p_draw, std::sqrt(_sum1 + _sum2));
                    margins[idx] = _m;
                }
            }
        }
        void partial_evidence(const int e)
        {
            double mu = diff_messages[e].prior.mu;
            double sigma = diff_messages[e].prior.sigma;
            double _mul;
            if (tie[e])
            {
                _mul = cdf(margins[e], mu, sigma) - cdf(-margins[e], mu, sigma);
            }
            else
            {
                _mul = 1 - cdf(margins[e], mu, sigma);
            }
            evidence = evidence * _mul;
        }
        void update_from_front(std::pair<double, double> &step, const int i)
        {
            for (int e = 0; e < diff_messages.size() - 1; e++)
            {
                // 更新_1
                diff_messages[e].prior = team_variables[e].posterior_win() - team_variables[e + 1].posterior_lose();
                // このクラスのevidence値を更新
                if (i == 0)
                {
                    partial_evidence(e);
                }
                // 更新_2
                diff_messages[e].likelihood = approx(diff_messages[e].prior, margins[e], tie[e]) / diff_messages[e].prior;
                // 更新_3 (stepも更新)
                Gaussian likelihood_lose = team_variables[e].posterior_win() - diff_messages[e].likelihood;
                step = max_tuple(step, team_variables[e + 1].likelihood_lose.delta(likelihood_lose));
                team_variables[e + 1].likelihood_lose = likelihood_lose;
            }
        }
        void update_from_back(std::pair<double, double> &step, const int i)
        {
            // 勝利側の値を更新
            for (int e = diff_messages.size() - 1; e > 0; e--)
            {
                // 更新_1
                diff_messages[e].prior = team_variables[e].posterior_win() - team_variables[e + 1].posterior_lose();
                // このクラスのevidence値を更新
                if ((i == 0) && (e == (diff_messages.size() - 1)))
                {
                    partial_evidence(e);
                };
                // 更新_2
                diff_messages[e].likelihood = approx(diff_messages[e].prior, margins[e], tie[e]) / diff_messages[e].prior;
                // 更新_3 (stepも更新)
                Gaussian likelihood_win = team_variables[e + 1].posterior_lose() + diff_messages[e].likelihood;
                step = max_tuple(step, team_variables[e].likelihood_win.delta(likelihood_win));
                team_variables[e].likelihood_win = likelihood_win;
            }
        }
        double evidence;
        std::vector<double> result;
        std::vector<int> order;
        std::vector<TeamVariable> team_variables;
        std::vector<DiffMessage> diff_messages;
        std::vector<bool> tie;
        std::vector<double> margins;
    }; // GraphicalModel;

    /*
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
            grm = GraphicalModel(self.teams, self.result, self.weights)
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
            for t in self.weights:
                for w in t:
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
    */
    class Game
    {
    public:
        Game(
            const std::vector<std::vector<Player>> &_teams,
            const std::vector<double> &_result,
            const double _p_draw,
            const std::vector<double> &_weights)
        {
            check_inputs(_result, _teams, _p_draw, _weights);
            teams = _teams;
            result = _result;
            p_draw = _p_draw;
            weights = _weights;
            evidence = 0.0;
            compute_likelihoods();
        }
        void check_inputs(const std::vector<double> &_result, const std::vector<std::vector<Player>> &_teams, const double p_draw, const std::vector<double> &_weights)
        {
            if (_teams.size() != _result.size())
            {
                throw std::invalid_argument("len(teams) != len(result)");
            }
            if ((p_draw <= 0) || (p_draw >= 1))
            {
                throw std::invalid_argument("0 < p_draw < 1");
            }
            for (auto _team : _teams)
            {
                if (_team.size() != _weights.size())
                {
                    throw std::invalid_argument("len(team) != len(weights)");
                }
            }
        }
        Gaussian performance(const int i)
        {
            return team_performance(teams[i], weights);
        }
        std::vector<std::vector<Gaussian>> likelihood_analitico()
        {
            GraphicalModel grm(teams, result, weights, p_draw);
            grm.partial_evidence(0);
            Gaussian diffmsg = grm.diff_messages[0].prior;
            double margin = grm.margins[0];
            bool tie = grm.tie[0];

            double mu_trunc, sigma_trunc;
            std::tie(mu_trunc, sigma_trunc) = trunc(diffmsg.mu, diffmsg.sigma, margin, tie);
            double sigma_trunc_2 = std::pow(sigma_trunc, 2);
            double diff_sigma_2 = std::pow(diffmsg.sigma, 2);
            double delta_div = diff_sigma_2 * mu_trunc - sigma_trunc_2 * diffmsg.mu;
            double theta_div_pow2;
            if (diffmsg.sigma == sigma_trunc)
            {
                theta_div_pow2 = inf;
            }
            else
            {
                double _div = diff_sigma_2 - sigma_trunc_2;
                delta_div = delta_div / _div;
                theta_div_pow2 = (sigma_trunc_2 * diff_sigma_2) / _div;
            };

            // チームごとに計算
            std::vector<std::vector<Gaussian>> res;
            for (int team_idx = 0; team_idx < grm.team_variables.size(); team_idx++)
            {
                std::vector<Player> &players = teams[grm.order[team_idx]];
                std::vector<Gaussian> team(players.size());
                for (int j = 0; j < players.size(); j++)
                {
                    double mu;
                    if (diffmsg.sigma == sigma_trunc)
                    {
                        mu = 0.0;
                    }
                    else
                    {
                        double _mu = players[j].prior.mu;
                        double _d = delta_div - diffmsg.mu;
                        if (team_idx == 1)
                        {
                            mu = _mu - _d;
                        }
                        else
                        {
                            mu = _mu + _d;
                        }
                    }
                    double _team_sigma = std::pow(players[j].prior.sigma, 2);
                    double sigma_analitico = std::sqrt(theta_div_pow2 + diff_sigma_2 - _team_sigma);
                    team[j] = Gaussian(mu, sigma_analitico);
                }
                res.push_back(team);
            }
            // evidence更新
            evidence = grm.evidence;
            // 大小関係確認して返却
            if (grm.order[0] < grm.order[1])
            {
                return {res[0], res[1]};
            }
            else
            {
                return {res[1], res[0]};
            }
        }
        std::vector<Gaussian> likelihood_teams()
        {
            GraphicalModel grm(teams, result, weights, p_draw);
            std::pair<double, double> step = {inf, inf};
            int i = 0;
            while (gr_tuple(step, 1e-6) && (i < 10))
            {
                step = {0.0, 0.0};
                grm.update_from_front(step, i);
                grm.update_from_back(step, i);
                i += 1;
            }

            if (grm.diff_messages.size() == 1)
            {
                // evidence値を更新
                grm.partial_evidence(0);
                // diff_messages更新
                grm.diff_messages[0].prior = grm.team_variables[0].posterior_win() - grm.team_variables[1].posterior_lose();
                grm.diff_messages[0].likelihood = approx(grm.diff_messages[0].prior, grm.margins[0], grm.tie[0]) / grm.diff_messages[0].prior;
            }

            // Gameクラスのevidenceを更新
            evidence = grm.evidence;
            grm.team_variables[0].likelihood_win = grm.team_variables[1].posterior_lose() + grm.diff_messages[0].likelihood;
            grm.team_variables[grm.team_variables.size() - 1].likelihood_lose = grm.team_variables[grm.team_variables.size() - 2].posterior_win() - grm.diff_messages[grm.diff_messages.size() - 1].likelihood;
            std::vector<Gaussian> ret(grm.team_variables.size());
            for (int e = 0; e < grm.team_variables.size(); e++)
            {
                ret[e] = grm.team_variables[grm.order[e]].likelihood();
            }
            return ret;
        }
        bool hasNotOneWeights() const
        {
            for (int i = 0; i < weights.size(); i++)
            {
                if (weights[i] != 1.0)
                {
                    return true;
                }
            }
            return false;
        }
        void compute_likelihoods()
        {
            if ((teams.size() > 2) || hasNotOneWeights())
            {
                std::vector<Gaussian> m_t_ft = likelihood_teams();
                likelihoods.clear();
                for (int e = 0; e < teams.size(); e++)
                {
                    std::vector<Gaussian> likelihood_e(teams[e].size());
                    for (int i = 0; i < teams[e].size(); i++)
                    {
                        double _lh;
                        if (weights[i] != 0.0)
                        {
                            _lh = 1 / weights[i];
                        }
                        else
                        {
                            _lh = inf;
                        }
                        Gaussian lh = (m_t_ft[e] - performance(e).exclude(teams[e][i].prior * weights[i])) * _lh;
                        likelihood_e[i] = lh;
                    }
                    likelihoods.push_back(likelihood_e);
                }
            }
            else
            {
                likelihoods = likelihood_analitico();
            }
        }
        std::vector<std::vector<Player>> teams;
        std::vector<double> result;
        double p_draw;
        std::vector<double> weights;
        std::vector<std::vector<Gaussian>> likelihoods;
        double evidence;
    }; // Game
}; // namespace TTT
