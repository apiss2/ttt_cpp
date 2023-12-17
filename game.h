#pragma once

#include <vector>
#include <set>
#include "utils.h"
#include "gaussian.h"
#include "team.h"

namespace TTT
{
    struct GraphicalModel
    {
        std::vector<int> order;
        std::vector<TeamVariable> team_variables;
        std::vector<DiffMessage> diff_messages;
        std::vector<bool> tie_list;
        std::vector<double> margins;
    };

    class Game
    {
    private:
        std::vector<std::vector<Player>> teams;
        std::vector<double> result;
        double p_draw;
        race_weight weights;
        std::vector<std::vector<Gaussian>> likelihoods;
        double evidence;

    public:
        Game(std::vector<std::vector<Player>> _teams, std::vector<double> _result = {}, double _p_draw = 0.0, std::vector<std::vector<double>> _weights = {})
        {
            this->p_draw = _p_draw;
            this->teams = _teams;
            this->result = _result;
            check_input();
            if (_weights.size() == 0)
            {
                for (int i = 0; i < weights.size(); i++)
                {
                    this->weights.push_back(std::vector<double>(teams[i].size(), 1.0));
                }
            }
            else
            {
                this->weights = _weights;
            }
            this->likelihoods = std::vector<std::vector<Gaussian>>();
            this->evidence = 0.0;
            compute_likelihoods();
        }

        Gaussian get_likelihoods(int i, int j)
        {
            return this->likelihoods[i][j];
        }

        double get_evidence()
        {
            return this->evidence;
        }

        void check_input() const
        {
            if (this->result.size() && (this->teams.size() != this->result.size()))
            {
                throw std::invalid_argument("len(result) and (len(teams) != len(result))");
            }
            if ((0.0 > this->p_draw) || (1.0 <= this->p_draw))
            {
                throw std::invalid_argument("0.0 <= proba < 1.0");
            }
            if ((this->p_draw == 0.0) && (this->result.size() > 0) && (std::set<std::vector<double>>(this->result.begin(), this->result.end()).size() != this->result.size()))
            {
                throw std::invalid_argument("(p_draw == 0.0) and (len(result)>0) and (len(set(result))!=len(result))");
            }
            if ((this->weights.size() > 0) && (this->teams.size() != this->weights.size()))
            {
                throw std::invalid_argument("(len(weights)>0) & (len(teams)!= len(weights))");
            }
            for (int i = 0; i < this->weights.size(); i++)
            {
                if (this->teams[i].size() != this->weights[i].size())
                {
                    throw std::invalid_argument("(len(weights)>0) & exists i (len(teams[i]) != len(weights[i])");
                }
            }
        }

        int teams_size() const
        {
            return this->teams.size();
        }

        std::vector<int> team_size_list() const
        {
            std::vector<int> s;
            for (int i = 0; this->teams.size(); i++)
            {
                s.push_back(this->teams[i].size());
            }
            return s;
        }

        Gaussian performance(int i)
        {
            return team_performance(teams[i], weights[i]);
        }

        void partial_evidence(const std::vector<DiffMessage> &d, std::vector<double> &margin, const std::vector<bool> &tie, int e)
        {
            double mu = d[e].getPrior().getMu();
            double sigma = d[e].getPrior().getSigma();
            double _mul;
            if (tie[e])
            {
                _mul = cdf(margin[e], mu, sigma) - cdf(-margin[e], mu, sigma);
            }
            else
            {
                _mul = 1.0 - cdf(margin[e], mu, sigma);
            }
            this->evidence = this->evidence * _mul;
        }

        GraphicalModel graphical_model()
        {
            auto &this_game = *this;
            GraphicalModel ret;
            std::vector<double> r;

            if (this_game.result.size() > 0)
            {
                r = this_game.result;
            }
            else
            {
                for (int i = this_game.teams.size() - 1; i > -1; i--)
                {
                    r.push_back(i);
                }
            }

            ret.order = sortperm(r, true);

            for (int e = 0; e < this_game.teams_size(); e++)
            {
                ret.team_variables.push_back(TeamVariable(this_game.performance(ret.order[e]), Ninf, Ninf, Ninf));
            }

            for (int e = 0; e < this_game.teams_size() - 1; e++)
            {
                ret.diff_messages.push_back(DiffMessage(ret.team_variables[e].getPrior() - ret.team_variables[e + 1].getPrior(), Ninf));
            }

            for (int e = 0; e < ret.diff_messages.size(); e++)
            {
                ret.tie_list.push_back(r[ret.order[e]] == r[ret.order[e + 1]]);
            }

            for (int e = 0; e < ret.diff_messages.size(); e++)
            {
                if (this_game.p_draw == 0.0)
                {
                    ret.margins.push_back(0.0);
                }
                else
                {
                    double sum_beta_squared_1 = 0.0;
                    double sum_beta_squared_2 = 0.0;

                    for (const auto &a : this_game.teams[ret.order[e]])
                    {
                        sum_beta_squared_1 += std::pow(a.getBeta(), 2);
                    }

                    for (const auto &a : this_game.teams[ret.order[e + 1]])
                    {
                        sum_beta_squared_2 += std::pow(a.getBeta(), 2);
                    }

                    ret.margins.push_back(compute_margin(this_game.p_draw, std::sqrt(sum_beta_squared_1 + sum_beta_squared_2)));
                }
            }

            this_game.evidence = 1.0;

            return ret;
        }

        std::vector<std::vector<Gaussian>> likelihood_analitico()
        {
            auto &this_game = *this;
            GraphicalModel g_result = this_game.graphical_model();
            this_game.partial_evidence(g_result.diff_messages, g_result.margins, g_result.tie_list, 0);
            Gaussian d_gaussian = g_result.diff_messages[0].getPrior();
            auto [mu_trunc, sigma_trunc] = trunc(d_gaussian.getMu(), d_gaussian.getSigma(), g_result.margins[0], g_result.tie_list[0]);

            double delta_div, theta_div_pow2;
            if (d_gaussian.getSigma() == sigma_trunc)
            {
                delta_div = d_gaussian.getSigma() * d_gaussian.getSigma() * mu_trunc - sigma_trunc * sigma_trunc * d_gaussian.getMu();
                theta_div_pow2 = TTT::inf;
            }
            else
            {
                delta_div = (d_gaussian.getSigma() * d_gaussian.getSigma() * mu_trunc - sigma_trunc * sigma_trunc * d_gaussian.getMu()) / (d_gaussian.getSigma() * d_gaussian.getSigma() - sigma_trunc * sigma_trunc);
                theta_div_pow2 = (sigma_trunc * sigma_trunc * d_gaussian.getSigma() * d_gaussian.getSigma()) / (d_gaussian.getSigma() * d_gaussian.getSigma() - sigma_trunc * sigma_trunc);
            }

            std::vector<std::vector<Gaussian>> res;
            for (size_t i = 0; i < g_result.team_variables.size(); i++)
            {
                std::vector<Gaussian> team;
                for (size_t j = 0; j < this->teams[g_result.order[i]].size(); j++)
                {
                    double mu_analitico;
                    if (d_gaussian.getSigma() == sigma_trunc)
                    {
                        mu_analitico = 0.0;
                    }
                    else
                    {
                        double raw_mu = this->teams[g_result.order[i]][j].getPrior().getMu();
                        double add_mu = delta_div - d_gaussian.getMu();
                        if (i == 1)
                        {
                            mu_analitico = raw_mu - add_mu;
                        }
                        else
                        {
                            mu_analitico = raw_mu + add_mu;
                        }
                    }
                    double sigma_analitico = sqrt(theta_div_pow2 + d_gaussian.getSigma() * d_gaussian.getSigma() - this_game.teams[g_result.order[i]][j].getPrior().getSigma() * this_game.teams[g_result.order[i]][j].getPrior().getSigma());
                    team.push_back(Gaussian(mu_analitico, sigma_analitico));
                }
                res.push_back(team);
            }
            std::vector<std::vector<Gaussian>> ret;
            if (g_result.order[0] < g_result.order[1])
            {
                ret.push_back(res[0]);
                ret.push_back(res[1]);
            }
            else
            {
                ret.push_back(res[1]);
                ret.push_back(res[0]);
            }
            return ret;
        }

        std::vector<Gaussian> likelihood_teams()
        {
            Game g = *this;
            auto [o, t, d, tie, margin] = g.graphical_model();

            std::pair<double, double> step(TTT::inf, TTT::inf);
            int i = 0;
            while (gr_tuple(step, 1e-6) && (i < 10))
            {
                step = std::make_pair(0.0, 0.0);
                for (int e = 0; e < d.size() - 1; e++)
                {
                    d[e].getPrior() = t[e].posterior_win() - t[e + 1].posterior_lose();
                    if (i == 0)
                    {
                        g.partial_evidence(d, margin, tie, e);
                    }
                    Gaussian p = d[e].getPrior();
                    d[e].setLikelihood(approx(p, margin[e], tie[e]) / p);
                    Gaussian likelihood_lose = t[e].posterior_win() - d[e].getLikelihood();
                    std::pair<double, double> t2 = t[e + 1].getLikelihoodLose().delta(likelihood_lose);
                    step = max_tuple(step, t2);
                    t[e + 1].setLikelihoodLose(likelihood_lose);
                }
                for (int e = d.size() - 1; e > 0; e--)
                {
                    d[e].setPrior(t[e].posterior_win() - t[e + 1].posterior_lose());
                    if ((i == 0) && (e == d.size() - 1))
                    {
                        g.partial_evidence(d, margin, tie, e);
                    }
                    Gaussian p = d[e].getPrior();
                    d[e].setLikelihood(approx(p, margin[e], tie[e]) / p);
                    Gaussian likelihood_win = t[e + 1].posterior_lose() + d[e].getLikelihood();
                    std::pair<double, double> t2 = t[e].getLikelihoodWin().delta(likelihood_win);
                    step = max_tuple(step, t2);
                    t[e].setLikelihoodWin(likelihood_win);
                }
                i++;
            }

            if (d.size() == 1)
            {
                g.partial_evidence(d, margin, tie, 0);
                d[0].setPrior(t[0].posterior_win() - t[1].posterior_lose());
                Gaussian p = d[0].getPrior();
                d[0].setLikelihood(approx(p, margin[0], tie[0]) / p);
            }

            t[0].setLikelihoodWin(t[1].posterior_lose() + d[0].getLikelihood());
            t.back().setLikelihoodLose(t[t.size() - 2].posterior_win() - d.back().getLikelihood());

            std::vector<Gaussian> likelihoods;
            for (auto &team : t)
            {
                likelihoods.push_back(team.likelihood());
            }
            return likelihoods;
        }

        bool hasNotOneWeights()
        {
            for (auto t : this->weights)
            {
                for (auto w : t)
                {
                    if (w != 1.0)
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        void compute_likelihoods()
        {
            Game &thisgame = *this;
            if ((teams.size() > 2) || hasNotOneWeights())
            {
                std::vector<Gaussian> m_t_ft = thisgame.likelihood_teams();
                thisgame.likelihoods.clear();
                for (int e = 0; e < thisgame.teams_size(); e++)
                {
                    std::vector<Gaussian> lh_list;
                    for (int i = 0; i < teams[e].size(); i++)
                    {
                        double _lh_w;
                        Gaussian _lh;
                        if (weights[e][i] != 0.0)
                        {
                            _lh_w = (1 / weights[e][i]);
                        }
                        else
                        {
                            _lh_w = TTT::inf;
                        }
                        _lh = (m_t_ft[e] - thisgame.performance(e).exclude(thisgame.teams[e][i].getPrior() * thisgame.weights[e][i])) * _lh_w;
                        lh_list.push_back(_lh);
                    }
                    thisgame.likelihoods.push_back(lh_list);
                }
            }
            else
            {
                thisgame.likelihoods = thisgame.likelihood_analitico();
            }
        }

        std::vector<std::vector<Gaussian>> posteriors()
        {
            Game &thisgame = *this;
            std::vector<std::vector<Gaussian>> res;
            for (int e = 0; e < thisgame.teams_size(); e++)
            {
                std::vector<Gaussian> _res;
                for (int i = 0; i < thisgame.teams[e].size(); i++)
                {
                    _res.push_back(thisgame.likelihoods[e][i] * thisgame.teams[e][i].getPrior());
                }
                res.push_back(_res);
            }
            return res;
        }
    };

    class Event
    {
    public:
        std::vector<Team> teams;
        double evidence;
        race_weight weights;

        Event(
            std::vector<Team> t,
            double ev,
            std::vector<std::vector<double>> w) : teams(t), evidence(ev), weights(w) {}

        std::vector<std::vector<std::string>> names()
        {
            std::vector<std::vector<std::string>> res;
            for (const Team &team : teams)
            {
                std::vector<std::string> team_names;
                for (const Item &item : team.items)
                {
                    team_names.push_back(item.name);
                }
                res.push_back(team_names);
            }
            return res;
        }

        std::vector<double> result()
        {
            std::vector<double> res;
            for (const Team &team : teams)
            {
                res.push_back(team.output);
            }
            return res;
        }
    };
}