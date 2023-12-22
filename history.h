#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <map>
#include <set>
#include <numeric>
#include <array>
#include <string>
#include <fstream>
#include <sstream>
#include <ctime>

#include "params.h"
#include "gaussian.h"
#include "utils.h"
#include "team.h"
#include "game.h"
#include "batch.h"

namespace TTT
{
    /*

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
            assert 0 < p_draw < 1
            self.use_specific_time = len(times) > 0

            self.check_games(games)
            self.init_results(games, results)
            self.init_times(games, times)
            self.init_weights(games, weights)
            self.init_agents(games)

        def check_games(games: List[List[List[str]]]):
            # 一番内側のリスト(team)が全て同じ長さであることを確認
            team_len = len(games[0][0])
            for game in games:
                for team in game:
                    if len(team) != team_len:
                        raise ValueError("len(team) != len(game[0])")

        def init_results(games: List[List[List[str]]], results: List[List[float]]):
            # resultのチェックと初期化
            if (len(results) > 0):
                assert len(games) == len(results)
            else:
                # 結果がない場合には、ゲーム数分の結果を降順で作成
                results = [[i for i in range(len(g))][::-1] for g in games]
            self.results = results
            
        def init_times(games: List[List[List[str]]], times: List[float]):
            if len(times) > 0:
                assert len(games) == len(times)
            else:
                times = [i for i in range(len(games))]
            self.times = times
            
        def init_weights(games: List[List[List[str]]], weights: List[float]):
            team_len = len(games[0][0])
            if (len(weights) > 0):
                assert len(weights) == team_len
            else:
                weights = [1.0 for _ in range(team_len)]
            self.weights = weights

        def get_unique_agents_from_games(self, games: List[List[List[str]]]) -> Set[str]:
            result_set = set()
            for inner_list in games:
                for sub_list in inner_list:
                    result_set.update(sub_list)
            return result_set
        
        def init_agents(self, games: List[List[List[str]]]):
            unique_agents = self.get_unique_agents_from_games(games)
            for agent_name in unique_agents:
                _p = Player(Gaussian(self.mu, self.sigma), self.beta, self.gamma)
                self.agents[agent_name] = Agent(_p, Ninf, -inf,)

        def run(self):
            if self.use_specific_time:
                # 時間の指定がある場合には時間順
                order = sortperm(times)
            else:
                # そうでない場合には単純にリスト順
                order = [i for i in range(len(games))]

            idx_from = 0
            while idx_from < len(games):
                if self.use_specific_time:
                    idx_to, time = idx_from + 1, times[order[idx_from]]
                else:
                    idx_to, time = idx_from + 1, idx_from+ 1

                # このイテレーションでどの範囲まで見るか決定する
                while (len(times) > 0) and (idx_to < len(games)) and (times[order[idx_to]] == time):
                    idx_to += 1

                # 上記で決定した範囲の結果を取ってくる
                _games = [games[k] for k in order[idx_from:idx_to]]
                _results = [results[k] for k in order[idx_from:idx_to]]

                # バッチを作成
                batch = Batch(
                    _games,
                    _results,
                    time,
                    self.agents,
                    self.p_draw,
                    weights
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
        games = [
            [[w1, w2], [l1, l2]] if d == "t" else [[w1], [l1]]
            for w1, w2, l1, l2, d in columns
        ]
        times = [
            datetime.strptime(t, "%Y-%m-%d").timestamp() / (60 * 60 * 24)
            for t in df.time_start
        ]

        h = History(composition=games, times=times, sigma=1.6, gamma=0.036)
        h.run(games, [], times, [])
    */
    class History
    {
    public:
        History(
            const std::vector<std::vector<std::vector<std::string>>> &_games,
            const std::vector<std::vector<double>> &_results = {},
            const std::vector<double> &_weights = {},
            const std::vector<double> &_times = {},
            double _mu = MU,
            double _sigma = SIGMA,
            double _beta = BETA,
            double _gamma = GAMMA,
            double _p_draw = P_DRAW)
        {
            this->mu = _mu;
            this->sigma = _sigma;
            this->beta = _beta;
            this->gamma = _gamma;
            this->p_draw = _p_draw;
            if (p_draw <= 0 || p_draw >= 1)
            {
                throw std::invalid_argument("p_draw should be in (0, 1)");
            }
            this->use_specific_time = _times.size() > 0;

            this->init_games(_games);
            this->init_results(_games, _results);
            this->init_times(_games, _times);
            this->init_weights(_games, _weights);
            this->init_agents(_games);
        }
        void init_games(const std::vector<std::vector<std::vector<std::string>>> &_games)
        {
            if (_games.size() == 0)
            {
                throw std::invalid_argument("games should not be empty");
            }
            // 一番内側のリスト(team)が全て同じ長さであることを確認
            int team_len = _games[0][0].size();
            for (auto game : _games)
            {
                for (auto team : game)
                {
                    if (team.size() != team_len)
                    {
                        throw std::invalid_argument("len(team) != len(game[0])");
                    }
                }
            }
            this->games = _games;
        }
        void init_results(const std::vector<std::vector<std::vector<std::string>>> &games, const std::vector<std::vector<double>> &in_results)
        {
            // resultのチェックと初期化
            if (in_results.size() > 0)
            {
                if (games.size() != in_results.size())
                {
                    throw std::invalid_argument("len(games) != len(results)");
                }
                this->results = in_results;
            }
            else
            {
                // 結果がない場合には、ゲーム数分の結果を降順で作成
                std::vector<std::vector<double>> _results;
                for (int i = 0; i < games.size(); i++)
                {
                    std::vector<double> inner_results;
                    for (int j = games[i].size()-1; j >= 0; j--)
                    {
                        inner_results.push_back((double)j);
                    }
                    _results.push_back(inner_results);
                }
                this->results = _results;
            }
        }
        void init_times(const std::vector<std::vector<std::vector<std::string>>> &games, const std::vector<double> &_times)
        {
            if (_times.size() > 0)
            {
                if (games.size() != _times.size())
                {
                    throw std::invalid_argument("len(games) != len(times)");
                }
                this->times = _times;
            }
            else
            {
                std::vector<double> _t;
                for (int i = 0; i < games.size(); i++)
                {
                    _t.push_back((double)i);
                }
                this->times = _t;
            }
        }
        void init_weights(const std::vector<std::vector<std::vector<std::string>>> &games, const std::vector<double> &_weights)
        {
            int team_len = games[0][0].size();
            if (_weights.size() > 0)
            {
                if (_weights.size() != team_len)
                {
                    throw std::invalid_argument("len(weights) != len(games[0][0])");
                }
                this->weights = _weights;
            }
            else
            {
                std::vector<double> _w;
                for (int i = 0; i < team_len; i++)
                {
                    _w.push_back(1.0);
                }
                this->weights = _w;
            }
        }
        std::set<std::string> get_unique_agents_from_games(const std::vector<std::vector<std::vector<std::string>>> &games)
        {
            std::set<std::string> result_set;
            for (auto inner_list : games)
            {
                for (auto sub_list : inner_list)
                {
                    result_set.insert(sub_list.begin(), sub_list.end());
                }
            }
            return result_set;
        }
        void init_agents(const std::vector<std::vector<std::vector<std::string>>> &games)
        {
            std::set<std::string> unique_agents = this->get_unique_agents_from_games(games);
            for (auto agent_name : unique_agents)
            {
                Player _p(Gaussian(this->mu, this->sigma), this->beta, this->gamma);
                this->agents[agent_name] = Agent(_p, Ninf, -inf);
            }
        }
        void run()
        {
            std::vector<int> order;
            if (this->use_specific_time)
            {
                // 時間の指定がある場合には時間順
                order = sortperm(this->times);
            }
            else
            {
                // そうでない場合には単純にリスト順
                order = std::vector<int>(this->times.size());
                std::iota(order.begin(), order.end(), 0);
            }

            int idx_from = 0;
            while (idx_from < this->times.size())
            {
                int idx_to;
                double time;
                if (this->use_specific_time)
                {
                    idx_to = idx_from + 1;
                    time = times[order[idx_from]];
                }
                else
                {
                    idx_to = idx_from + 1;
                    time = idx_from + 1;
                }

                // このイテレーションでどの範囲まで見るか決定する
                while ((idx_to < times.size()) && (times[order[idx_to]] == time))
                {
                    idx_to++;
                }

                // 上記で決定した範囲の結果を取ってくる
                std::vector<std::vector<std::vector<std::string>>> _games;
                std::vector<std::vector<double>> _results;
                for (int i = idx_from; i < idx_to; i++)
                {
                    _games.push_back(this->games[order[i]]);
                    _results.push_back(this->results[order[i]]);
                }

                // バッチを作成
                Batch batch(_games, _results, time, this->agents, this->p_draw, this->weights);
                // agentsの更新
                this->update_agents(batch, time);
                // バッチを保存
                this->batches.push_back(batch);
                idx_from = idx_to;
            }
        }
        void update_agents(const Batch &batch, double time)
        {
            for (auto& [agent, s] : batch.skills)
            {
                this->agents[agent].last_time = this->use_specific_time ? time : inf;
                this->agents[agent].message = batch.forward_prior_out(agent);
            }
        }
        std::map<std::string, std::vector<std::pair<double, Gaussian>>> get_results()
        {
            std::map<std::string, std::vector<std::pair<double, Gaussian>>> res;
            for (auto b : this->batches)
            {
                for (auto [agent, s] : b.skills)
                {
                    std::pair<double, Gaussian> t_p(b.time, b.posterior(agent));
                    if (res.find(agent) != res.end())
                    {
                        res[agent].push_back(t_p);
                    }
                    else
                    {
                        res[agent] = {t_p};
                    }
                }
            }
            return res;
        }
        double mu;
        double sigma;
        double beta;
        double gamma;
        double p_draw;
        bool use_specific_time;
        std::vector<std::vector<std::vector<std::string>>> games;
        std::vector<Batch> batches;
        std::map<std::string, Agent> agents;
        std::vector<std::vector<double>> results;
        std::vector<double> times;
        std::vector<double> weights;
    };
}