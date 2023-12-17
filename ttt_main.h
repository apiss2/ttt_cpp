#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <ctime>

#include "utils.h"
#include "ttt_params.h"
#include "gaussian.h"
#include "team.h"
#include "game.h"

namespace TTT
{
    class Batch
    {
    private:
        std::map<std::string, Agent> agents;
        std::map<std::string, Skill> skills;
        std::vector<Event> events;
        double time;
        double p_draw;

    public:
        Batch(
            std::vector<race_keys> _composition = {},
            std::vector<std::vector<double>> _results = {},
            double _time = 0,
            std::map<std::string, Agent> _agents = {},
            double _p_draw = 0,
            std::vector<race_weight> _weights = {})
        {
            this->input_ok(_composition, _results, _weights);
            std::set<std::string> agents_set;
            for (const race_keys &teams : _composition)
            {
                for (const team_keys &team : teams)
                {
                    for (const std::string &agent : team)
                    {
                        agents_set.insert(agent);
                    }
                }
            }

            std::map<std::string, double> elapsed;
            for (const std::string &agent : agents_set)
            {
                elapsed[agent] = compute_elapsed(agents[agent].getLasttime(), time);
                skills[agent] = Skill(agents[agent].receive(elapsed[agent]), Ninf, Ninf, elapsed[agent]);
            }

            for (size_t i = 0; i < _composition.size(); i++)
            {
                std::vector<std::vector<std::string>> team_composition = _composition[i];
                std::vector<std::vector<Item>> items_list;
                for (const std::vector<std::string> &team : team_composition)
                {
                    std::vector<Item> items;
                    for (const std::string &item : team)
                    {
                        items.push_back(Item(item, Ninf));
                    }
                    items_list.push_back(items);
                }
                std::vector<Team> team_list;
                for (size_t j = 0; j < team_composition.size(); j++)
                {
                    if (_results.size() > 0){
                        team_list.push_back(Team(items_list[j], _results[i][j]));
                    }
                    else
                    {
                        team_list.push_back(Team(items_list[j], team_composition.size() - j - 1));
                    }
                }

                std::vector<std::vector<double>> w;
                if (_weights.size() > 0)
                {
                    w = _weights[i];
                }
                else
                {
                    w = {};
                }
                events.push_back(Event(team_list, 0.0, w));
            }

            this->time = time;
            this->agents = _agents;
            this->p_draw = p_draw;
            iteration();
        }
        
        bool input_ok(
            std::vector<race_keys> &composition,
            std::vector<std::vector<double>> &results,
            std::vector<race_weight> &weights){
            if (results.size() > 0 && composition.size() != results.size())
            {
                throw std::invalid_argument("(len(results)>0) and (len(composition)!= len(results))");
            }
            if (weights.size() > 0 && composition.size() != weights.size())
            {
                throw std::invalid_argument("(len(weights)>0) & (len(composition)!= len(weights))");
            }
            return true;
        }

        int size()
        {
            return events.size();
        }

        Gaussian posterior(const std::string &agent)
        {
            return skills[agent].likelihood * skills[agent].backward * skills[agent].forward;
        }

        std::map<std::string, Gaussian> posteriors()
        {
            std::map<std::string, Gaussian> res;
            for (auto &pair : skills)
            {
                res[pair.first] = posterior(pair.first);
            }
            return res;
        }

        Player _within_prior(Item item)
        {
            Player r = this->agents[item.name].getPlayer();
            Gaussian posterior_item = this->posterior(item.name);
            Gaussian _p = posterior_item / item.likelihood;
            return Player(_p, r.getBeta(), r.getGamma(), Ninf);
        }

        std::vector<std::vector<Player>> within_priors(size_t event)
        {
            std::vector<std::vector<Player>> res;
            for (const Team &team : events[event].teams)
            {
                std::vector<Player> team_players;
                for (const Item &item : team.items)
                {
                    team_players.push_back(this->_within_prior(item));
                }
                res.push_back(team_players);
            }
            return res;
        }

        void iteration(size_t from = 0)
        {
            for (size_t e = from; e < size(); e++)
            {
                std::vector<std::vector<Player>> teams = within_priors(e);
                std::vector<double> results = events[e].result();
                race_weight weights = events[e].weights;

                Game game(teams, results, p_draw, weights);
                for (size_t t = 0; t < events[e].teams.size(); t++)
                {
                    Team &team = events[e].teams[t];
                    for (size_t i = 0; i < team.items.size(); i++)
                    {
                        Item &item = team.items[i];
                        Gaussian likelihood = game.get_likelihoods(t, i);
                        skills[item.name].likelihood = (skills[item.name].likelihood / item.likelihood) * likelihood;
                        item.likelihood = game.get_likelihoods(t, i);
                    }
                }
                events[e].evidence = game.get_evidence();
            }
        }

        int convergence(double epsilon = 1e-6, int iterations = 20)
        {
            std::pair<double, double> step(TTT::inf, TTT::inf);
            int i = 0;
            while (gr_tuple(step, epsilon) && i < iterations)
            {
                std::map<std::string, Gaussian> old_posterior = posteriors();
                iteration();
                std::map<std::string, Gaussian> new_posterior = posteriors();
                step = dict_diff(old_posterior, new_posterior);
                i++;
            }
            return i;
        }

        Gaussian forward_prior_out(std::string agent_key)
        {
            return skills[agent_key].forward * skills[agent_key].likelihood;
        }

        Gaussian backward_prior_out(std::string agent_key)
        {
            Gaussian N = skills[agent_key].likelihood * skills[agent_key].backward;
            return N.forget(agents[agent_key].getPlayer().getGamma(), skills[agent_key].elapsed);
        }

        void new_backward_info()
        {
            for (auto &pair : skills)
            {
                std::string agent = pair.first;
                skills[agent].backward = agents[agent].getMessage();
            }
            iteration();
        }

        void new_forward_info()
        {
            for (auto &pair : skills)
            {
                std::string agent = pair.first;
                skills[agent].forward = agents[agent].receive(skills[agent].elapsed);
            }
            iteration();
        }

        std::map<std::string, Skill> getSkills() const {
            return skills;
        }

        double getTime() const {
            return time;
        }
    };

    class History
    {
    private:
        std::vector<Batch> batches;
        std::map<std::string, Agent> agents;
        std::vector<double> times;
        int size;
        double mu;
        double sigma;
        double gamma;
        double p_draw;
        bool time;

    public:
        History(
            std::vector<std::vector<std::vector<std::string>>> composition,
            std::vector<std::vector<double>> results,
            std::vector<double> times = {},
            std::map<std::string, Player> priors = {},
            double mu = TTT::MU,
            double sigma = TTT::SIGMA,
            double beta = TTT::BETA,
            double gamma = TTT::GAMMA,
            double p_draw = TTT::P_DRAW,
            std::vector<race_weight> weights = {})
        {
            if (composition.size() != results.size())
            {
                throw std::invalid_argument("composition.size() != results.size()");
            }
            if (times.size() > 0 && composition.size() != times.size())
            {
                throw std::invalid_argument("composition.size() != times.size()");
            }
            if (weights.size() > 0 && composition.size() != weights.size())
            {
                throw std::invalid_argument("composition.size() != weights.size()");
            }

            size = composition.size();
            batches = {};

            for (const auto& teams : composition) {
                for (const auto& team : teams) {
                    for (const auto& a : team) {
                        // Batchに存在しない場合には追加
                        if (agents.count(a) == 0) {
                            if (priors.count(a) != 0) {
                                agents[a] = Agent(priors[a], Ninf, -inf);
                            } else {
                                // 今までで登場しない場合には既定値で初期化
                                agents[a] = Agent(Player(Gaussian(mu, sigma), beta, gamma), Ninf, -inf);
                            }
                        }
                    }
                }
            }
            this->mu = mu;
            this->sigma = sigma;
            this->gamma = gamma;
            this->p_draw = p_draw;
            time = times.size() > 0;
            trueskill(composition, results, times, weights);
        }

        std::string repr()
        {
            return "History(Events=" + std::to_string(size) + ", Batches=" +
                   std::to_string(batches.size()) + ", Agents=" +
                   std::to_string(agents.size()) + ")";
        }

        int length()
        {
            return size;
        }

        void trueskill(const std::vector<std::vector<std::vector<std::string>>> &composition,
                       const std::vector<std::vector<double>> &results,
                       const std::vector<double> &times,
                       const std::vector<race_weight> &weights)
        {
            std::vector<int> o;
            if (times.size() > 0)
            {
                o = sortperm(times);
            }
            else
            {
                for (int i = 0; i < composition.size(); ++i)
                {
                    o.push_back(i);
                }
            }
            int i = 0;
            while (i < length())
            {
                int j = i + 1;
                double t;
                if (times.size() == 0)
                {
                    t = i + 1;
                }
                else
                {
                    t = times[o[i]];
                }
                while (times.size() > 0 && j < length() && times[o[j]] == t)
                {
                    j++;
                }
                std::vector<race_keys> compositions;
                std::vector<std::vector<double>> batch_results;
                std::vector<race_weight> batch_weights;
                for (int k = i; k < j; k++)
                {
                    compositions.push_back(composition[o[k]]);
                    batch_results.push_back(results[o[k]]);
                    if (weights.size() > 0)
                    {
                        batch_weights.push_back(weights[o[k]]);
                    }
                }
                Batch b(compositions, batch_results, t, agents, p_draw, batch_weights);
                batches.push_back(b);
                for (std::pair<std::string, Skill> a : b.getSkills())
                {
                    if (time)
                    {
                        agents[a.first].setLasttime(t);
                    }
                    else
                    {
                        agents[a.first].setLasttime(TTT::inf);
                    }
                    agents[a.first].setMessage(b.forward_prior_out(a.first));
                }
                i = j;
            }
        }

        std::pair<double, double> iteration()
        {
            std::pair<double, double> step = {0.0, 0.0};
            clean(agents);
            for (int j = batches.size() - 2; j >= 0; --j)
            {
                for (const std::pair<std::string, Skill> a : batches[j + 1].getSkills())
                {
                    agents[a.first].setMessage(batches[j + 1].backward_prior_out(a.first));
                }
                std::map<std::string, Gaussian> old_posteriors = batches[j].posteriors();
                batches[j].new_backward_info();
                std::map<std::string, Gaussian> new_posteriors = batches[j].posteriors();
                std::pair<double, double> new_step = dict_diff(old_posteriors, new_posteriors);
                step = max_tuple(step, new_step);
            }
            clean(agents);
            for (int j = 1; j < batches.size(); ++j)
            {
                for (auto a : batches[j - 1].getSkills())
                {
                    agents[a.first].setMessage(batches[j - 1].forward_prior_out(a.first));
                }
                std::map<std::string, Gaussian> old_posteriors = batches[j].posteriors();
                batches[j].new_forward_info();
                std::map<std::string, Gaussian> new_posteriors = batches[j].posteriors();
                std::pair<double, double> new_step = dict_diff(old_posteriors, new_posteriors);
                step = max_tuple(step, new_step);
            }

            if (batches.size() == 1)
            {
                std::map<std::string, Gaussian> old_posteriors = batches[0].posteriors();
                batches[0].convergence();
                std::map<std::string, Gaussian> new_posteriors = batches[0].posteriors();
                std::pair<double, double> new_step = dict_diff(old_posteriors, new_posteriors);
                step = max_tuple(step, new_step);
            }

            return step;
        }
        /*
        std::pair<double, double> convergence(double epsilon = TTT::EPSILON,
                                              int iterations = TTT::ITERATIONS,
                                              bool verbose = true)
        {
            std::pair<double, double> step = {TTT::inf, TTT::inf};
            int i = 0;
            while (gr_tuple(step, epsilon) && i < iterations)
            {
                if (verbose)
                {
                    std::cout << "Iteration = " << i << " ";
                }
                step = iteration();
                i++;
                if (verbose)
                {
                    std::cout << ", step = " << step.first << ", " << step.second << std::endl;
                }
            }
            if (verbose)
            {
                std::cout << "End" << std::endl;
            }
            return {step, i};
        }
        */

        std::map<std::string, std::vector<std::pair<double, Gaussian>>> learning_curves()
        {
            std::map<std::string, std::vector<std::pair<double, Gaussian>>> res;
            for (Batch &b : batches)
            {
                for (const std::pair<std::string, Gaussian> &pair : b.posteriors())
                {
                    std::string a = pair.first;
                    double t_p = b.getTime();
                    Gaussian posterior = b.posterior(a);
                    if (res.count(a))
                    {
                        res[a].push_back(std::make_pair(t_p, posterior));
                    }
                    else
                    {
                        std::pair<double, Gaussian> new_pair = std::make_pair(t_p, posterior);
                        res[a] = std::vector<std::pair<double, Gaussian>>{new_pair};
                    }
                }
            }
            return res;
        }
        /*
        double log_evidence()
        {
            double log_ev = 0.0;
            for (auto &b : batches)
            {
                for (auto &event : b.events())
                {
                    log_ev += std::log(event.evidence);
                }
            }
            return log_ev;
        }
        */
    };
}