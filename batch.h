#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <map>
#include <set>

#include "params.h"
#include "gaussian.h"
#include "utils.h"
#include "team.h"
#include "game.h"

namespace TTT
{
    /*
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
    */
    struct Skill
    {
        Skill(Gaussian _f = Ninf, Gaussian _b = Ninf, Gaussian _l = Ninf, double _e = 0)
            : forward(_f), backward(_b), likelihood(_l), elapsed(_e) {}

        Gaussian forward;
        Gaussian backward;
        Gaussian likelihood;
        double elapsed;
    };

    /*
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
    */
    class Agent
    {
    public:
        Agent() = default; // Default constructor
        Agent(Player p, Gaussian m = Ninf, bool l = false)
            : player(p), message(m), last_time(l) {}

        Gaussian receive(const double elapsed) const
        {
            if (message != Ninf)
            {
                return message.forget(player.gamma, elapsed);
            }
            else
            {
                return player.prior;
            }
        }

        Player player;
        Gaussian message;
        bool last_time;
    };

    /*
    def clean(agents: Dict[str, Agent], last_time: bool = False):
        for a in agents:
            agents[a].message = Ninf
            if last_time:
                agents[a].last_time = -inf
    */

    void clean(std::map<std::string, Agent> &agents, bool last_time = false)
    {
        for (auto &a : agents)
        {
            a.second.message = Ninf;
            if (last_time)
            {
                a.second.last_time = -inf;
            }
        }
    }

    /*
    class Item(object):
        def __init__(self, name: str, likelihood: Gaussian):
            self.name = name
            self.likelihood = likelihood
    */
   class Item
   {
    public:
         Item(std::string _name, Gaussian _likelihood)
              : name(_name), likelihood(_likelihood) {}
    
         std::string name;
         Gaussian likelihood;
   };
    /*
    class Team(object):
        def __init__(self, items: List[Item], output: float):
            self.items = items
            self.output = output
    */
    class Team
    {
    public:
        Team(std::vector<Item> _items, double _output)
            : items(_items), output(_output) {}

        std::vector<Item> items;
        double output;
    };

    /*
    class Event(object):
        def __init__(self, teams: List[Team], evidence: float, weights: List[float]):
            self.teams = teams
            self.evidence = evidence
            self.weights = weights

        @property
        def result(self) -> List[float]:
            return [team.output for team in self.teams]
    */
    class Event
    {
    public:
        Event(std::vector<Team> teams, double evidence, std::vector<double> weights)
            : teams(teams), evidence(evidence), weights(weights) {}

        std::vector<double> result() const
        {
            std::vector<double> res;
            for (auto &team : teams)
            {
                res.push_back(team.output);
            }
            return res;
        }

        std::vector<Team> teams;
        double evidence;
        std::vector<double> weights;
    };
    /*
    def compute_elapsed(last_time: float, actual_time: float) -> float:
        if last_time == -inf:
            return 0
        else:
            if last_time == inf:
                return 1
            else:
                return actual_time - last_time
    */

    double compute_elapsed(double last_time, double actual_time)
    {
        if (last_time == -inf)
        {
            return 0;
        }
        else
        {
            if (last_time == inf)
            {
                return 1;
            }
            else
            {
                return actual_time - last_time;
            }
        }
    }

    /*
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
            self._init_events(results)
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
    */

    class Batch
    {
    public:
        Batch(
            std::vector<std::vector<std::vector<std::string>>> _games,
            std::vector<std::vector<double>> _results,
            double _time,
            std::map<std::string, Agent> _agents,
            double _p_draw,
            std::vector<double> _weights)
            : agents(_agents), time(_time), p_draw(_p_draw), weights(_weights)
        {
            std::set<std::string> batch_unique_agents = get_agent_set(_games);
            init_skills(batch_unique_agents);
            init_events(_games, _results);
            iteration();
        }

        std::set<std::string> get_agent_set(std::vector<std::vector<std::vector<std::string>>> games)
        {
            std::set<std::string> this_batch_agents;
            for (auto &teams : games)
            {
                for (auto &team : teams)
                {
                    for (auto &agent : team)
                    {
                        this_batch_agents.insert(agent);
                    }
                }
            }
            return this_batch_agents;
        }

        void init_skills(std::set<std::string> agents)
        {
            for (auto &a : agents)
            {
                double elapsed = compute_elapsed(this->agents[a].last_time, time);
                skills[a] = Skill(this->agents[a].receive(elapsed), Ninf, Ninf, elapsed);
            }
        }

        void init_events(std::vector<std::vector<std::vector<std::string>>> games, std::vector<std::vector<double>> results)
        {
            for (int event_idx = 0; event_idx < games.size(); event_idx++)
            {
                std::vector<Team> event_teams;
                for (int team_idx = 0; team_idx < games[event_idx].size(); team_idx++)
                {
                    std::vector<Item> team_items;
                    for (int a = 0; a < games[event_idx][team_idx].size(); a++)
                    {
                        team_items.push_back(Item(games[event_idx][team_idx][a], Ninf));
                    }
                    double team_result;
                    if (results.size() > 0)
                    {
                        team_result = results[event_idx][team_idx];
                    }
                    else
                    {
                        team_result = games[event_idx].size() - team_idx - 1;
                    }
                    Team team(team_items, team_result);
                    event_teams.push_back(team);
                }
                Event event(event_teams, 0.0, weights);
                events.push_back(event);
            }
        }

        Gaussian posterior(const std::string agent) const
        {
            return skills.at(agent).likelihood * skills.at(agent).backward * skills.at(agent).forward;
        }

        Player within_prior(const Item &item) const
        {
            const Player& r = this->agents.at(item.name).player;
            double mu, sigma;
            Gaussian _p = posterior(item.name) / item.likelihood;
            Player res(_p, r.beta, r.gamma);
            return res;
        }

        std::vector<std::vector<Player>> within_priors(const int event_idx) const
        {
            std::vector<std::vector<Player>> result;
            for (auto &team : events[event_idx].teams)
            {
                std::vector<Player> inner_result;
                for (auto &item : team.items)
                {
                    inner_result.push_back(within_prior(item));
                }
                result.push_back(inner_result);
            }
            return result;
        }
        void iteration(int _from = 0)
        {
            for (int event_idx = _from; event_idx < events.size(); event_idx++)
            {
                std::vector<std::vector<Player>> teams = within_priors(event_idx);
                std::vector<double> result = events[event_idx].result();
                Game game(teams, result, p_draw, weights);
                for (int team_idx = 0; team_idx < events[event_idx].teams.size(); team_idx++)
                {
                    for (int item_idx = 0; item_idx < events[event_idx].teams[team_idx].items.size(); item_idx++)
                    {
                        std::cout << "event idx: " << event_idx << ", team idx: " << team_idx << ", item idx: " << item_idx << std::endl;
                        skills[events[event_idx].teams[team_idx].items[item_idx].name].likelihood = skills[events[event_idx].teams[team_idx].items[item_idx].name].likelihood / events[event_idx].teams[team_idx].items[item_idx].likelihood * game.likelihoods[team_idx][item_idx];
                        events[event_idx].teams[team_idx].items[item_idx].likelihood = game.likelihoods[team_idx][item_idx];
                    }
                }
                events[event_idx].evidence = game.evidence;
            }
        }
        Gaussian forward_prior_out(const std::string agent) const
        {
            return skills.at(agent).forward * skills.at(agent).likelihood;
        }

        std::map<std::string, Skill> skills;
        std::map<std::string, Agent> agents;
        std::vector<Event> events;
        double time;
        double p_draw;
        std::vector<double> weights;
    };
}