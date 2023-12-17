#pragma once

#include <vector>
#include "gaussian.h"
namespace TTT
{
    class Player
    {
    private:
        Gaussian prior;
        double beta;
        double gamma;
        Gaussian prior_draw;

    public:
        Player(Gaussian p = Gaussian(TTT::MU, TTT::SIGMA), double b = TTT::BETA, double g = TTT::GAMMA, Gaussian pd = TTT::Ninf)
        {
            this->prior = p;
            this->beta = b;
            this->gamma = g;
            this->prior_draw = pd;
        }

        Gaussian performance() const
        {
            return Gaussian(prior.getMu(), std::sqrt(prior.getSigma() * prior.getSigma() + beta * beta));
        }

        std::string to_string() const
        {
            return "Player(Gaussian(mu=" + std::to_string(prior.getMu()) + ", sigma=" + std::to_string(prior.getSigma()) + "), beta=" + std::to_string(beta) + ", gamma=" + std::to_string(gamma) + ")";
        }

        double getGamma() const
        {
            return this->gamma;
        }

        Gaussian getPrior() const
        {
            return this->prior;
        }

        double getBeta() const
        {
            return this->beta;
        }

        Gaussian getPriorDraw() const
        {
            return this->prior_draw;
        }
    };

    class TeamVariable
    {
    private:
        Gaussian prior;
        Gaussian likelihood_lose;
        Gaussian likelihood_win;
        Gaussian likelihood_draw;

    public:
        TeamVariable(Gaussian p, Gaussian ll, Gaussian lw, Gaussian ld) : prior(p), likelihood_lose(ll), likelihood_win(lw), likelihood_draw(ld) {}

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

        Gaussian getPrior()
        {
            return this->prior;
        }

        void setPrior(Gaussian p)
        {
            this->prior = p;
        }

        Gaussian getLikelihoodLose() const
        {
            return this->likelihood_lose;
        }

        void setLikelihoodLose(Gaussian l)
        {
            this->likelihood_lose = l;
        }

        Gaussian getLikelihoodWin() const
        {
            return this->likelihood_win;
        }

        void setLikelihoodWin(Gaussian l)
        {
            this->likelihood_win = l;
        }

        Gaussian getLikelihoodDraw() const
        {
            return this->likelihood_draw;
        }

        void setLikelihoodDraw(Gaussian l)
        {
            this->likelihood_draw = l;
        }
    };

    Gaussian team_performance(std::vector<Player> &team, team_weight &weights)
    {
        Gaussian res = TTT::N00;
        for (int i = 0; i < team.size(); i++)
        {
            res = res + team[i].performance() * weights[i];
        }
        return res;
    }

    struct Skill
    {
        Gaussian forward;
        Gaussian backward;
        Gaussian likelihood;
        double elapsed;
        Skill(Gaussian _f = TTT::Ninf, Gaussian _b = TTT::Ninf, Gaussian _l = TTT::Ninf, double _e = 0.0) : forward(_f), backward(_b), likelihood(_l), elapsed(_e) {}
    };

    class Agent
    {
    private:
        Player player;
        Gaussian message;
        double last_time;
    public:
        Agent(){};
        Agent(Player p, Gaussian m, double l)
        {
            this->player = p;
            this->message = m;
            this->last_time = l;
        }

        Gaussian receive(double elapsed)
        {
            Gaussian res;
            if (message != Ninf)
            {
                res = message.forget(player.getGamma(), elapsed);
            }
            else
            {
                res = player.getPrior();
            }
            return res;
        }

        Player getPlayer()
        {
            return this->player;
        }

        void setPlayer(Player p)
        {
            this->player = p;
        }

        Gaussian getMessage()
        {
            return this->message;
        }

        void setMessage(Gaussian m)
        {
            this->message = m;
        }

        double getLasttime()
        {
            return this->last_time;
        }

        void setLasttime(double l)
        {
            this->last_time = l;
        }
    };

    struct Item
    {
        std::string name;
        Gaussian likelihood;
        Item(std::string _n, Gaussian _l) : name(_n), likelihood(_l) {}
    };

    struct Team
    {
        std::vector<Item> items;
        double output;
        Team(std::vector<Item> _i, double _o) : items(_i), output(_o) {}
    };

}