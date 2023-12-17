#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <ctime>
#include "ttt_main.h"


struct Data
{
    std::string match_id;
    std::string double_;
    int round_number;
    std::string w1_id;
    std::string w1_name;
    std::string w2_id;
    std::string w2_name;
    std::string l1_id;
    std::string l1_name;
    std::string l2_id;
    std::string l2_name;
    std::string time_start; // 追加: 日付のメンバー変数
    std::string time_end;
    std::string ground;
    int tour_id;
    std::string tour_name;
};

int main(void)
{

    std::ifstream file("history.csv");
    std::vector<Data> data;
    std::string line;

    std::getline(file, line);
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        while (std::getline(ss, token, ','))
        {
            tokens.push_back(token);
        }

        Data d;
        d.match_id = tokens[0];
        d.double_ = tokens[1];
        d.round_number = std::stoi(tokens[2]);
        d.w1_id = tokens[3];
        d.w1_name = tokens[4];
        d.w2_id = tokens[5];
        d.w2_name = tokens[6];
        d.l1_id = tokens[7];
        d.l1_name = tokens[8];
        d.l2_id = tokens[9];
        d.l2_name = tokens[10];
        d.time_start = tokens[11]; // 追加: 日付をセットする
        d.time_end = tokens[12];
        d.ground = tokens[13];
        d.tour_id = std::stoi(tokens[14]);
        d.tour_name = tokens[15];

        data.push_back(d);
    }

    std::vector<std::vector<std::vector<std::string>>> composition;

    for (const auto &d : data)
    {
        std::vector<std::vector<std::string>> matchComposition;
        std::vector<std::string> winners;
        std::vector<std::string> losers;

        if (d.double_ == "t")
        {
            winners.push_back(d.w1_id);
            winners.push_back(d.w2_id);
            losers.push_back(d.l1_id);
            losers.push_back(d.l2_id);
        }
        else
        {
            winners.push_back(d.w1_id);
            losers.push_back(d.l1_id);
        }

        matchComposition.push_back(winners);
        matchComposition.push_back(losers);
        composition.push_back(matchComposition);
    }

    std::vector<double> days;

    for (const auto &d : data)
    {
        std::tm tm = {};
        strptime(d.time_start.c_str(), "%Y-%m-%d", &tm);
        std::time_t t = mktime(&tm);
        double day = t / (60 * 60 * 24);
        days.push_back(day);
    }
    std::vector<std::vector<double>> results;
    for (const std::vector<std::vector<std::string>> &d : composition)
    {
        std::vector<double> res;
        for (int i=1; i < d.size()+1; i++){
            res.push_back((double)i);
        }
        results.push_back(res);
    }

    TTT::History h = TTT::History(composition, results, days);
    std::map<std::string, std::vector<std::pair<double, TTT::Gaussian>>> output;
    output = h.learning_curves();
    std::vector<std::pair<double, TTT::Gaussian>> output_data;
    output_data = output["lf73"];
    for (const auto &d : output_data){
        std::cout << "  time: " << d.first;
        std::cout << "    mu: " << d.second.getMu();
        std::cout << " sigma: " << d.second.getSigma() << std::endl;
    }
    std::cout << composition.size();

    return 0;
}
