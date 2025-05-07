#ifndef PSEUMENT_H
#define PSEUMENT_H

#include "convolayer.hpp"
#include "denselayer.hpp"
#include "Eigen/Dense"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cstdlib>
#include <ctime>
#include <memory>

using namespace std;
using namespace Eigen;

struct MakeLayer {

    size_t l_type = 0;
    string a_func_name = "";

    vector<size_t> l_size = {0};
    vector<size_t> o_size = {0, 0};
    size_t k_size;

    MakeLayer(string lt, string afn, vector<size_t> ls, vector<size_t> os = {0, 0}, size_t ks = 3) : a_func_name(afn), l_size(ls), o_size(os), k_size(ks) {

        if (lt == "dense") {
            l_type = 0;
            l_size.push_back(1);
        } else if (lt == "convo") {
            l_type = 1;
        } else if (lt == "pool") {
            l_type = 2;
        } else {
            l_type = 0;
            l_size.push_back(1);
        }
        if(l_type == 1 && o_size[0] == 0 && o_size[1] == 0) {
            o_size[0] = l_size[0];
            o_size[1] = l_size[1];
        }

    };
};

class NeuralNetwork {

private:

    vector<unique_ptr<Layer>> layers;

    size_t descent = 0;
    size_t t = 0;
    size_t tested = 0;
    size_t correct = 0;

    bool debugging = false;

public:

    NeuralNetwork(const vector<MakeLayer>& layers);

    vector<double> forward(const vector<double>& in);
    VectorXd forward(const VectorXd& in);
    void getOutputDeltas(const VectorXd& target);
    void backward();
    void stepSGD(double& lr, size_t& bs);
    void stepAdamW(double& lr, size_t& bs, size_t& t);

    void train(vector<vector<double>>& X, vector<vector<double>>& Y, size_t& epochs, 
        size_t& bs, double& lr, string da, bool print);

    void save(const string& fn);
    void load(const string& fn);

    vector<size_t> getLayerSizes();

};

#endif
