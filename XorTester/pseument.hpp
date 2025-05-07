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

    vector<uint> l_size = {0};
    uint l_type = 0;
    string a_func_name = "";

    MakeLayer(vector<uint> ls) : l_size(ls), l_type(0) {};
    MakeLayer(vector<uint> ls, string lt, string afn) : l_size(ls), a_func_name(afn) {

        if (lt == "dense") {
            l_type = 0;
        } else if (lt == "convolutional") {
            l_type = 1;
        } else if (lt == "pooling") {
            l_type = 2;
        } else {
            l_type = 0; // Default to dense if unknown
        }

    };
};

class NeuralNetwork {

private:

    vector<ConvoLayer> c_layers;
    vector<DenseLayer> d_layers;
    int descent = 0;

    int t = 0;
    int tested = 0;
    int correct = 0;

    bool debugging = false;

public:

    NeuralNetwork(const vector<MakeLayer>& layers);

    vector<double> forward(const vector<double>& input);
    VectorXd forward(const VectorXd& input);

    void getOutputDeltas(const VectorXd& target);
    void backward();

    void stepSGD(double& lr, int& batch_size);
    void stepAdamW(double& lr, int& batch_size, int& t);

    void train(vector<vector<double>>& X, vector<vector<double>>& Y, int& epochs, 
        int& batch_size, double& learning_rate, string da, bool print);

    void save(const string& filename);
    void load(const string& filename);

    vector<uint> getLayerSizes();

};

#endif // PSEUMENT_H
