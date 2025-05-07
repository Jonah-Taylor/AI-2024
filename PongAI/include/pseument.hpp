#ifndef PSEUMENT_H
#define PSEUMENT_H

#include "layer.hpp"
#include "Eigen/Dense"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace Eigen;

class NeuralNetwork {

private:

    vector<DenseLayer> layers;
    
    int t = 0;
    int tested = 0;
    int correct = 0;

    bool debugging = false;

public:

    NeuralNetwork(const vector<int>& layers);

    vector<double> forward(const vector<double>& input);
    VectorXd forward(const VectorXd& input);

    void getOutputDeltas(const VectorXd& target);
    void backward();

    void stepSGD(double& lr, int& batch_size);
    void stepAdamW(double& lr, int& batch_size, int& t);

    void train(vector<vector<double>>& X, vector<vector<double>>& Y, int& epochs, 
        int& batch_size, double& learning_rate, bool print);

    void save(const string& filename);
    void load(const string& filename);

    vector<uint> getLayerSizes();

};

#endif // PSEUMENT_H
