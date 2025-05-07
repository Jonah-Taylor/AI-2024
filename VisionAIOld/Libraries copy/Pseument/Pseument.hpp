#ifndef PSEUMENT_H
#define PSEUMENT_H

#include "Libraries/Eigen/Dense"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cstdlib>
#include <ctime>

// Bring standard and Eigen types into scope
using namespace std;
using namespace Eigen;

class NeuralNetwork {

private:
    vector<int> layers;            // Layer sizes
    vector<MatrixXd> weights;      // Weights[layer] as Eigen matrices
    vector<VectorXd> biases;       // Biases[layer] as Eigen vectors
    vector<VectorXd> activations;  // Activations[layer] as Eigen vectors
    vector<VectorXd> z_vals;       // Z values (pre-activation) as Eigen vectors

    // Adam optimizer moments
    vector<MatrixXd> m_weights;
    vector<MatrixXd> v_weights;
    vector<VectorXd> m_biases;
    vector<VectorXd> v_biases;

    double beta1 = 0.9;       // Exponential decay rate for first moment
    double beta2 = 0.999;     // Exponential decay rate for second moment
    double epsilon = 1e-8;   // Small constant to avoid division by zero
    double lambda = 0.0;     // Weight decay coefficient (Set to 10% of learning rate)
    int t = 0;                // Time step for Adam bias correction
    int tested = 0;
    int correct = 0;

    bool debugging = false;

    double randomWeight() {
        static random_device rd;
        static mt19937 gen(rd());
        static uniform_real_distribution<> dis(-1.0, 1.0);
        return dis(gen);
    }

    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    VectorXd sigmoid(const VectorXd& x) {
        return x.unaryExpr([](double v) { return 1.0 / (1.0 + exp(-v)); });
    }

    double sigmoidDerivative(double x) {
        double sig = sigmoid(x);
        return sig * (1.0 - sig);
    }
    
    VectorXd sigmoidDerivative(const VectorXd& x) {
        return x.unaryExpr([](double v) {
            double sig = 1.0 / (1.0 + exp(-v));
            return sig * (1.0 - sig);
        });
    }

    double leakyRelu(double x, double alpha = 0.01) {
        return (x > 0) ? x : alpha * x;
    }
    
    VectorXd leakyRelu(const VectorXd& x, double alpha = 0.01) {
        return x.unaryExpr([alpha](double v) { return v > 0 ? v : alpha * v; });
    }

    double leakyReluDerivative(double x, double alpha = 0.01) {
        return (x > 0) ? 1 : alpha;
    }

    VectorXd leakyReluDerivative(const VectorXd& x, double alpha = 0.01) {
        return x.unaryExpr([alpha](double v) { return v > 0 ? 1 : alpha; });
    }

    void initializeAdamMiniStates();

public:

    NeuralNetwork(const vector<int>& layers);

    int forward(const vector<double>& input);

    VectorXd forward(const VectorXd& input);

    void getOutputDeltas(vector<VectorXd>& deltas, const VectorXd& target);

    void backward(vector<VectorXd>& deltas, vector<MatrixXd>& avg_grad_w, 
        vector<VectorXd>& avg_grad_b, double learning_rate);

    void stepSGD(vector<MatrixXd>& avg_grad_w, 
        vector<VectorXd>& avg_grad_b, double learning_rate, int batch_size);

    void stepAdamW(vector<MatrixXd>& avg_grad_w, 
        vector<VectorXd>& avg_grad_b, double learning_rate, int batch_size);

    void train(vector<vector<double>>& X, vector<vector<double>>& Y, int& epochs, 
        int& batch_size, double& learning_rate, bool print);

    void save(const string& filename);
    void load(const string& filename);

    vector<int> getLayers() const {
        return layers;
    }

};

#endif // PSEUMENT_H
