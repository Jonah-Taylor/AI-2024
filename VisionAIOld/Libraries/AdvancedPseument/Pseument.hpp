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
    double epsilon = 1e-10;   // Small constant to avoid division by zero
    double lambda = 0.01;     // Weight decay coefficient
    int t = 0;                // Time step for Adam bias correction

    // Helper function to initialize random weights
    double randomWeight() {
        static random_device rd;
        static mt19937 gen(rd());
        static uniform_real_distribution<> dis(-1.0, 1.0);
        return dis(gen);
    }

    // Leaky ReLU activation function
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

    // Function to initialize Adam moments
    void initializeAdamMiniStates();

public:

    NeuralNetwork(const vector<int>& layers);

    // Predict output from inputs using forward pass
    VectorXd forward(const VectorXd& input);

    void getOutputDeltas(vector<VectorXd>& deltas, const VectorXd& target, 
        int& correct, unsigned int& loc);

    // Backward pass for gradient descent
    void backward(vector<VectorXd>& deltas, vector<MatrixXd>& avg_grad_w, 
        vector<VectorXd>& avg_grad_b, double learning_rate);

    void updateNeurons(vector<MatrixXd>& avg_grad_w, 
        vector<VectorXd>& avg_grad_b, double learning_rate, int batch_size);

    // Train the network with multiple epochs
    void train(vector<vector<double>>& X, vector<vector<double>>& Y, int epochs, 
        int batch_size, double learning_rate, float& reward, bool print);

    void save(const string& filename);
    void load(const string& filename);

    vector<int> getLayers() const {
        return layers;
    }

};

#endif // PSEUMENT_H
