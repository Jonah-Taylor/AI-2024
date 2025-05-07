#ifndef PSEUMENT_H
#define PSEUMENT_H

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cstdlib>
#include <ctime>

class NeuralNetwork {
    
private:

    std::vector<int> layers;  // layers[layer]
    std::vector<std::vector<std::vector<double>>> weights;  // weights[layer][neuron_in][neuron_out]
    std::vector<std::vector<double>> biases;  // biases[layer][neuron]
    std::vector<std::vector<double>> activations;  // activations[layer][neuron]
    std::vector<std::vector<double>> z_vals;  // z_val[layer][neuron]

    std::vector<std::vector<std::vector<double>>> m_weights;  // 1st moment weights: m_weights[layer][neuron_in][neuron_out]
    std::vector<std::vector<std::vector<double>>> v_weights;  // 2nd moment weights: v_weights[layer][neuron_in][neuron_out]
    std::vector<std::vector<double>> m_biases;  // 1st moment biases: m_biases[layer][neuron]
    std::vector<std::vector<double>> v_biases;  // 2nd moment biases: m_biases[layer][neuron]

    double beta1 = 0.9; // Exponential decay rate for first moment
    double beta2 = 0.999; // Exponential decay rate for second moment
    double epsilon = 1e-10; // Constant to avoid division by zero
    double lambda = 0.01; // Weight decay coefficient
    int t = 0; // Time step for Adam bias correction


    // Helper function to initialize weights and biases
    double randomWeight() {
        return (double)rand() / RAND_MAX * 2 - 1;  // random value in range [-1, 1]
    }

    // Leaky ReLU activation function with fixed alpha (e.g., 0.01)
    double leakyRelu(double x, double alpha = 0.01) {
        return (x > 0) ? x : alpha * x;
    }

    // Derivative of Leaky ReLU
    double leakyReluDerivative(double x, double alpha = 0.01) {
        return x > 0 ? 1 : alpha;
    }

    // A place to initialize moments for Adam optimization
    void initializeAdamMiniStates();

public:

    NeuralNetwork(const std::vector<int>& layers);

    // Predict output from inputs
    std::vector<double> forward(const std::vector<double>& input);

    void getOutputDeltas(std::vector<std::vector<double>>& deltas,
            const std::vector<std::vector<double>>& inputs, const std::vector<double>& target, 
            int& batch_size, float& reward, int& correct, uint& loc);

    // Backward pass for gradient descent
    void backward(std::vector<std::vector<double>>& deltas, std::vector<std::vector<std::vector<double>>>& avg_grad_w, 
        std::vector<std::vector<double>>& avg_grad_b, double& learning_rate);

    void updateNeurons(std::vector<std::vector<std::vector<double>>>& avg_grad_w, 
        std::vector<std::vector<double>>& avg_grad_b, double& learning_rate, int& batchSize);

    // Train the network with multiple epochs
    void train(const std::vector<std::vector<double>>& inputs,
           const std::vector<std::vector<double>>& targets,
           int& epochs, int& batch_size, double& learning_rate, float& reward,
           bool& print);

    void save(std::string filename);
    void load(std::string filename);

    std::vector<int> getLayers() {
        return layers;
    }
};


#endif // PSEUMENT_H