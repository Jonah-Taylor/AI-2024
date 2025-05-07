#ifndef PSEUMENT_H
#define PSEUMENT_H

#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

class NeuralNetwork {
    
private:
    std::vector<int> layers;  // layers[layer]
    std::vector<std::vector<std::vector<double>>> weights;  // weights[layer][neuron_in][neuron_out]
    std::vector<std::vector<double>> biases;  // biases[layer][neuron]
    std::vector<std::vector<double>> activations;  // activations[layer][neuron]
    std::vector<std::vector<double>> z_vals;  // z_val[layer][neuron]

    std::vector<std::vector<std::vector<double>>> vel_weights;  // vel_weights[layer][neuron_in][neuron_out]
    std::vector<std::vector<double>> vel_biases;  // vel_biases[layer][neuron]

    double momentum = 0.5; // [0.0, 1.0) Multiplied by velocities in gradient decent optimizer

    // Helper function to initialize weights and biases
    double randomWeight() {
        return (double)rand() / RAND_MAX * 2 - 1;  // random value in range [-1, 1]
    }

    // Leaky ReLU activation function with fixed alpha (e.g., 0.01)
    double leakyRelu(double x, double alpha = 0.01) {
        return x > 0 ? x : alpha * x;
    }

    // Derivative of Leaky ReLU
    double leakyReluDerivative(double x, double alpha = 0.01) {
        return x > 0 ? 1 : alpha;
    }

public:

    NeuralNetwork(const std::vector<int>& layers);

    // Predict output from inputs
    std::vector<double> forward(const std::vector<double>& input);

    // Backward pass for gradient descent
    void backward(const std::vector<double>& input, const std::vector<double>& target, double learning_rate, float reward);

    // Train the network with multiple epochs
    void train(const std::vector<std::vector<double>>& inputs,
           const std::vector<std::vector<double>>& targets,
           int epochs, double learning_rate, float reward,
           bool print);

    void save(std::string filename);
    void load(std::string filename);

    std::vector<int> getLayers() {
        return layers;
    }
};


#endif // PSEUMENT_H