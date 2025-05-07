// tester.cpp

#include <iostream>
#include <vector>

#include "Libraries/Eigen/Dense"
#include "Libraries/Pseument/Pseument.hpp"  // Ensure this matches the header file for your neural network

using namespace std;
using namespace Eigen;

int main() {
    // Define the XOR dataset
    vector<vector<double>> X = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    vector<vector<double>> Y = {
        {0}, // Output for [0, 0]
        {1}, // Output for [0, 1]
        {1}, // Output for [1, 0]
        {0}  // Output for [1, 1]
    };

    // Neural network architecture: 2 input neurons, 3 hidden neurons, 1 output neuron
    vector<int> layers = {2, 3, 1};
    NeuralNetwork nn(layers);

    // Training parameters
    int epochs = 1000;
    int batch_size = 1;
    double learning_rate = 3;
    float reward = 0.0;
    
    // Train the network
    nn.train(X, Y, epochs, batch_size, learning_rate, reward, true);

    // Test the trained network
    cout << "Testing the trained neural network on XOR data:\n";
    for (size_t i = 0; i < X.size(); ++i) {
        VectorXd input(2);
        input << X[i][0], X[i][1];

        VectorXd output = nn.forward(input);
        cout << "Input: [" << X[i][0] << ", " << X[i][1] << "] -> Output: " << output(0) << " (Expected: " << Y[i][0] << ")\n";
    }

    return 0;
}