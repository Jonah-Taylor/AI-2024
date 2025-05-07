// Filename: Pseument.cpp
// Author: Jonah Taylor
// Date: November 16, 2024
// Description: Pseument (Pseudo Mantis) is my first neural network

#include "Pseument.hpp"

NeuralNetwork::NeuralNetwork(const std::vector<int>& layers) : layers(layers) {
    // Seed for random number generation
    std::srand(std::time(nullptr));

    // Initialize weight matrices and biases
    for (size_t i = 1; i < layers.size(); ++i) {

        // Create weighted connection between neurons in different layers
        std::vector<std::vector<double>> layer_weights(layers[i - 1], std::vector<double>(layers[i]));
        // Create neuron biases
        std::vector<double> layer_biases(layers[i], 0.0);

        // Randomize weights
        for (auto& neuron_connections : layer_weights) {
            for (double& w : neuron_connections) {
                w = randomWeight();
            }
        }
        weights.push_back(layer_weights);
        biases.push_back(layer_biases);
    }

    // Initialize velocity for weights and biases
    vel_weights.resize(weights.size());
    for (size_t l = 0; l < weights.size(); ++l) {
        // vel_weights = vel_weights[layerSize][neurons[l]][neurons[l+1]]
        vel_weights[l].resize(weights[l].size(), std::vector<double>(weights[l][0].size(), 0.0));
    }
    vel_biases.resize(biases.size());
    for (size_t l = 0; l < biases.size(); ++l) {
        // vel_biases = vel_biases[layerSize][neurons[l]]
        vel_biases[l].resize(biases[l].size(), 0.0);
    }

    // Initialize activations and z-values (linear combinations) for each layer
    for (size_t i = 0; i < layers.size(); ++i) {
        activations.push_back(std::vector<double>(layers[i]));
        z_vals.push_back(std::vector<double>(layers[i]));
    }
}

    // Forward pass
std::vector<double> NeuralNetwork::forward(const std::vector<double>& input) {
    // Set first layer of neurons to input values
    activations[0] = input;

    for (size_t l = 1; l < layers.size(); ++l) { // Iterate through all layers
        for (int j = 0; j < layers[l]; ++j) {   // Iterate through all neurons

            // Set z_value to neuron bias
            z_vals[l][j] = biases[l - 1][j];

            // Add the summation of all activations multiplied by weights in the previous layer
            for (int k = 0; k < layers[l - 1]; ++k) {
                z_vals[l][j] += activations[l - 1][k] * weights[l - 1][k][j];
            }
            // Plug result into sigmoid to make the activation between -1 and 1
            activations[l][j] = leakyRelu(z_vals[l][j]);
        }
    }
    // Return the output of the last layer of neurons
    return activations.back();
}

// Backward pass for gradient descent

void NeuralNetwork::backward(const std::vector<double>& input, const std::vector<double>& target, double learning_rate, float reward) {
    // Create layer structure to determine output sensitivity to each neuron
    std::vector<std::vector<double>> deltas(layers.size());

    // Fill back layer with output neurons
    deltas.back() = std::vector<double>(layers.back());

    // Calculate output error (How far the prediction was off)
    for (int i = 0; i < layers.back(); ++i) {   // Iterate over output neurons

        // dC/dA of last layer using Mean Squared Error (MSE)
        double error = activations.back()[i] - target[i] + reward;

        // dC/dZ = dC/dA * dA/dZ: last layer; Multiply error by derivative of the activation function
        deltas.back()[i] = error * leakyReluDerivative(activations.back()[i]);
    }

    // Backpropagate errors to hidden layers
    for (int l = layers.size() - 2; l > 0; --l) {

        // Fill layer with correct number of neurons
        deltas[l] = std::vector<double>(layers[l]);
        
        // Iterate over neurons in layer
        for (int i = 0; i < layers[l]; ++i) {
            double influence = 0.0;

            // Iterate over neurons in following layer
            for (int j = 0; j < layers[l + 1]; ++j) {

                // Determine the influence of this layer to the next: dC/dA
                influence += deltas[l + 1][j] * weights[l][i][j];
            }
            // Compute the delta dC/dZ for the current neuron in layer l: dC/dA * dA/dZ
            deltas[l][i] = influence * leakyReluDerivative(activations[l][i]);
        }
    }

    // Update weights and biases
    for (size_t l = 1; l < layers.size(); ++l) {
        for (int i = 0; i < layers[l]; ++i) {
            for (int j = 0; j < layers[l - 1]; ++j) {

                // weight_gradient = dC/dZ * dZ/dW(ij) = dC/dW(ij)
                double weight_gradient = deltas[l][i] * activations[l - 1][j];

                // Update velocity of weights
                vel_weights[l-1][j][i] *= momentum;
                vel_weights[l-1][j][i] -= learning_rate * weight_gradient;
                    
                // Nudge weight in the direction that decreases the cost function
                //weights[l - 1][j][i] -= learning_rate * gradient;

                // Update weight values with velocity in direction decreasing cost function
                weights[l - 1][j][i] += vel_weights[l-1][j][i];

            }
            // Nudge bias in the direction that decreases the cost function
            // deltas[l][i] * 1 = dC/dZ * dZ/dB(i) = dC/dB(i)
            double bias_gradient = deltas[l][i];

            // Update velocity of biases
            vel_biases[l - 1][i] *= momentum;
            vel_biases[l - 1][i] -= learning_rate * bias_gradient;

            //biases[l - 1][i] -= learning_rate * deltas[l][i];

            // Update bias values with velocity in direction decreasing the cost function
            biases[l - 1][i] += vel_biases[l - 1][i];
        }
    }
}

// Train the network with multiple epochs
void NeuralNetwork::train(const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& targets,
        int epochs, double learning_rate, float reward,
        bool print) {
    
    // Iterate over epochs (passes through dataset)
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // To count the number of correct predictions
        int correct_predictions = 0; 

        // Iterate over training inputs
        for (size_t i = 0; i < inputs.size(); ++i) {
            forward(inputs[i]);
            backward(inputs[i], targets[i], learning_rate, reward);

            // Calculate the number of correct predictions
            for (size_t j = 0; j < targets[i].size(); ++j) {
                // If the output is 0.5 or more, consider it 1; otherwise, consider it 0
                int predicted = (activations.back()[j] >= 0.5) ? 1 : 0;
                int actual = (targets[i][j] >= 0.5) ? 1 : 0;
                // If the predicted value matches the actual value, it's correct
                if (predicted == actual) {
                    correct_predictions++;
                }
            }
        }

        // Calculate the accuracy of AI predicting values in epoch
        double accuracy = (double)correct_predictions / (inputs.size() * targets[0].size()) * 100;

        // Print accuracy every 100 epochs
        if (print) {
            std::cout << "Epoch " << epoch + 1 << ", Accuracy: " << accuracy << "%" << std::endl;
        }
    }
}

void NeuralNetwork::save(std::string filename) {

    // Open/create file
    std::ofstream file(filename);
    if(!file) {
        std::cerr << "File couldn't be accessed for saving\n";
        return;
    }

    // Save layer data in file
    file << layers.size() << "\n";
    for(int& layer : layers) {
        file << layer << " ";
    }
    file << "\n\n";

    // Save weight data in file
    for(size_t i = 0; i < weights.size(); i++) {
        for(size_t j = 0; j < weights[i].size(); j++) {
            for(size_t k = 0; k < weights[i][j].size(); k++) {
                file << weights[i][j][k] << " ";
            }
            file << "\n";
        }
        file << "\n";
    }

    // Save bias data in file
    for(size_t i = 0; i < biases.size(); i++) {
        for(size_t j = 0; j < biases[i].size(); j++) {
            file << biases[i][j] << " ";
        }
        file << "\n";
    }
    file << "\n";

    file.close();
}

void NeuralNetwork::load(std::string filename) {

    // Open File
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "File couldn't be accessed for loading\n";
        return;
    }

    // Load layers
    int layerSize;
    file >> layerSize;
    layers.resize(layerSize);
    for (int& layer : layers) {
        file >> layer;
    }

    // Load weights
    weights.resize(layers.size() - 1);
    for (size_t i = 0; i < weights.size(); i++) {
        weights[i].resize(layers[i]);
        for (size_t j = 0; j < weights[i].size(); j++) {
            weights[i][j].resize(layers[i + 1]);
            for (double& weight : weights[i][j]) {
                file >> weight;
            }
        }
    }

    // Load biases
    biases.resize(layers.size() - 1);
    for (size_t i = 0; i < biases.size(); i++) {
        biases[i].resize(layers[i + 1]);
        for (double& bias : biases[i]) {
            file >> bias;
        }
    }

    file.close();
}
