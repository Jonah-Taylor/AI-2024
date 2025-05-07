#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

class NeuralNetwork {
private:
    std::vector<int> layers;  // layers[layer]
    std::vector<std::vector<std::vector<double>>> weights;  // weights[layer][neuron_in][neuron_out]
    std::vector<std::vector<double>> biases;  // biases[layer]pneuron
    std::vector<std::vector<double>> activations;  // activations[layer][neuron]
    std::vector<std::vector<double>> z_values;  // z_val[layer][neuron]

    // Helper function to initialize weights and biases
    double randomWeight() {
        return (double)rand() / RAND_MAX * 2 - 1;  // random value in range [-1, 1]
    }

    // Sigmoid activation function
    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    // Derivative of sigmoid
    double sigmoidDerivative(double x) {
        return x * (1.0 - x);
    }

public:
    NeuralNetwork(const std::vector<int>& layers) : layers(layers) {
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
            // Add layer data to weights
            weights.push_back(layer_weights);
            // Add layer data to biases
            biases.push_back(layer_biases);
        }

        // Initialize activations and z-values (linear combinations) for each layer
        for (size_t i = 0; i < layers.size(); ++i) {
            // Designates an activation and z_value for every neuron
            activations.push_back(std::vector<double>(layers[i]));
            z_values.push_back(std::vector<double>(layers[i]));
        }
    }

        // Forward pass
    std::vector<double> forward(const std::vector<double>& input) {
        // Set first layer of neuron to input
        activations[0] = input;

        for (size_t l = 1; l < layers.size(); ++l) {
            for (int j = 0; j < layers[l]; ++j) {
                // Set z_value to neuron bias
                z_values[l][j] = biases[l - 1][j];
                // Add the summation of all activations multiplied by weights in the previous layer
                for (int k = 0; k < layers[l - 1]; ++k) {
                    z_values[l][j] += activations[l - 1][k] * weights[l - 1][k][j];
                }
                // Plug result into sigmoid to make the activation between -1 and 1
                activations[l][j] = sigmoid(z_values[l][j]);
            }
        }
        // Return the output of the last layer of neurons
        return activations.back();
    }

    // Backward pass for gradient descent
    
    void backward(const std::vector<double>& input, const std::vector<double>& target, double learning_rate) {
        // Create layer structure to determine output sensitivity to each neuron
        std::vector<std::vector<double>> deltas(layers.size());

        // Fill back layer with output neurons
        deltas.back() = std::vector<double>(layers.back());

        // Calculate output error (How far the prediction was off)
        for (size_t i = 0; i < layers.back(); ++i) {

            // dC/dA of first layer (factor of 2 is not nessecary)
            double error = activations.back()[i] - target[i];

            // dC/dA * dA/dZ: last layer; Multiply error by derivative of the activation function (sigmoidDerivative)
            deltas.back()[i] = error * sigmoidDerivative(activations.back()[i]);
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

                    // Determine the scalar influence of this layer to the next
                    influence += deltas[l + 1][j] * weights[l][i][j];
                }
                // Compute the delta dC/dZ for the current neuron in layer l: dC/dA * dA/dZ
                deltas[l][i] = influence * sigmoidDerivative(activations[l][i]);
            }
        }

        // Update weights and biases
        for (size_t l = 1; l < layers.size(); ++l) {
            for (int i = 0; i < layers[l]; ++i) {
                for (int j = 0; j < layers[l - 1]; ++j) {
                    // Nudge weight in the direction that decreases the cost function
                    weights[l - 1][j][i] -= learning_rate * deltas[l][i] * activations[l - 1][j];
                }
                // Nudge bias in the direction that decreases the cost function
                biases[l - 1][i] -= learning_rate * deltas[l][i];
            }
        }
    }

    // Train the network with multiple epochs
    void train(const std::vector<std::vector<double>>& inputs,
           const std::vector<std::vector<double>>& targets,
           int epochs, double learning_rate) {
        
        // Iterate over epochs (passes through dataset)
        for (int epoch = 0; epoch < epochs; ++epoch) {
            // To count the number of correct predictions
            int correct_predictions = 0; 

            // Iterate over training inputs
            for (size_t i = 0; i < inputs.size(); ++i) {
                forward(inputs[i]);
                backward(inputs[i], targets[i], learning_rate);

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
            if (epoch % 100 == 0) {
                std::cout << "Epoch " << epoch << ", Accuracy: " << accuracy << "%" << std::endl;
            }
        }
    }

    // Prediction
    std::vector<double> predict(const std::vector<double>& input) {
        return forward(input);
    }
};

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> generateXOR(int inputSize) {
    std::vector<std::vector<double>> X;
    std::vector<std::vector<double>> Y;

    int combinations = 1 << inputSize;  // 2^inputSize

    // Generate all possible combinations for `inputSize` inputs
    for (int i = 0; i < combinations; i++) {
        std::vector<double> input(inputSize);
        // Convert i into binary form, store it in the input vector
        for (int j = 0; j < inputSize; j++) {
            input[inputSize - 1 - j] = (i >> j) & 1;
        }

        // Calculate the XOR of all inputs (odd count of 1s gives 1, even count gives 0)
        double output = 0;
        for (int j = 0; j < inputSize; j++) {
            output += input[j];
        }
        output = (int)output % 2; // Output 1 if odd, 0 if even

        X.push_back(input);
        Y.push_back({output});
    }

    return std::make_pair(X, Y);  // Return both vectors wrapped in a pair
}

int main() {
    // Define the neural network architecture (e.g., 2 inputs, 2 hidden layers with 4 and 3 neurons, and 1 output)
    NeuralNetwork nn({6, 16, 16, 1});

    // XOR dataset
    auto data = generateXOR(6);
    std::vector<std::vector<double>> X = data.first;
    std::vector<std::vector<double>> Y = data.second;

    std::vector<double> t_in = X[4];
    std::vector<double> t_out = Y[4];
    X.erase(X.begin() + 4);
    Y.erase(Y.begin() + 4);

    // Train the neural network
    nn.train(X, Y, 2000, 3);

    // Test predictions
    for (const auto& input : X) {
        std::vector<double> output = nn.predict(input);
        std::cout << "Input: (";
        for(int i = 0; i < input.size(); i++) {
            std::cout << input[i];
            if(i != input.size() - 1)
                std::cout << ", ";
        }
        std::cout << ") -> Output: " << output[0] << std::endl;
    }

    std::cout << "\nUnqiue Tests:";
    std::cout << "Input: ";
    for(int i = 0; i < t_in.size(); i++) {
            std::cout << t_in[i];
            if(i != t_in.size() - 1)
                std::cout << ", ";
        }
    std::cout << " -> Output: " << nn.predict(t_in)[0] << " ans: " << t_out[0] << "\n";


    return 0;
}
