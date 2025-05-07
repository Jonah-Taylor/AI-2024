// Filename: Pseument.cpp
// Author: Jonah Taylor
// Date: November 16, 2024
// Description: Pseument (Pseudo Mantis) is my first neural network

#include "Pseument.hpp"

void NeuralNetwork::initializeAdamMiniStates() {
    m_weights.resize(weights.size());
    v_weights.resize(weights.size());
    m_biases.resize(biases.size());
    v_biases.resize(biases.size());

    for (size_t l = 0; l < weights.size(); ++l) {
        // moment_weights = moment_weights[layerSize][neurons[l]][neurons[l+1]]
        m_weights[l].resize(weights[l].size(), std::vector<double>(weights[l][0].size(), 0.0));
        v_weights[l].resize(weights[l].size(), std::vector<double>(weights[l][0].size(), 0.0));
    }

    for (size_t l = 0; l < biases.size(); ++l) {
        // moment_biases = moment_biases[layerSize][neurons[l]]
        m_biases[l].resize(biases[l].size(), 0.0);
        v_biases[l].resize(biases[l].size(), 0.0);
    }
}

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

    // Initialize Adam moments for weights and biases
    initializeAdamMiniStates();

    // Initialize activations and z-values (linear combinations) for each layer
    for (size_t i = 0; i < layers.size(); ++i) {
        activations.push_back(std::vector<double>(layers[i]));
        z_vals.push_back(std::vector<double>(layers[i]));
    }
    //std::cout << weights[1][1][1] << " w2\n";
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
                //std::cout << "weights: " << weights[l - 1][k][j] << " " << weights[1][1][1] << "\n";
            }
            // Plug result into sigmoid to make the activation between -1 and 1
            activations[l][j] = leakyRelu(z_vals[l][j]);
        }
    }

    //std::cout << "Act back: " << activations.back()[0] << "\n";
    // Return the output of the last layer of neurons
    return activations.back();
}

void NeuralNetwork::getOutputDeltas(std::vector<std::vector<double>>& deltas,
        const std::vector<std::vector<double>>& inputs, const std::vector<double>& target, 
        int& batch_size, float& reward, int& correct, uint& loc) {


    // Fill back layer with output neurons
    deltas.back() = std::vector<double>(layers.back());

    int largest_index = 0;
    int largest_value = activations.back()[0];

    // Calculate output error (How far the prediction was off)
    for (int i = 0; i < layers.back(); ++i) {   // Iterate over output neurons

        // For accuracy print in train method
        if(activations.back()[i] > largest_value) {
            largest_value = activations.back()[i];
            largest_index = i;
        }

        // dC/dA of last layer using Mean Squared Error (MSE)
        double error = activations.back()[i] - target[i] + reward;

        // dC/dZ = dC/dA * dA/dZ: last layer; Multiply error by derivative of the activation function
        deltas.back()[i] = error * leakyReluDerivative(activations.back()[i]);
        //std::cout << activations.back().size() << " " << activations.back()[i] << " " << error << " " << deltas.back()[i] << "\n";
    }

    if(target[largest_index] == 1)
        correct++;

    loc += 1;
    if(loc % 1000 == 0)
        std::cout << "AI: " << largest_index << " Correct: " << (bool)target[largest_index] << "\n";
}

// Backward pass for gradient descent

void NeuralNetwork::backward(std::vector<std::vector<double>>& deltas, 
        std::vector<std::vector<std::vector<double>>>& avg_grad_w, 
        std::vector<std::vector<double>>& avg_grad_b, double& learning_rate) {


    // Backpropagate dC/dZ to hidden layers
    for (int l = layers.size() - 1; l > 0; --l) {
        //std::cout << l << "\n";

        // Fill layer with correct number of neurons
        if (deltas[l - 1].size() != (uint)layers[l - 1]) {
            deltas[l - 1] = std::vector<double>(layers[l - 1]);
        }

        if (avg_grad_w[l].size() != (uint)layers[l - 1]) {
            avg_grad_w[l] = std::vector<std::vector<double>>(layers[l - 1], std::vector<double>(layers[l], 0.0));
        }

        if (avg_grad_b[l].size() != (uint)layers[l]) {
            avg_grad_b[l] = std::vector<double>(layers[l], 0.0);
        }
        
        // Iterate over neurons in layer
        for (int j = 0; j < layers[l - 1]; ++j) {
            double influence = 0.0;

            // Iterate over neurons in following layer
            for (int i = 0; i < layers[l]; ++i) {

                // Determine the influence of this layer to the next: dC/dA
                influence += deltas[l][i] * weights[l - 1][j][i];
                
                // Compute the weight gradient: dC/dW = dC/dZ * dZ/dW
                avg_grad_w[l][j][i] += deltas[l][i] * activations[l - 1][j];

                if(j == 0) { // Compute Once
                    // Compute the bias gradient: dC/dB = dC/dZ
                    avg_grad_b[l][i] += deltas[l][i];
                }
            
            }
            // Compute the delta dC/dZ for the current neuron in layer l: dC/dA * dA/dZ
            deltas[l - 1][j] = influence * leakyReluDerivative(activations[l][j]);
            
        }
    }
}

void NeuralNetwork::updateNeurons(std::vector<std::vector<std::vector<double>>>& avg_grad_w, 
        std::vector<std::vector<double>>& avg_grad_b, double& learning_rate, int& batch_size) {
    // Update weights and biases
    for (size_t l = 1; l < layers.size(); ++l) {
        for (int i = 0; i < layers[l]; ++i) {
            for (int j = 0; j < layers[l - 1]; ++j) {

                // weight_gradient = dC/dZ * dZ/dW(ij) = dC/dW(ij)
                double grad_w = avg_grad_w[l][j][i] * learning_rate / batch_size;

                // Update weight moments
                m_weights[l-1][j][i] = beta1 * m_weights[l - 1][j][i] + (1 - beta1) * grad_w; 
                v_weights[l-1][j][i] = beta2 * v_weights[l - 1][j][i] + (1 - beta2) * grad_w * grad_w; 
                
                // Bias corrections
                double m_hat_w = m_weights[l - 1][j][i] / (1 - pow(beta1, t));
                double v_hat_w = v_weights[l - 1][j][i] / (1 - pow(beta2, t));

                // Update weight values with velocity in direction decreasing cost function
                weights[l - 1][j][i] -= learning_rate * (m_hat_w / (sqrt(v_hat_w) + epsilon) + lambda * weights[l - 1][j][i]);

            }
            // Nudge bias in the direction that decreases the cost function
            // deltas[l][i] * 1 = dC/dZ * dZ/dB(i) = dC/dB(i)
            double grad_b = avg_grad_b[l][i] * learning_rate / batch_size;

            // Update velocity of biases
            // Update weight moments
            m_biases[l-1][i] = beta1 * m_biases[l - 1][i] + (1 - beta1) * grad_b; 
            v_biases[l-1][i] = beta2 * v_biases[l - 1][i] + (1 - beta2) * grad_b * grad_b; 

            double m_hat_b = m_biases[l - 1][i] / (1 - pow(beta1, t));
            double v_hat_b = v_biases[l - 1][i] / (1 - pow(beta2, t));

            // Update bias values with velocity in direction decreasing the cost function
            biases[l - 1][i] -= learning_rate * (m_hat_b / (sqrt(v_hat_b) + epsilon) + lambda * biases[l - 1][i]);
        }
    }
}


// Train the network with multiple epochs
void NeuralNetwork::train(const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& targets,
        int& epochs, int& batch_size, double& learning_rate, float& reward,
        bool& print) {

    // Item Order
    std::vector<int> shuffled(inputs.size());
    std::iota(shuffled.begin(), shuffled.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());

    // Iterate over epochs (passes through dataset)
    for (int epoch = 0; epoch < epochs; ++epoch) {

        // Shuffle Order
        std::shuffle(shuffled.begin(), shuffled.end(), g);

        // To count the number of correct predictions
        int correct = 0;
        uint loc = 0;
        t++;

        for(uint i = 0; i < inputs.size(); i += batch_size) {

            // Create layer structure to determine output sensitivity to each neuron
            // Initialize gradients for weights and biases
            std::vector<std::vector<std::vector<double>>> avg_grad_w(layers.size());
            std::vector<std::vector<double>> avg_grad_b(layers.size());

            // Initialize gradients to zero
            for (uint l = 0; l < layers.size(); ++l) {
                avg_grad_w[l] = std::vector<std::vector<double>>(layers[l]);
                avg_grad_b[l] = std::vector<double>(layers[l], 0.0);
            }
            
            for(int b = 0; b < batch_size; b++) {
                std::vector<std::vector<double>> deltas(layers.size());
            
                forward(inputs[shuffled[i]]); // Puts results in last layer of activations
                getOutputDeltas(deltas, inputs, targets[shuffled[i]], batch_size, reward, correct, loc); // dC/dZ calculated for last layer
                backward(deltas, avg_grad_w, avg_grad_b, learning_rate); // dC/dZ calculated for all layers
        
            }           

            updateNeurons(avg_grad_w, avg_grad_b, learning_rate, batch_size); // Weights and Biases updated with deltas and optimizer
        }
        // Calculate the accuracy of AI predicting values in epoch
        //double accuracy = (double)correct / (loc) * 100.0;
        //std::cout << correct << " " << loc << "\n";
        // Print accuracy every epochs
        if (print) {
            std::cout << "Epoch " << epoch + 1 << ": " << correct << " / " << loc << "\n";
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
