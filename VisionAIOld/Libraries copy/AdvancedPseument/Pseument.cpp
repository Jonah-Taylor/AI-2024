// Filename: Pseument.cpp
// Author: Jonah Taylor
// Date: November 27, 2024
// Description: Pseument (Pseudo Mantis) is my first neural network

#include "Pseument.hpp"

void NeuralNetwork::initializeAdamMiniStates() {

    m_weights.resize(weights.size());
    v_weights.resize(weights.size());
    m_biases.resize(biases.size());
    v_biases.resize(biases.size());

    for (size_t l = 0; l < layers.size() - 1; ++l) {
        // Resize matrices for moment weights
        m_weights[l] = MatrixXd::Zero(layers[l], layers[l + 1]);
        v_weights[l] = MatrixXd::Zero(layers[l], layers[l + 1]);
    
        // Resize vectors for moment biases
        m_biases[l] = VectorXd::Zero(biases[l].size());
        v_biases[l] = VectorXd::Zero(biases[l].size());
    }

}


NeuralNetwork::NeuralNetwork(const vector<int>& layers) : layers(layers) {

    // Initialize weight matrices and biases
    for (size_t i = 1; i < layers.size(); ++i) {
        // Create weight matrix for the layer: [future_neuron_count][previous_neuron_count]
        MatrixXd layer_weights(layers[i], layers[i - 1]);  // [future][previous]
        // Create bias vector for the layer: [future_neuron_count]
        VectorXd layer_biases(layers[i]);

        // Randomize weights using Eigen's Random() method to generate values between [-1, 1]
        layer_weights = MatrixXd::Random(layers[i], layers[i - 1]);

        // Biases can be initialized to zero or small random values
        layer_biases.setZero(); // Initialize biases to zero (or you could use Random() for small random values)

        // Add the weights and biases to the respective vectors
        weights.push_back(layer_weights);
        biases.push_back(layer_biases);
    }

    // Initialize Adam moments for weights and biases
    initializeAdamMiniStates();

    // Initialize activations and z-values (linear combinations) for each layer
    for (size_t i = 0; i < layers.size(); ++i) {
        activations.push_back(VectorXd::Zero(layers[i]));
        z_vals.push_back(VectorXd::Zero(layers[i]));
    }

}

VectorXd NeuralNetwork::forward(const VectorXd& input) {

    // Set the first layer of activations to the input values
    activations[0] = input;

    // Iterate through each layer starting from layer 1 (since layer 0 is input)
    for (size_t l = 1; l < layers.size(); ++l) {

        // Calculate z-values for the current layer
        z_vals[l] = weights[l - 1] * activations[l - 1] + biases[l - 1];

        // Apply the activation function (Leaky ReLU) to z-values
        activations[l] = leakyRelu(z_vals[l]);
    }

    // Return the output of the last layer
    return activations.back();

}


void NeuralNetwork::getOutputDeltas(vector<VectorXd>& deltas, const VectorXd& target, int& correct, unsigned int& loc) {

    // Ensure deltas has the correct size (output neurons)
    deltas.back() = VectorXd::Zero(layers.back()); // Zero-initialize the last layer deltas

    // Calculate output error (How far the prediction was off)
    VectorXd error = activations.back() - target; // Element-wise error calculation: activations - target

    // Calculate deltas using the derivative of the activation function (leaky ReLU)
    deltas.back() = error.array().cwiseProduct(leakyReluDerivative(activations.back()).array()); // Element-wise product


    // Accuracy Testing

    int largest_index;
    activations.back().maxCoeff(&largest_index);
    int ans;
    target.maxCoeff(&ans);
    // Update the correct count for accuracy
    if(target[largest_index] == 1) {
        correct++;
    }

    // Increment location and print progress every 1000 steps
    loc++;
    if (loc % 100 == 0) {
        std::cout << "AI: " << largest_index << " Correct: " << (bool)target(largest_index) << " --- Ans: " << ans << "\n";
    }

}

void NeuralNetwork::backward(vector<VectorXd>& deltas, vector<MatrixXd>& avg_grad_w, 
        vector<VectorXd>& avg_grad_b, double learning_rate) {

    // Backpropagate dC/dZ to hidden layers
    for (int l = layers.size() - 1; l > 0; --l) {

        // Initialize deltas, avg_grad_w, and avg_grad_b if not already initialized
        if (deltas[l - 1].size() != layers[l - 1]) {
            deltas[l - 1] = VectorXd::Zero(layers[l - 1]);
        }

        if (avg_grad_w[l].rows() != layers[l]) {
            avg_grad_w[l] = MatrixXd::Zero(layers[l], layers[l - 1]);
        }

        if (avg_grad_b[l].size() != layers[l]) {
            avg_grad_b[l] = VectorXd::Zero(layers[l]);
        }

        // Calculate error for previous layer
        deltas[l - 1] = (weights[l - 1].matrix().transpose() * deltas[l].matrix()).array() * leakyReluDerivative(z_vals[l - 1]).array();


        // Compute weight gradient dC/dW = dC/dZ * dZ/dW (element-wise for each neuron in the layer)
        avg_grad_w[l - 1] += deltas[l - 1].matrix() * activations[l].matrix().transpose();

        // Compute bias gradient dC/dB = dC/dZ for the entire layer (only once per layer)    
        avg_grad_b[l] += deltas[l];

    }
}

void NeuralNetwork::updateNeurons(std::vector<MatrixXd>& avg_grad_w, 
        std::vector<VectorXd>& avg_grad_b, double learning_rate, int batch_size) {
    
    //cout << "Updating Neurons\n";
    for (size_t l = 1; l < layers.size(); ++l) {
      
        // Weight gradients update
        MatrixXd grad_w = avg_grad_w[l - 1].array() / batch_size;  // Element-wise scaling of grad_w

        // Update weight moments
        m_weights[l-1] = (beta1 * m_weights[l-1] + (1 - beta1) * grad_w).matrix(); 
        v_weights[l-1] = beta2 * v_weights[l-1] + (1 - beta2) * grad_w.array().square().matrix(); // Element-wise square of grad_w

        // Bias gradients update
        VectorXd grad_b = avg_grad_b[l].array() / batch_size;  // Element-wise scaling of grad_b

        // Update bias moments
        m_biases[l-1] = (beta1 * m_biases[l-1] + (1 - beta1) * grad_b).matrix(); 
        v_biases[l-1] = beta2 * v_biases[l-1] + (1 - beta2) * grad_b.array().square().matrix(); 

        // Bias corrections
        MatrixXd m_hat_w = m_weights[l-1].array() / (1 - pow(beta1, t));
        MatrixXd v_hat_w = v_weights[l-1].array() / (1 - pow(beta2, t));

        MatrixXd m_hat_b = m_biases[l-1].array() / (1 - pow(beta1, t));
        MatrixXd v_hat_b = v_biases[l-1].array() / (1 - pow(beta2, t));

        // Update weights using Adam optimization
        weights[l-1] -= (learning_rate * (m_hat_w.array().transpose() / (v_hat_w.array().transpose().sqrt() + epsilon)).matrix() + lambda * weights[l-1]).matrix();

        // Update biases with Adam's correction term
        biases[l-1] -= (learning_rate * (m_hat_b.array() / (v_hat_b.array().sqrt() + epsilon)).matrix() + lambda * biases[l-1]).matrix();
    }
}



void NeuralNetwork::train(vector<vector<double>>& X, vector<vector<double>>& Y, 
        int epochs, int batch_size, double learning_rate, float& reward, bool print) {

    int numSamples = X.size();
    
    // Convert inputs to MatrixXd (e.g., 60000 x 784 for 28x28 images)
    MatrixXd inputs(numSamples, X[0].size());
    for (int i = 0; i < numSamples; ++i) {
        for (uint j = 0; j < X[i].size(); ++j) {
            inputs(i, j) = X[i][j];
        }
    }

    // Convert targets to MatrixXd (e.g., 60000 x 10 for one-hot encoded labels)
    MatrixXd targets(numSamples, Y[0].size());
    for (int i = 0; i < numSamples; ++i) {
        for (uint j = 0; j < Y[i].size(); ++j) {
            targets(i, j) = Y[i][j];
        }
    }

    // Item Order
    std::vector<int> shuffled(inputs.rows());
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

        // Iterate through the dataset in batches
        for (uint i = 0; i < inputs.rows(); i += batch_size) {

            // Initialize gradients for weights and biases
            std::vector<MatrixXd> avg_grad_w(layers.size());  // layers.size() - 1 because there are (n-1) weight matrices
            std::vector<VectorXd> avg_grad_b(layers.size());  // (n-1) bias vectors

            // Initialize gradients to zero
            for (uint l = 1; l < layers.size(); ++l) {  // Start from 1 because layer[0] is input layer
                avg_grad_w[l - 1] = MatrixXd::Zero(layers[l - 1], layers[l]);  // weights[l-1] connects layer[l-1] to layer[l]
                avg_grad_b[l - 1] = VectorXd::Zero(layers[l]);  // biases for each layer
            }

            // Process each example in the current batch
            for (int b = 0; b < batch_size; ++b) {

                // Forward pass: calculates activations
                forward(inputs.row(shuffled[i + b]));  // Puts results in last layer of activations

                // Calculate the output deltas (dC/dZ for the output layer)
                vector<VectorXd> deltas(layers.size()); // Initialize deltas for the output layer
                getOutputDeltas(deltas, targets.row(shuffled[i + b]), correct, loc);

                // Backward pass: calculates gradients for all layers
                backward(deltas, avg_grad_w, avg_grad_b, learning_rate);
            }

            // Update weights and biases using the calculated gradients
            updateNeurons(avg_grad_w, avg_grad_b, learning_rate, batch_size); 
        }

        // Print accuracy at every epoch
        if (print) {
            std::cout << "Epoch " << epoch + 1 << ": " << correct << " / " << loc << "\n";
        }
    }
}


void NeuralNetwork::save(const std::string& filename) {

    // Open/create file
    std::ofstream file(filename);
    if(!file) {
        std::cerr << "File couldn't be accessed for saving\n";
        return;
    }

    // Save layer sizes
    file << layers.size() << "\n";
    for (int& layer : layers) {
        file << layer << " ";
    }
    file << "\n\n";

    // Save weight data in file
    for (size_t i = 0; i < weights.size(); i++) {
        file << weights[i] << "\n\n"; // Saves each matrix in Eigen format (row-major)
    }

    // Save bias data in file
    for (size_t i = 0; i < biases.size(); i++) {
        file << biases[i].transpose() << "\n";  // Save biases as a row (transpose to save as row)
    }
    file << "\n";

    file.close();
}


void NeuralNetwork::load(const std::string& filename) {

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
        weights[i].resize(layers[i], layers[i + 1]);  // Resize matrix according to layers
        for (int row = 0; row < layers[i]; row++) {
            for (int col = 0; col < layers[i + 1]; col++) {
                file >> weights[i](row, col);  // Read individual element into matrix
            }
        }
    }

    // Load biases
    biases.resize(layers.size() - 1);
    for (size_t i = 0; i < biases.size(); i++) {
        biases[i].resize(layers[i + 1]);
        for (int j = 0; j < layers[i + 1]; j++) {
            file >> biases[i](j);  // Read individual element into vector
        }
    }

    file.close();
}

