// Filename: Pseument.cpp
// Author: Jonah Taylor
// Date: November 27, 2024
// Description: Pseument (Pseudo Mantis) is my first neural network

#include "pseument.hpp"

void NeuralNetwork::initializeAdamMiniStates() {

    m_weights.resize(weights.size());
    v_weights.resize(weights.size());
    m_biases.resize(biases.size());
    v_biases.resize(biases.size());

    for (size_t l = 1; l < layers.size(); ++l) {
        // Resize matrices for moment weights
        m_weights[l] = MatrixXd::Zero(layers[l], layers[l - 1]);
        v_weights[l] = MatrixXd::Zero(layers[l], layers[l - 1]);
    
        // Resize vectors for moment biases
        m_biases[l] = VectorXd::Zero(biases[l].size());
        v_biases[l] = VectorXd::Zero(biases[l].size());
    }

}


NeuralNetwork::NeuralNetwork(const vector<int>& layers) : layers(layers) {
    
    if(debugging)
        cout << "Started constructor\n";

    weights.resize(layers.size());  // Resize weights to the size of layers
    biases.resize(layers.size());   // Resize biases to the size of layers

    // Initialize weight matrices and biases
    for (size_t l = 1; l < layers.size(); ++l) {

        // Create weight matrix for the layer: [future_neuron_count][previous_neuron_count]
        MatrixXd layer_weights(layers[l], layers[l - 1]);  // [future][previous]
        // Create bias vector for the layer: [future_neuron_count]
        VectorXd layer_biases(layers[l]);

        // He Initialization: Randomize weights using Eigen's Random() method to generate values between [-1, 1]
        layer_weights = MatrixXd::Random(layers[l], layers[l - 1]) * sqrt(2.0 / layers[l - 1]);
        // Initialize biases to zero (or you could use Random() for small random values)
        layer_biases.setZero();

       
        weights[l] = layer_weights; // Store at layer index l
        biases[l] = layer_biases;   // Store at layer index l
        
    }

    // Initialize Adam moments for weights and biases
    initializeAdamMiniStates();

    // Initialize activations and z-values (linear combinations) for each layer
    for (size_t l = 0; l < layers.size(); ++l) {
        activations.push_back(VectorXd::Zero(layers[l]));
        z_vals.push_back(VectorXd::Zero(layers[l]));
    }

    if(debugging)
        cout << "Finished constructor\n";

}

int NeuralNetwork::forward(const vector<double>& input) {
    
    if (input.size() != (uint)layers[0]) {
        return -1;
    }

    // Convert vector<double> to VectorXd
    VectorXd eigen_input(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        eigen_input[i] = input[i];
    }

    forward(eigen_input);

    int largest_output;
    activations.back().maxCoeff(&largest_output);
    return largest_output;
}

VectorXd NeuralNetwork::forward(const VectorXd& input) {

    if(debugging)
        cout << "Started forward\n";

    // Set the first layer of activations to the input values
    activations[0] = input;

    // Iterate through each layer starting from layer 1 (since layer 0 is input)
    for (size_t l = 1; l < layers.size(); ++l) {

        // Calculate z-values for the current layer
        z_vals[l] = weights[l] * activations[l - 1] + biases[l];

        // Apply the activation function to z-values
        activations[l] = leakyRelu(z_vals[l]);
    }

    if(debugging)
        cout << "Finished forward " << "\n";

    // Return the output of the last layer
    return activations.back();

}

void NeuralNetwork::getOutputDeltas(vector<VectorXd>& deltas, const VectorXd& target) {
    

    if(debugging)
        cout << "Started getOutputDeltas\n";

    // Ensure deltas has the correct size (output neurons)
    deltas.back() = VectorXd::Zero(layers.back()); // Zero-initialize the last layer deltas

    // Calculate output error (How far the prediction was off)
    VectorXd error = activations.back() - target; // Element-wise error calculation: activations - target
 
    // Calculate deltas using the derivative of the activation function
    deltas.back() = error.array().cwiseProduct(leakyReluDerivative(z_vals.back()).array()); // Element-wise product

    if(debugging)
        cout << "Start accuracy testing\n";
    // Accuracy Testing

    int largest_index;
    activations.back().maxCoeff(&largest_index);
    int ans;
    target.maxCoeff(&ans);
    // Update the correct count for accuracy
    if(largest_index == ans) {
        correct++;
    }

    // Increment location and print progress every 1000 steps
    
    // if (tested % 5000 == 0) {
    //     cout << "AI: " << largest_index << " Correct: " << (bool)target(largest_index) 
    //     << " --- Ans: " << ans << "  ---- " << weights[1](0, 0) << "\n";
    // }
    tested++;

    if(debugging)
        cout << "Finished accuracy testing\n";

    if(debugging)
        cout << "Finished getOutputDeltas\n";
}

void NeuralNetwork::backward(vector<VectorXd>& deltas, vector<MatrixXd>& avg_grad_w, 
        vector<VectorXd>& avg_grad_b, double learning_rate) {

    if(debugging)
        cout << "Started backward\n";

    

    // Backpropagate dC/dZ to hidden layers
    for (uint l = layers.size() - 1; l > 0; --l) {

        if (avg_grad_w[l].rows() != layers[l]) {
            avg_grad_w[l] = MatrixXd::Zero(layers[l], layers[l - 1]);
        }

        if (avg_grad_b[l].size() != layers[l]) {
            avg_grad_b[l] = VectorXd::Zero(layers[l]);
        }
        
        if(l != layers.size() - 1) {
            // Initialize deltas, avg_grad_w, and avg_grad_b if not already initialized
            if (deltas[l - 1].size() != layers[l - 1]) {
                deltas[l - 1] = VectorXd::Zero(layers[l - 1]);
            }

            // Calculate error for previous layer
            deltas[l] = (weights[l + 1].matrix().transpose() * deltas[l + 1].matrix()).array() * leakyReluDerivative(z_vals[l]).array();

        }

        // Compute weight gradient dC/dW = dC/dZ * dZ/dW (element-wise for each neuron in the layer)
        avg_grad_w[l] += deltas[l].matrix() * activations[l - 1].matrix().transpose();

        // Compute bias gradient dC/dB = dC/dZ for the entire layer (only once per layer)    
        avg_grad_b[l] += deltas[l];

    }

    if(debugging)
        cout << "Finished backward\n";

}

void NeuralNetwork::stepSGD(vector<MatrixXd>& avg_grad_w, 
        vector<VectorXd>& avg_grad_b, double learning_rate, int batch_size) {

    if(debugging)
        cout << "Started updateNeurons\n";
    
    //cout << "Updating Neurons\n";
    for (size_t l = 1; l < layers.size(); ++l) {
 
        // Weight gradients update (No batch size scaling or momenta in SGD)
        MatrixXd grad_w = avg_grad_w[l] / batch_size;  // Average gradient for weights

        // Update weights with SGD
        weights[l] -= learning_rate * grad_w;  // L2 regularization term

        // Bias gradients update (No batch size scaling or momenta in SGD)
        VectorXd grad_b = avg_grad_b[l] / batch_size;  // Average gradient for biases
        
        // Update biases with SGD
        biases[l] -= learning_rate * grad_b;  // L2 regularization term
    }

     if(debugging)
        cout << "Finished updateNeurons\n";

}

void NeuralNetwork::stepAdamW(vector<MatrixXd>& avg_grad_w, 
        vector<VectorXd>& avg_grad_b, double learning_rate, int batch_size) {

    if(debugging)
        cout << "Started updateNeurons\n";
    
    //cout << "Updating Neurons\n";
    for (size_t l = 1; l < layers.size(); ++l) {
      
        // Weight gradients update
        MatrixXd grad_w = avg_grad_w[l].array() / batch_size;  // Element-wise scaling of grad_w

        //cout << "D1\n";

        // Update weight moments
        m_weights[l] = (beta1 * m_weights[l] + (1 - beta1) * grad_w).matrix(); 
        v_weights[l] = beta2 * v_weights[l] + (1 - beta2) * grad_w.array().square().matrix(); // Element-wise square of grad_w

        // Bias gradients update
        VectorXd grad_b = avg_grad_b[l].array() / batch_size;  // Element-wise scaling of grad_b

        //cout << "D2\n";

        // Update bias moments
        m_biases[l] = (beta1 * m_biases[l] + (1 - beta1) * grad_b).matrix(); 
        v_biases[l] = beta2 * v_biases[l] + (1 - beta2) * grad_b.array().square().matrix(); 

        //cout << "D3\n";

        // Bias corrections
        MatrixXd m_hat_w = m_weights[l].array() / (1 - pow(beta1, t));
        MatrixXd v_hat_w = v_weights[l].array() / (1 - pow(beta2, t));

        MatrixXd m_hat_b = m_biases[l].array() / (1 - pow(beta1, t));
        MatrixXd v_hat_b = v_biases[l].array() / (1 - pow(beta2, t));

        // cout << "D4\n";

        // cout << "Dimensions: " << weights[l].rows() << " x " << weights[l].cols() << "\n";
        // cout << "Dimensions: " << m_hat_w.array().rows() << " x " << m_hat_w.array().cols() << "\n";

        // Update weights using Adam optimization
        weights[l] -= (learning_rate * (m_hat_w.array() / (v_hat_w.array().sqrt() + epsilon)).matrix() + lambda * weights[l]).matrix();

        //cout << "D5\n";

        // Update biases with Adam's correction term
        biases[l] -= (learning_rate * (m_hat_b.array() / (v_hat_b.array().sqrt() + epsilon)).matrix() + lambda * biases[l]).matrix();
    }

     if(debugging)
        cout << "Finished updateNeurons\n";

}


void NeuralNetwork::train(vector<vector<double>>& X, vector<vector<double>>& Y, 
        int& epochs, int& batch_size, double& learning_rate, bool print) {

    if(debugging)
        cout << "Started train\n";

    lambda = learning_rate / 10; // Weight decay to prevent overfitting

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
    vector<int> shuffled(inputs.rows());
    iota(shuffled.begin(), shuffled.end(), 0);

    random_device rd;
    mt19937 g(rd());

    // Iterate over epochs (passes through dataset)
    for (int epoch = 0; epoch < epochs; ++epoch) {

        // Shuffle Order
        shuffle(shuffled.begin(), shuffled.end(), g);

        // To count the number of correct predictions
        t++;
        correct = 0;
        tested = 0;

        // Iterate through the dataset in batches
        for (uint i = 0; i < inputs.rows(); i += batch_size) {

            // Initialize gradients for weights and biases
            vector<MatrixXd> avg_grad_w(layers.size());  // layers.size() - 1 because there are (n-1) weight matrices
            vector<VectorXd> avg_grad_b(layers.size());  // (n-1) bias vectors

            // Process each example in the current batch
            for (int b = 0; b < batch_size; ++b) {
                vector<VectorXd> deltas(layers.size()); // Initialize deltas for the output layer

                // Forward pass: calculates activations
                forward(inputs.row(shuffled[i + b]));  // Puts results in last layer of activations

                // Calculate the output deltas (dC/dZ for the output layer)
                getOutputDeltas(deltas, targets.row(shuffled[i + b]));

                // Backward pass: calculates gradients for all layers
                backward(deltas, avg_grad_w, avg_grad_b, learning_rate);
            }

            // Update weights and biases using the calculated gradients
            stepAdamW(avg_grad_w, avg_grad_b, learning_rate, batch_size); 
        }

        // Print accuracy at every epoch
        if (print) {
            cout << "Epoch " << epoch + 1 << ": " << correct << " / " << tested << "\n";
        }

        // cout << "First Hidden Weight Connections: \n";
        // for(int i = 0; i < layers[0]; i++) {
        //     cout << weights[0](i, 0) << "\n";
        // }
    }

    if(debugging)
        cout << "Finished train\n";

}


void NeuralNetwork::save(const string& filename) {

    // Open/create file
    ofstream file(filename);
    if(!file) {
        cerr << "File couldn't be accessed for saving\n";
        return;
    }

    // Save layer sizes
    file << layers.size() << " ";
    for (int& layer : layers) {
        file << layer << " ";
    }
    file << "\n";

    // Save weight data in file
    for (size_t i = 0; i < weights.size(); i++) {
        file << weights[i] << "\n\n"; // Saves each matrix in Eigen format (row-major)
    }

    // Save bias data in file
    for (size_t i = 0; i < biases.size(); i++) {
        file << biases[i].transpose() << "\n\n";  // Save biases as a row (transpose to save as row)
    }
    file << "\n";

    file.close();
}


void NeuralNetwork::load(const string& filename) {

    // Open File
    ifstream file(filename);
    if (!file) {
        cerr << "File couldn't be accessed for loading\n";
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
    weights.resize(layers.size());
    for (size_t l = 1; l < weights.size(); l++) {
        weights[l].resize(layers[l], layers[l - 1]);  // Resize matrix according to layers
        for (int row = 0; row < layers[l]; row++) {
            for (int col = 0; col < layers[l - 1]; col++) {
                file >> weights[l](row, col);  // Read individual element into matrix
            }
        }
    }

    // Load biases
    biases.resize(layers.size());
    for (size_t l = 1; l < biases.size(); l++) {
        biases[l].resize(layers[l]);
        for (int i = 0; i < layers[l]; i++) {
            file >> biases[l](i);  // Read individual element into vector
        }
    }

    file.close();
}


