// Filename: pseument.cpp
// Author: Jonah Taylor
// Date: December 20, 2024
// Description: Pseument (Pseudo Mantis) is my first neural network

#include "pseument.hpp"

NeuralNetwork::NeuralNetwork(const vector<int>& layer_sizes) {
    
    if(debugging) cout << "Started constructor\n";

    DenseLayer dl = DenseLayer(layer_sizes[0], layer_sizes[0]);
    layers.push_back(dl);
    for (size_t l = 1; l < layer_sizes.size(); ++l) {
        dl = DenseLayer(layer_sizes[l], layer_sizes[l - 1]);
        layers.push_back(dl);
    }

    if(debugging) cout << "Finished constructor\n";

}

vector<double> NeuralNetwork::forward(const vector<double>& input) {
    
    // Convert vector<double> to VectorXd
    VectorXd in(input.size());
    for (size_t i = 0; i < input.size(); ++i)
        in[i] = input[i];

    forward(in);

    vector<double> aws(layers.back().a.data(), layers.back().a.data() + layers.back().a.size());
    return aws;

}

VectorXd NeuralNetwork::forward(const VectorXd& in) {

    if(debugging) cout << "Started forward\n";

    layers[0].a = in;

    VectorXd out = in;
    for (size_t l = 1; l < layers.size(); ++l)
        out = layers[l].forward(out);

    if(debugging) cout << "Finished forward " << "\n";

    return out;

}

void NeuralNetwork::getOutputDeltas(const VectorXd& target) {
    

    if(debugging) cout << "Started getOutputDeltas\n";

    // Calculate deltas using the derivative of the activation function
    layers.back().getOutputDeltas(target);
    layers.back().updateGrads(layers[layers.size() - 2].a);

    if(debugging) cout << "Start accuracy testing\n";

    tested++;
    if(layers.back().a(0) > 0.5 ? 1 : 0 == target(0))
        correct++;

    if(debugging) cout << "Finished accuracy testing\n";

    if(debugging) cout << "Finished getOutputDeltas\n";
}

void NeuralNetwork::backward() {

    if(debugging) cout << "Started backward\n";

    for (uint l = layers.size() - 2; l > 0; --l) {
        
        layers[l].backward(layers[l + 1].w, layers[l + 1].dz);
        layers[l].updateGrads(layers[l - 1].a);

    }

    if(debugging) cout << "Finished backward\n";

}

void NeuralNetwork::stepSGD(double& lr, int& batch_size) {

    if(debugging) cout << "Started updateNeurons\n";
    
    for (size_t l = 1; l < layers.size(); ++l)
        layers[l].stepSGD(lr, batch_size);

     if(debugging) cout << "Finished updateNeurons\n";

}

void NeuralNetwork::stepAdamW(double& lr, int& batch_size, int& t) {

    if(debugging) cout << "Started updateNeurons\n";
    
    for (size_t l = 1; l < layers.size(); ++l)
        layers[l].stepAdamW(lr, batch_size, t);
    
     if(debugging) cout << "Finished updateNeurons\n";

}


void NeuralNetwork::train(vector<vector<double>>& X, vector<vector<double>>& Y, 
        int& epochs, int& batch_size, double& lr, bool print) {

    if(debugging) cout << "Started train\n";

    for(uint l = 1; l < layers.size(); ++l) {
        layers[l].lambda = lr / 10000; // Weight decay
    }

    uint numSamples = X.size();
    
    // Convert inputs to MatrixXd
    MatrixXd inputs(numSamples, X[0].size());
    for (uint i = 0; i < numSamples; ++i) {
        for (uint j = 0; j < X[i].size(); ++j) {
            inputs(i, j) = X[i][j];
        }
    }

    // Convert targets to MatrixXd
    MatrixXd targets(numSamples, Y[0].size());
    for (uint i = 0; i < numSamples; ++i) {
        for (uint j = 0; j < Y[i].size(); ++j) {
            targets(i, j) = Y[i][j];
        }
    }

    vector<int> shuffled(inputs.rows());
    iota(shuffled.begin(), shuffled.end(), 0);
    random_device rd;
    mt19937 g(rd());

    for (int epoch = 0; epoch < epochs; ++epoch) {

        shuffle(shuffled.begin(), shuffled.end(), g);
        t++;
        correct = 0;
        tested = 0;

        for (uint i = 0; i < inputs.rows(); i += batch_size) {

            for (int b = 0; b < batch_size; ++b) {

                forward(inputs.row(shuffled[i + b]));
                getOutputDeltas(targets.row(shuffled[i + b]));
                backward();

            }

            stepAdamW(lr, batch_size, t); 
        }

        if (print) cout << "Epoch " << epoch + 1 << ": " << correct << " / " << tested << "\n";
    }

    if(debugging) cout << "Finished train\n";

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
    for (uint l = 0; l < layers.size(); l++) {
        file << layers[l].layer_size << " ";
    }
    file << "\n";

    // Save weight data in file
    for (uint l = 1; l < layers.size(); l++) {
        file << layers[l].w << "\n\n"; // Saves each matrix in Eigen format (row-major)
    }

    // Save bias data in file 
    for (uint l = 1; l < layers.size(); l++) {
        file << layers[l].b.transpose() << "\n\n";  // Save biases as a row (transpose to save as row)
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
    uint layer_size;
    file >> layer_size;
    layers.resize(layer_size);
    for (uint l = 0; l < layer_size; l++) {
        if(l == 0) 
            layers[l] = DenseLayer(layer_size, 0);
        else 
            layers[l] = DenseLayer(layer_size, layers[l - 1].size());
    }

    // Load weights
    for (uint l = 1; l < layers.size(); l++) {
        for (uint row = 0; row < layers[l].size(); row++) {
            for (uint col = 0; col < layers[l - 1].size(); col++) {
                file >> layers[l].w(row, col);  // Read individual element into matrix
            }
        }
    }

    // Load biases
    for (uint l = 1; l < layers.size(); l++) {
        for (uint i = 0; i < layers[l].size(); i++) {
            file >> layers[l].b(i);  // Read individual element into vector
        }
    }

    file.close();
}

vector<uint> NeuralNetwork::getLayerSizes() {

    vector<uint> layer_sizes(layers.size());
    for(uint i = 0; i < layers.size(); i++)
        layer_sizes[i] = layers[i].size();
    return layer_sizes;

}


