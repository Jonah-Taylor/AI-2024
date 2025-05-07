// Filename: pseument.cpp
// Author: Jonah Taylor
// Date: December 20, 2024
// Description: Pseument (Pseudo Mantis) is my first neural network

#include "pseument.hpp"

NeuralNetwork::NeuralNetwork(const vector<MakeLayer>& l_info) {
    
    if(debugging) cout << "started constructor\n";

    d_layers.push_back(DenseLayer(l_info[0].l_size[0], l_info[0].l_size[0], l_info[0].a_func_name));
    
    for (size_t l = 1; l < l_info.size(); ++l) {
        switch(l_info[l].l_type) {
            case 0:
                d_layers.push_back(DenseLayer(l_info[l].l_size[0], l_info[l - 1].l_size[0], l_info[l].a_func_name));
                break;
            case 1:
                c_layers.push_back(ConvoLayer(l_info[l].l_size[0], l_info[l].l_size[1], l_info[l].a_func_name, 3));
                break;
            default:
                d_layers.push_back(DenseLayer(l_info[l].l_size[0], l_info[l - 1].l_size[0], l_info[l].a_func_name));
                break;
        }
    }

    if(debugging) cout << "finished constructor\n";

}

vector<double> NeuralNetwork::forward(const vector<double>& input) {

    if(debugging) cout << "Started forward\n";
    
    // convert vector<double> to vectorxd
    VectorXd in(input.size());
    for (size_t i = 0; i < input.size(); ++i)
        in[i] = input[i];

    forward(in);

    vector<double> aws(d_layers.back().a.data(), d_layers.back().a.data() + d_layers.back().a.size());

    if(debugging) cout << "Finished forward\n";

    return aws;

}

VectorXd NeuralNetwork::forward(const VectorXd& in) {

    if(debugging) cout << "Started forward\n";

    d_layers[0].a = in;

    VectorXd out = in;
    for (size_t l = 1; l < d_layers.size(); ++l)
        out = d_layers[l].forward(out);

    if(debugging) cout << "Finished forward " << "\n";

    return out;

}

void NeuralNetwork::getOutputDeltas(const VectorXd& target) {
    

    if(debugging) cout << "Started getOutputDeltas\n";

    d_layers.back().getOutputDeltas(target);
    d_layers.back().updateGrads(d_layers[d_layers.size() - 2].a);

    if(debugging) cout << "Start accuracy testing\n";

    tested++;
    if(d_layers.back().a(0) > 0.5 ? 1 : 0 == target(0))
        correct++;

    if(debugging) cout << "Finished accuracy testing\n";

    if(debugging) cout << "Finished getOutputDeltas\n";
    
}

void NeuralNetwork::backward() {

    if(debugging) cout << "Started backward\n";

    for (uint l = d_layers.size() - 2; l > 0; --l) {
        
        d_layers[l].backward(d_layers[l + 1].w, d_layers[l + 1].dz);
        d_layers[l].updateGrads(d_layers[l - 1].a);

    }

    if(debugging) cout << "Finished backward\n";

}

void NeuralNetwork::stepSGD(double& lr, int& batch_size) {

    if(debugging) cout << "Started updateNeurons\n";
    
    for (size_t l = 1; l < d_layers.size(); ++l)
        d_layers[l].stepSGD(lr, batch_size);

     if(debugging) cout << "Finished updateNeurons\n";

}

void NeuralNetwork::stepAdamW(double& lr, int& batch_size, int& t) {

    if(debugging) cout << "Started updateNeurons\n";
    
    for (size_t l = 1; l < d_layers.size(); ++l)
        d_layers[l].stepAdamW(lr, batch_size, t);
    
     if(debugging) cout << "Finished updateNeurons\n";

}


void NeuralNetwork::train(vector<vector<double>>& X, vector<vector<double>>& Y, 
        int& epochs, int& batch_size, double& lr, string da, bool print) {

    if(debugging) cout << "Started train\n";

    if(da == "sgd")
        descent = 0;
    else if(da == "adamw")
        descent = 1;
    else
        descent = 0;

    for(uint l = 1; l < d_layers.size(); ++l) {
        d_layers[l].lambda = lr / 100; // Weight decay
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
            switch(descent) {
                case 0:
                    stepSGD(lr, batch_size);
                    break;
                case 1:
                    stepAdamW(lr, batch_size, t);
                    break;
            }
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
    file << d_layers.size() << "\n";
    for (uint l = 0; l < d_layers.size(); l++) {
        file << d_layers[l].size() << " ";
    }
    file << "\n";
    
    // Save layer activation function
    for (uint l = 0; l < d_layers.size(); l++) {
        file << d_layers[l].a_func_name << " ";
    }
    file << "\n\n";
    
    // Save weight data in file
    for (uint l = 1; l < d_layers.size(); l++) {
        file << d_layers[l].w << "\n\n"; // Saves each matrix in Eigen format (row-major)
    }

    // Save bias data in file 
    for (uint l = 1; l < d_layers.size(); l++) {
        file << d_layers[l].b.transpose() << "\n\n";  // Save biases as a row (transpose to save as row)
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

    // Load number of layers
    uint layer_count;
    file >> layer_count;
    d_layers.resize(layer_count);
    
    // Load layer sizes
    uint l_size;
    for (uint l = 0; l < layer_count; l++) {
        file >> l_size;
        if(l == 0) 
            d_layers[l] = DenseLayer(l_size, 0, 0);
        else 
            d_layers[l] = DenseLayer(l_size, d_layers[l - 1].size(), 0);
    }

    // Load layer activations
    string afn;
    for (uint l = 0; l < layer_count; l++) {
        file >> afn;
        d_layers[l].setActFunc(afn);
    }

    // Load weights
    for (uint l = 1; l < d_layers.size(); l++) {
        for (uint row = 0; row < d_layers[l].size(); row++) {
            for (uint col = 0; col < d_layers[l - 1].size(); col++) {
                file >> d_layers[l].w(row, col);  // Read individual element into matrix
            }
        }
    }

    // Load biases
    for (uint l = 1; l < d_layers.size(); l++) {
        for (uint i = 0; i < d_layers[l].size(); i++) {
            file >> d_layers[l].b(i);  // Read individual element into vector
        }
    }

    file.close();

}

vector<uint> NeuralNetwork::getLayerSizes() {

    vector<uint> layer_sizes(d_layers.size());
    for(uint i = 0; i < d_layers.size(); i++)
        layer_sizes[i] = d_layers[i].size();
    return layer_sizes;

}


