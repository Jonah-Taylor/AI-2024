// Filename: VisionAI.cpp
// Author: Jonah Taylor
// Date: November 25, 2024
// Description: A Neural Network that sees pixels and identifies numbers

#include <fstream>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <deque>

#include "Libraries/Pseument/Pseument.hpp"
#include "Libraries/SMFL/Graphics.hpp"
#include "Libraries/SMFL/Window.hpp"
#include "Libraries/SMFL/System.hpp"
#include "Libraries/Eigen/Dense"

using namespace Eigen;
using namespace std;

const int WINDOW_WIDTH = 1120;
const int WINDOW_HEIGHT = 1120;
const float squareWidth = 40;

int incorrect = 0;
int correct = 0;
int correctRateLast100 = 0;
std::deque<int> last100(100, 0);
int detectFPS = 0; // Frames Per Second
float FPS = 60;

bool A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, Z = false; // Save X and Y for Inputs and Predictions
bool training = false;
bool fast = false;
bool display = true;
bool printEpochs = true;

std::vector<std::vector<double>> images;
std::vector<double> labels;
int trainingSamples = 60000;
int trainedSize = 0;

NeuralNetwork nn({784, 20, 10});
std::vector<double> inputs;
std::vector<std::vector<double>> X;
std::vector<std::vector<double>> Y;
int epochs = 1000;
int batchSize = 20;
double trainingSpeed = 0.01;

void keyBoardInputs();
std::vector<std::vector<double>> getMnistImages(const std::string& file_path, int num_images);
std::vector<double> getMnistLabels(const std::string& file_path, int num_labels);
int getNum(std::vector<double> outputs);

int main(int argc, char* argv[]) {
    
    // Load network if used as argument
    if (argc > 1) {
        std::string filename = argv[1];
        std::cout << "Loading network: " << filename << "\n";
        nn.load("Saved/" + filename + ".txt");
    }

    sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "");
    window.setTitle("Vision " + std::to_string(incorrect) + " to " + std::to_string(correct) + "  |  AI WR: " + 
                    std::to_string(correctRateLast100) + " / 100  |  detectFPS = " + std::to_string(detectFPS));

    // sf::Image icon;
    // if (!icon.loadFromFile("Images/icon.png")) {
    //     std::cerr << "Error loading icon\n";
    // }
    // else {
    //     window.setIcon(icon.getSize().x, icon.getSize().y, icon.getPixelsPtr());
    // }

    // Create random seed
    std::srand(std::time(0));

    // Get Mnist Data
    images = getMnistImages("MnistData/train-images.idx3-ubyte", trainingSamples);
    labels = getMnistLabels("MnistData/train-labels.idx1-ubyte", trainingSamples);
    std::cout << labels[0] << "\n";
    for(int i = 0; i < 28; i++) {
        for(int j = 0; j < 28; j++) {
            std::cout << round(images[0][28 * i + j]);
        }
        std::cout << "\n";
    }





    for(int i = 0; i < 60000; i++) {
        X.push_back(images[i]);
        Y.push_back({0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
        Y[Y.size() - 1][labels[i]] = 1;
        trainedSize++;
    }

    float reward = 0;

    nn.train(X, Y, epochs, batchSize, trainingSpeed, reward, printEpochs);









    
    // Set up clock for frame timing
    sf::Clock clock;
    const sf::Time targetTime = sf::seconds(1.f / FPS);

    // Main game loop
    while (window.isOpen()) {
        sf::Time elapsed = clock.getElapsedTime();
 
        // Process frame every 1/FPS of a second or, if training, as fast as possible
        if (elapsed >= targetTime || fast) {
            detectFPS = 1 / (elapsed.asSeconds());
            clock.restart();

            // Close window when exiting the program
            sf::Event event;
            while (window.pollEvent(event)) {
                if (event.type == sf::Event::Closed) {
                    // Auto Save
                    std::cout << "Saving network: " << "exit" << std::to_string(correctRateLast100) << "_" << std::to_string(int(epochs)) << ".txt" << "\n";
                    nn.save("Saved/exit" + std::to_string(correctRateLast100) + "_" + std::to_string(int(epochs)) + ".txt");
                    window.close();
                }
            }
            
            keyBoardInputs();

            // Render objects to screen
            if(display) {
                window.clear(sf::Color::Black);
                window.display();
            }
        }
    }

    return 0;
};

void keyBoardInputs() {
    

    // Toggle Display for Faster Training
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::D)) {
        if(!D)
            display = !display;
        D = true;
    }
    else {
        D = false;
    }

    // Toggle Epoch Feedback
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::E)) {
        if(!E)
            printEpochs = !printEpochs;
        E = true;
    }
    else {
        E = false;
    }

    // Toggle Fast Mode
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::F)) {
        if(!F)
            fast = !fast;
        F = true;
    }
    else {
        F = false;
    }

    // Print Network Information
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::N)) {
        if(!N) {
            std::cout << "\nName: " << "nw" << std::to_string(correctRateLast100) << "_" << std::to_string(int(epochs)) << ".txt" << "\n";
            //std::vector<int> layers = nn.getLayers();
            //std::cout << "Layer Count: " << layers.size() << "\n";
            // std::cout << "Layers: ";
            // for(int layer : layers) {
            //     std::cout << layer << " ";
            // }
            std::cout << "\n\n";
        }
        N = true;
    }
    else {
        N = false;
    }

    // Save Network
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) {
        if(!S) {
            std::cout << "Saving network: " << "nw" << std::to_string(correctRateLast100) << "_" << std::to_string(int(epochs)) << ".txt" << "\n";
            nn.save("Saved/nw" + std::to_string(correctRateLast100) + "_" + std::to_string(int(epochs)) + ".txt");
        }
        S = true;
    }
    else {
        S = false;
    }

    // Toggle Training Mode
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::T)) {
        if(!T)
            training = !training;
        if(training)
            std::cout << "Training!\n";
        else {
            fast = false;
        }
        T = true;
    }
    else {
        T = false;
    }
}

std::vector<std::vector<double>> getMnistImages(const std::string& file_path, int num_images) {
    const int image_size = 28 * 28; // Each image is 28x28 pixels
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + file_path);

    // Read the header
    file.ignore(16); // Skip the 16-byte header

    // Prepare a container for the images
    std::vector<std::vector<double>> images(num_images, std::vector<double>(image_size));

    // Read each image
    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < image_size; ++j) {
            unsigned char pixel;
            file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
            images[i][j] = pixel / 255.0f; // Normalize pixel values to [0, 1]
        }
    }
    file.close();
    return images;
}

std::vector<double> getMnistLabels(const std::string& file_path, int num_labels) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + file_path);

    // Read the header
    file.ignore(8); // Skip the 8-byte header

    // Prepare a container for the labels
    std::vector<double> labels(num_labels);

    // Read each label
    for (int i = 0; i < num_labels; ++i) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        labels[i] = label; // Store the label
    }
    file.close();
    return labels;
}

int getNum(std::vector<double> outputs) {
    double largestVal = outputs[0];
    int largestIndex = 0;
    for(int i = 1; i < 10; i++) {
        if(outputs[i] > largestVal) {
            largestVal = outputs[i];
            largestIndex = i;
        }
    }
    return largestIndex;
}