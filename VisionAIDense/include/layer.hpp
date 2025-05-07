#ifndef LAYER_HPP
#define LAYER_HPP

#include "Eigen/Dense"
#include <vector>

using namespace Eigen;
using namespace std;

// Base Layer class
class Layer {
public:
    virtual void forward(const MatrixXd& input) = 0;
    virtual void backward(const MatrixXd& d_output) = 0;
    virtual void update(double lr) = 0;
    virtual MatrixXd getOutput() const = 0;
};

// Dense Layer
class DenseLayer : public Layer {
public:
    MatrixXd w, b, input, output, d_w, d_b, d_input;

    DenseLayer(int in_size, int out_size);
    void forward(const MatrixXd& input) override;
    void backward(const MatrixXd& d_output) override;
    void update(double lr) override;
    MatrixXd getOutput() const override;
};

// Convolutional Layer
class ConvLayer : public Layer {
public:
    vector<MatrixXd> filters, output, d_filters;
    MatrixXd input, d_input;
    int f_size, stride, padding;

    ConvLayer(int f_size, int num_filters, int stride, int padding);
    void forward(const MatrixXd& input) override;
    void backward(const MatrixXd& d_output) override;
    void update(double lr) override;
    MatrixXd getOutput() const override;

private:
    MatrixXd convolve(const MatrixXd& input, const MatrixXd& filter, int stride, int padding);
    MatrixXd calcFilterGrads(const MatrixXd& input, const MatrixXd& d_output);
    MatrixXd calcInputGrads(const MatrixXd& d_output, const MatrixXd& filter, int stride, int padding);
};

#endif // LAYER_HPP
