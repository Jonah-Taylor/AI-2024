// #include "layer.hpp"
// #include <random>
// #include <iostream>

// // Dense Layer Constructor
// DenseLayer::DenseLayer(int in_size, int out_size) {
//     w = MatrixXd::Random(in_size, out_size) * 0.01;
//     b = VectorXd::Zero(out_size);
//     d_w = MatrixXd::Zero(in_size, out_size);
//     d_b = VectorXd::Zero(out_size);
// }

// // Dense Layer Forward Pass
// void DenseLayer::forward(const MatrixXd& input) {
//     this->input = input;
//     output = (input * w).rowwise() + b.transpose();
// }

// // Dense Layer Backward Pass
// void DenseLayer::backward(const MatrixXd& d_output) {
//     d_w = input.transpose() * d_output;
//     d_b = d_output.colwise().sum();
//     d_input = d_output * w.transpose();
// }

// // Update Weights in Dense Layer
// void DenseLayer::update(double lr) {
//     w -= lr * d_w;
//     b -= lr * d_b;
// }

// // Get Output from Dense Layer
// MatrixXd DenseLayer::getOutput() const {
//     return output;
// }

// // Convolutional Layer Constructor
// ConvLayer::ConvLayer(int f_size, int num_filters, int stride, int padding)
//     : f_size(f_size), stride(stride), padding(padding) {
//     filters.resize(num_filters);
//     for (int i = 0; i < num_filters; i++) {
//         filters[i] = MatrixXd::Random(f_size, f_size) * 0.01;
//     }
// }

// // Convolutional Layer Forward Pass
// void ConvLayer::forward(const MatrixXd& input) {
//     this->input = input;
//     int out_size = (input.rows() - f_size + 2 * padding) / stride + 1;
//     output.resize(filters.size(), MatrixXd(out_size, out_size));

//     for (int i = 0; i < filters.size(); i++) {
//         output[i] = convolve(input, filters[i], stride, padding);
//     }
// }

// // Convolutional Layer Backward Pass
// void ConvLayer::backward(const MatrixXd& d_output) {
//     d_filters.resize(filters.size());
//     for (int i = 0; i < filters.size(); i++) {
//         d_filters[i] = calcFilterGrads(input, d_output);
//     }
//     d_input = calcInputGrads(d_output, filters[0], stride, padding);
// }

// // Update Filters in Convolutional Layer
// void ConvLayer::update(double lr) {
//     for (int i = 0; i < filters.size(); i++) {
//         filters[i] -= lr * d_filters[i];
//     }
// }

// // Get Output from Convolutional Layer
// MatrixXd ConvLayer::getOutput() const {
//     return output[0];  // Just return the first output (simplified)
// }

// // Convolution Operation
// MatrixXd ConvLayer::convolve(const MatrixXd& input, const MatrixXd& filter, int stride, int padding) {
//     int in_size = input.rows();
//     int out_size = (in_size - f_size + 2 * padding) / stride + 1;
//     MatrixXd result(out_size, out_size);

//     for (int i = 0; i < out_size; i++) {
//         for (int j = 0; j < out_size; j++) {
//             int row_start = i * stride;
//             int col_start = j * stride;
//             MatrixXd region = input.block(row_start, col_start, f_size, f_size);
//             result(i, j) = (region.array() * filter.array()).sum();
//         }
//     }

//     return result;
// }

// // Calculate Filter Gradients
// MatrixXd ConvLayer::calcFilterGrads(const MatrixXd& input, const MatrixXd& d_output) {
//     MatrixXd grad = MatrixXd::Zero(f_size, f_size);
//     for (int i = 0; i < d_output.rows(); i++) {
//         for (int j = 0; j < d_output.cols(); j++) {
//             int row_start = i * stride;
//             int col_start = j * stride;
//             MatrixXd region = input.block(row_start, col_start, f_size, f_size);
//             grad += region * d_output(i, j);
//         }
//     }
//     return grad;
// }

// // Calculate Input Gradients
// MatrixXd ConvLayer::calcInputGrads(const MatrixXd& d_output, const MatrixXd& filter, int stride, int padding) {
//     int in_size = input.rows();
//     MatrixXd d_in = MatrixXd::Zero(in_size, in_size);
//     for (int i = 0; i < d_output.rows(); i++) {
//         for (int j = 0; j < d_output.cols(); j++) {
//             int row_start = i * stride;
//             int col_start = j * stride;
//             MatrixXd region = input.block(row_start, col_start, f_size, f_size);
//             d_in.block(row_start, col_start, f_size, f_size) += filter * d_output(i, j);
//         }3
//     }
//     return d_in;
// }
