// Filename: convolayer.cpp
// Author: Jonah Taylor
// Date: December 21, 2024
// Description: My first layer based Neural Network

#include "convolayer.hpp"

ConvoLayer::ConvoLayer() {

    w = MatrixXd::Random(0, 0);
    b = MatrixXd::Zero(0, 0);
    a = MatrixXd::Zero(0, 0);
    z = MatrixXd::Zero(0, 0);
    dz = MatrixXd::Zero(0, 0);

    avg_grad_w = MatrixXd::Zero(0, 0);
    avg_grad_b = MatrixXd::Zero(0, 0);

    m_w = MatrixXd::Zero(0, 0);
    v_w = MatrixXd::Zero(0, 0);
    m_b = MatrixXd::Zero(0, 0);
    v_b = MatrixXd::Zero(0, 0);
};

ConvoLayer::ConvoLayer(uint r, uint c, string afn, uint ks) : rows(r), cols(c), kernel_size(ks) {

    w = MatrixXd::Random(kernel_size, kernel_size) * sqrt(2.0 / (kernel_size * kernel_size));
    b = MatrixXd::Zero(rows, cols);
    a = MatrixXd::Zero(rows, cols);
    z = MatrixXd::Zero(rows, cols);
    dz = MatrixXd::Zero(rows, cols);

    avg_grad_w = MatrixXd::Zero(kernel_size, kernel_size);
    avg_grad_b = MatrixXd::Zero(rows, cols);

    m_w = MatrixXd::Zero(kernel_size, kernel_size);
    v_w = MatrixXd::Zero(kernel_size, kernel_size);
    m_b = MatrixXd::Zero(rows, cols);
    v_b = MatrixXd::Zero(rows, cols);

    setActFunc(afn);

};

MatrixXd ConvoLayer::convolve(const MatrixXd& in, const MatrixXd& kernel) {

    if(w.rows() % 2 == 0) std::cerr << "Kernel must have an odd number of rows and columns!" << std::endl;
    if(w.rows() != w.cols()) std::cerr << "The kernel must be a square!" << std::endl;

    uint padding = (kernel.rows() - 1) / 2;
    MatrixXd padded_input = MatrixXd::Zero(in.rows() + 2 * padding, in.cols() + 2 * padding);
    padded_input.block(padding, padding, in.rows(), in.cols()) = in;

    MatrixXd output = MatrixXd::Zero(in.rows(), in.cols());

    for (int r = 0; r < output.rows(); r++) {
        for (int c = 0; c < output.cols(); c++) {
            output(r, c) = (padded_input.block(r, c, kernel.rows(), kernel.cols()).array() * kernel.array()).sum();
        }
    }

    return output;
}

MatrixXd ConvoLayer::forward(const MatrixXd& in) {

    z = convolve(in, w) + b;
    a = a_func(z);
    return a;

}

void ConvoLayer::getOutputDeltas(const MatrixXd& target) {
    
    MatrixXd error = a - target;
    dz = error.array().cwiseProduct(a_func_deri(z).array());

    // Calculate the gradient of the loss with respect to the kernel weights
    int padding = (w.rows() - 1) / 2;
    MatrixXd padded_dz = MatrixXd::Zero(dz.rows() + 2 * padding, dz.cols() + 2 * padding);
    padded_dz.block(padding, padding, dz.rows(), dz.cols()) = dz;

    for (int r = 0; r < w.rows(); ++r) {
        for (int c = 0; c < w.cols(); ++c) {
            avg_grad_w(r, c) = (padded_dz.block(r, c, dz.rows(), dz.cols()).array() * dz.array()).sum();
        }
    }

    avg_grad_b = dz.colwise().sum();
}

void ConvoLayer::backward(const MatrixXd& w_next, const VectorXd& d_next) {

    MatrixXd d_next_matrix = Eigen::Map<const MatrixXd>(d_next.data(), z.rows(), z.cols());
    backward(w_next, d_next_matrix);

}

void ConvoLayer::backward(const MatrixXd& w_next, const MatrixXd& d_next) {

    dz = (w_next.transpose() * d_next).array() * a_func_deri(z).array();

}

void ConvoLayer::updateGrads(const MatrixXd& in) {
    
    avg_grad_w = convolve(in, dz);
    avg_grad_b = dz;

}

void ConvoLayer::stepSGD(const double& lr, const int& bs) {

    w -= (lr * avg_grad_w.array() / bs).matrix();
    b -= (lr * avg_grad_b.array() / bs).matrix();

}

void ConvoLayer::stepAdamW(const double& lr, const int& bs, int& t) {

    MatrixXd grad_w = avg_grad_w / bs;
    MatrixXd grad_b = avg_grad_b / bs;

    m_w = (beta1 * m_w + (1 - beta1) * grad_w).matrix(); 
    v_w = beta2 * v_w + (1 - beta2) * grad_w.array().square().matrix();
    
    m_b = (beta1 * m_b + (1 - beta1) * grad_b).matrix(); 
    v_b = beta2 * v_b + (1 - beta2) * grad_b.array().square().matrix(); 
    
    MatrixXd m_hat_w = m_w / (1 - pow(beta1, t));
    MatrixXd v_hat_w = v_w / (1 - pow(beta2, t));

    MatrixXd m_hat_b = m_b / (1 - pow(beta1, t));
    MatrixXd v_hat_b = v_b / (1 - pow(beta2, t));

    w -= (lr * (m_hat_w.array() / (v_hat_w.array().sqrt() + epsilon)).matrix() + lambda * w).matrix();
    b -= (lr * (m_hat_b.array() / (v_hat_b.array().sqrt() + epsilon)).matrix() + lambda * b).matrix();
    
}

pair<uint, uint> ConvoLayer::size() {

    return {rows, cols};

}