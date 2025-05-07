// Filename: layer.cpp
// Author: Jonah Taylor
// Date: December 21, 2024
// Description: My first layer based Neural Network

#include "layer.hpp"

DenseLayer::DenseLayer() {

    w = MatrixXd::Random(0, 0);
    b = VectorXd::Zero(0);
    a = VectorXd::Zero(0);
    z = VectorXd::Zero(0);
    dz = VectorXd::Zero(0);

    avg_grad_w = MatrixXd::Zero(0, 0);
    avg_grad_b = VectorXd::Zero(0);

    m_w = MatrixXd::Zero(0, 0);
    v_w = MatrixXd::Zero(0, 0);
    m_b = VectorXd::Zero(0);
    v_b = VectorXd::Zero(0);

};

DenseLayer::DenseLayer(int ls, int input_size) : layer_size(ls) {

    w = MatrixXd::Random(layer_size, input_size) * sqrt(2.0 / input_size);
    b = VectorXd::Zero(layer_size);
    a = VectorXd::Zero(layer_size);
    z = VectorXd::Zero(layer_size);
    dz = VectorXd::Zero(layer_size);

    avg_grad_w = MatrixXd::Zero(layer_size, input_size);
    avg_grad_b = VectorXd::Zero(layer_size);

    m_w = MatrixXd::Zero(layer_size, input_size);
    v_w = MatrixXd::Zero(layer_size, input_size);
    m_b = VectorXd::Zero(layer_size);
    v_b = VectorXd::Zero(layer_size);

};

VectorXd DenseLayer::forward(const VectorXd& in) {

    z = w * in + b;
    a = leakyRelu(z);
    return a;

}

void DenseLayer::getOutputDeltas(const VectorXd& target) {
    
    VectorXd error = a - target;
    dz = error.array().cwiseProduct(leakyReluDerivative(z).array());

}

void DenseLayer::backward(const MatrixXd& w_next, const MatrixXd& d_next) {

    dz = (w_next.transpose() * d_next).array() * leakyReluDerivative(z).array();

}

void DenseLayer::updateGrads(const VectorXd& a_prev) {
    
    avg_grad_w = dz * a_prev.transpose();
    avg_grad_b = dz;

}

void DenseLayer::stepSGD(const double& lr, const int& batch_size) {

    w -= (lr * avg_grad_w.array() / batch_size).matrix();
    b -= (lr * avg_grad_b.array() / batch_size).matrix();

}

void DenseLayer::stepAdamW(const double& lr, const int& batch_size, int& t) {

    //cout << "LS: " << layer_size << "\n";
    
    MatrixXd grad_w = avg_grad_w / batch_size;
    VectorXd grad_b = avg_grad_b / batch_size;

    // cout << "LS2: " << layer_size << "\n";
    // cout << "Dimensions: " << avg_grad_w.rows() << " x " << avg_grad_w.cols() << "\n";
    // cout << "Dimensions: " << grad_w.rows() << " x " << grad_w.cols() << "\n";
    // cout << "Dimensions: " << m_w.rows() << " x " << m_w.cols() << "\n";

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

uint DenseLayer::size() {

    return layer_size;

}