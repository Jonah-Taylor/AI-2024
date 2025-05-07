// Filename: denselayer.cpp
// Author: Jonah Taylor
// Date: December 21, 2024
// Description: My first layer based Neural Network

#include "denselayer.hpp"

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

DenseLayer::DenseLayer(int ls, int in_size, string afn) : l_size(ls) {

    w = MatrixXd::Random(l_size, in_size) * sqrt(2.0 / in_size);
    b = VectorXd::Zero(l_size);
    a = VectorXd::Zero(l_size);
    z = VectorXd::Zero(l_size);
    dz = VectorXd::Zero(l_size);

    avg_grad_w = MatrixXd::Zero(l_size, in_size);
    avg_grad_b = VectorXd::Zero(l_size);

    m_w = MatrixXd::Zero(l_size, in_size);
    v_w = MatrixXd::Zero(l_size, in_size);
    m_b = VectorXd::Zero(l_size);
    v_b = VectorXd::Zero(l_size);

    setActFunc(afn);

};


VectorXd DenseLayer::forward(const MatrixXd& in) {

    VectorXd vec = Eigen::Map<const VectorXd>(in.data(), in.size());
    return forward(vec);

}

VectorXd DenseLayer::forward(const VectorXd& in) {

    z = w * in + b;
    a = a_func(z);
    return a;

}

void DenseLayer::getOutputDeltas(const VectorXd& target) {
    
    VectorXd error = a - target;
    dz = error.array().cwiseProduct(a_func_deri(z).array());

}

void DenseLayer::backward(const MatrixXd& w_next, const VectorXd& d_next) {
    
    dz = (w_next.transpose() * d_next).array() * a_func_deri(z).array();

}

void DenseLayer::updateGrads(const VectorXd& in) {
    
    avg_grad_w = dz * in.transpose();
    avg_grad_b = dz;

}

void DenseLayer::stepSGD(const double& lr, const int& bs) {

    w -= (lr * avg_grad_w.array() / bs).matrix();
    b -= (lr * avg_grad_b.array() / bs).matrix();

}

void DenseLayer::stepAdamW(const double& lr, const int& bs, int& t) {

    MatrixXd grad_w = avg_grad_w / bs;
    VectorXd grad_b = avg_grad_b / bs;

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

    return l_size;

}