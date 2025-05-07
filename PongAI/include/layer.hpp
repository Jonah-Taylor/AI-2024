#ifndef LAYER_HPP
#define LAYER_HPP

#include "Eigen/Dense"

#include <iostream>

using namespace std;
using namespace Eigen;

class DenseLayer {
public:
    uint layer_size = 0;

    MatrixXd w, m_w, v_w, avg_grad_w;
    VectorXd b, z, a, m_b, v_b, dz, avg_grad_b;

    double beta1 = 0.9;       // Exponential decay rate for first moment
    double beta2 = 0.999;     // Exponential decay rate for second moment
    double epsilon = 1e-8;   // Small constant to avoid division by zero
    double lambda = 0.0;     // Weight decay coefficient (Set to 10% of learning rate)

    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    VectorXd sigmoid(const VectorXd& x) {
        return x.unaryExpr([](double v) { return 1.0 / (1.0 + exp(-v)); });
    }

    double sigmoidDerivative(double x) {
        double sig = sigmoid(x);
        return sig * (1.0 - sig);
    }

    VectorXd sigmoidDerivative(const VectorXd& x) {
        return x.unaryExpr([](double v) {
            double sig = 1.0 / (1.0 + exp(-v));
            return sig * (1.0 - sig);
        });
    }

    double leakyRelu(double x, double alpha = 0.01) {
        return (x > 0) ? x : alpha * x;
    }

    VectorXd leakyRelu(const VectorXd& x, double alpha = 0.01) {
        return x.unaryExpr([alpha](double v) { return v > 0 ? v : alpha * v; });
    }

    double leakyReluDerivative(double x, double alpha = 0.01) {
        return (x > 0) ? 1 : alpha;
    }

    VectorXd leakyReluDerivative(const VectorXd& x, double alpha = 0.01) {
        return x.unaryExpr([alpha](double v) { return v > 0 ? 1 : alpha; });
    }

    DenseLayer();
    DenseLayer(int ls, int input_size);
    ~DenseLayer() = default;

    VectorXd forward(const VectorXd& input);
    void getOutputDeltas(const VectorXd& target);
    void backward(const MatrixXd& w_next, const MatrixXd& d_next);
    void updateGrads(const VectorXd& a_prev);
    void stepSGD(const double& lr, const int& batch_size);
    void stepAdamW(const double& lr, const int& batch_size, int& t);
    uint size();
};

#endif // LAYER_HPP