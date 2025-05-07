#ifndef DENSELAYER_HPP
#define DENSELAYER_HPP

#include "Eigen/Dense"
#include "layer.hpp"

#include <iostream>

using namespace std;
using namespace Eigen;

class DenseLayer : public Layer {

public:

    uint l_size = 0;

    MatrixXd w, m_w, v_w, avg_grad_w;
    VectorXd b, z, a, m_b, v_b, dz, avg_grad_b;

    DenseLayer();
    DenseLayer(int ls, int in_size, string afn);
    VectorXd forward(const MatrixXd& in);
    VectorXd forward(const VectorXd& in);
    void getOutputDeltas(const VectorXd& target);
    void backward(const MatrixXd& w_next, const VectorXd& d_next);
    void updateGrads(const VectorXd& in);
    void stepSGD(const double& lr, const int& bs);
    void stepAdamW(const double& lr, const int& bs, int& t);
    uint size();
    
    ~DenseLayer() = default;
};

#endif // LAYER_HPP