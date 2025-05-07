#ifndef CONVOLAYER_HPP
#define CONVOLAYER_HPP

#include "Eigen/Dense"
#include "layer.hpp"

#include <iostream>

using namespace std;
using namespace Eigen;


class ConvoLayer : public Layer {

public:

    uint rows = 0;
    uint cols = 0;
    uint kernel_size = 3;

    MatrixXd w, m_w, v_w, avg_grad_w;
    MatrixXd b, z, a, m_b, v_b, dz, avg_grad_b;
    MatrixXd kernel;

    ConvoLayer();
    ConvoLayer(uint rows, uint cols, string af, uint ks);

    MatrixXd convolve(const MatrixXd& input, const MatrixXd& kernel);
    MatrixXd forward(const MatrixXd& input);
    void getOutputDeltas(const MatrixXd& target);
    void backward(const MatrixXd& w_next, const VectorXd& d_next);
    void backward(const MatrixXd& w_next, const MatrixXd& d_next);
    void updateGrads(const MatrixXd& input);
    void stepSGD(const double& lr, const int& batch_size);
    void stepAdamW(const double& lr, const int& batch_size, int& t);
    pair<uint, uint> size();
    
    ~ConvoLayer() = default;
};

#endif // LAYER_HPP