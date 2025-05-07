#ifndef CONVOLAYER_HPP
#define CONVOLAYER_HPP

#include "Eigen/Dense"
#include "layer.hpp"

using namespace std;
using namespace Eigen;

class ConvoLayer : public Layer {

public:

    size_t in_rows = 0;
    size_t in_cols = 0;

    size_t out_rows = 0;
    size_t out_cols = 0;

    size_t k_size = 0;
    MatrixXd kernel;

    ConvoLayer();
    ConvoLayer(vector<size_t> in_size, vector<size_t> out_size, string af, size_t ks);

    MatrixXd convolve(const MatrixXd& in, const size_t out_rows, const size_t out_cols, const MatrixXd& k);

    MatrixXd forward(const MatrixXd& in) override;
    void getOutputDeltas(const MatrixXd& target) override;
    void backward(const MatrixXd& w_next, const MatrixXd& d_next) override;
    void updateGrads(const MatrixXd& in) override;
    void stepSGD(const double& lr, const size_t& bs) override;
    void stepAdamW(const double& lr, const size_t& bs, size_t& t) override;
    pair<size_t, size_t> size() override;
    
    ~ConvoLayer() = default;
};

#endif