#pragma once

#include "base.cuh"

class Linear : public Base {
    public:
        int numIn;
        float lr, beta;
        thrust::device_vector<float> weights, bias, newWeights, biasSquareGrad, wtSquareGrad;
        
        Linear(int numIn, int numOut, float lr, float b);

        void forward(thrust::device_vector<float> &input, int numData);
        void backward(thrust::device_vector<float> &gradient, int numData);
        void update(thrust::device_vector<float> &gradient, int numData);
};

