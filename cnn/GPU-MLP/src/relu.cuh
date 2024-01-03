#pragma once

#include "base.cuh"

class ReLU : public Base {
    public:
        ReLU(int numNodes);

        void forward(thrust::device_vector<float> &input, int numData);
        void backward(thrust::device_vector<float> &gradient, int numData);
};

