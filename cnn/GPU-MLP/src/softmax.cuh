#pragma once

#include "base.cuh"

class Softmax : public Base {
    public:
        Softmax(int numNodes);

        void forward(thrust::device_vector<float> &input, int numData);
        void backward(thrust::device_vector<float> &gradient, int numData);
};