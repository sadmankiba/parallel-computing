#pragma once

#include "base.cuh"

class NLLLoss : public Base {
    public:
        float loss;
        thrust::device_vector<float> labels;
        NLLLoss(int numNodes);

        void forward(thrust::device_vector<float> &input, thrust::device_vector<float> &labels, int numData);
        void backward(int numData);
};