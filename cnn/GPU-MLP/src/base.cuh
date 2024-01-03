#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

class Base{
    public:
        thrust::device_vector<float> input, output, gradient;
        int numOut=-1; //number of output nodes

        virtual void forward(thrust::device_vector<float> &input, int numData){};
        virtual void backward(thrust::device_vector<float> &gradient, int numData){};
        virtual void update(thrust::device_vector<float> &gradient, int numData){};
};
