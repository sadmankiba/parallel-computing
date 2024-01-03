#pragma once
#include "base.cuh"

class Model : public Base{
    public:
        std::vector<Base*> layers;
        int numLayers;

        Model(int numLayers, std::vector<Base*> &layers);
        void forward(thrust::device_vector<float> &input, int numData);
        void backward(thrust::device_vector<float> &lossGradient, int numData);        
};
