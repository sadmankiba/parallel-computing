#ifndef LAYER_H
#define LAYER_H

#include "base.h"

class Layer: public Base{
    public:
        float *weights, *bias, *oldWeights, lr, *newWeights;
        int numIn, weightSize;

        Layer(int nIn, int nOut, float _lr);

        void forward(float *input, float *output, int numData);
        void backward(int numData);
        void update(int numData);

        void forward(float *input, float *output);
        void backward();
        void update();
};

#endif