#ifndef RELU_H
#define RELU_H

#include "base.h"

class ReLU: public Base {
    public:
        ReLU(int numVals);
        void forward(float *input, float *output, int numData);
        void backward(int numData);

        void forward(float *input, float *output);
        void backward();
        void update();
};

#endif