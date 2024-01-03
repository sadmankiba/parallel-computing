#ifndef SOFTMAX_H
#define SOFTMAX_H
#include "base.h"

class Softmax: public Base {
    public:
        Softmax(int _numVals);
        void forward(float *input, float *output, int numData);
        void backward(int numData);

        void forward(float *input, float *output);
        void backward();
        void update();
};

#endif