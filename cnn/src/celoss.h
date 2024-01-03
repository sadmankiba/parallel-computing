#ifndef CE_LOSS_H
#define CE_LOSS_H

#include "base.h"

class CELoss: public Base{
    public:
        float loss, *labels;
        CELoss(int _numVals);
        void forward(float *input, float *labels, int numData);
        void backward(int numData);

        void forward(float *input, float *output);
        void backward(); 
        void update();
};

#endif