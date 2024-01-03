#include "celoss.h"
#include <cmath>

const float epsilon = 1e-10;

CELoss::CELoss(int _numVals){
    loss = 0.0f;
    numOut = _numVals;
}

void CELoss::forward(float *_in, float *_labels, int numData){
    loss = 0.0f;
    input = _in;
    labels = _labels;

    // iterate over data points
    for (int i=0; i<numData; i++){
        int label = int(labels[i]);
        float prob = input[i*numOut + label];
        float dataLoss = -1*log(std::max(prob, epsilon));
        loss += dataLoss;
    }
}

void CELoss::backward(int numData){
    // iterate over data points
    for (int i=0; i<numData; i++){
        int label = int(labels[i]);
        input[i*numOut + label] -= 1.0f;
    }
}

void CELoss::forward(float* input, float* output) {}

void CELoss::backward() {}

void CELoss::update() {}
