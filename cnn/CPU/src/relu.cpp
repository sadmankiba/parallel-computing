#include "relu.h"

ReLU::ReLU(int n){
    numOut = n;
}

void ReLU::forward(float *in, float *out, int numData){
    input = in;
    output = out;

    for (int i=0; i<numData; i++){
        for (int j=0; j<numOut; j++){
            int index = i*numOut+j;
            out[index] = (in[index] > 0) ? in[index] : 0.0f;
        }
    }
}

void ReLU::backward(int numData){
    for (int i=0; i<numData; i++){
        for (int j=0; j<numOut; j++){
            int index = i*numOut + j;
            input[index] = (input[index] > 0) * output[index];
        }
    }
    delete[] output;
}


void ReLU::forward(float *input, float *output) {}

void ReLU::backward() {}

void ReLU::update() {}
