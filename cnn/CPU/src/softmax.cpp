#include "softmax.h"
#include <cmath>
#include "../utils/utils.h"

Softmax::Softmax(int _numVals){
    numOut = _numVals;
}

void Softmax::forward(float *_input, float *_output, int numData){
    input = _input;
    output = _output;
    
    // iterate over data points
    for (int i=0; i<numData; i++){

        float exps[numOut];
        float sumExp = 0.0f;
        float max = arrMax(input, i*numOut, (i+1)*numOut-1);

        // iterate over output classes
        for (int j=0; j<numOut; j++){
            float inVal = input[i*numOut+j] - max;
            float expVal = exp(inVal);
            exps[j] = expVal;
            sumExp += expVal;
        }
        for (int j=0; j<numOut; j++){
            output[i*numOut+j] = exps[j]/sumExp;
        }
    }
}

void Softmax::backward(int numData){
    for (int i=0; i<numData; i++){
        for (int j=0; j<numOut; j++)
            input[i*numOut+j] = output[i*numOut+j];
    }
    delete[] output;
}

void Softmax::forward(float *input, float *output) {}

void Softmax::backward() {}

void Softmax::update() {}