#include "model.h"
#include <iostream>
#include "../utils/utils.h"
using namespace std;

Model::Model(Base** _layers, int _numLayers){
    layers = _layers;
    numLayers = _numLayers;
}

void Model::forward(float *input, float *output, int numData){
    float *currOut;
    int sizeOut, i;

    for (i=0; i<numLayers; i++){
        Base* layer = layers[i];
        sizeOut = layer->numOut;
        currOut = new float[numData*sizeOut];
        zeroes(currOut, numData*sizeOut);
        layer->forward(input, currOut, numData);

        input = currOut;
    }
    setEqual(currOut, output, numData*sizeOut);
}

void Model::update(int numData){

    for (int i=numLayers-1; i>=0; i--){
        Base* layer = layers[i];
        layer->update(numData);
        layer->backward(numData);
    }
}