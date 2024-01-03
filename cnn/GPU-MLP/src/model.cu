#include "model.cuh"

Model::Model(int n, std::vector<Base*> &_layers) {
    this->numLayers = n;
    this->layers = _layers;
}

void Model::forward(thrust::device_vector<float> &input, int numData) {
    thrust::device_vector<float> currOutput = input;
    for (int i=0; i<numLayers; i++){
        Base* layer = layers[i];
        layer->forward(currOutput, numData);
        currOutput = layer->output;
    }
    this->output = currOutput;
}

void Model::backward(thrust::device_vector<float> &lossGradient, int numData){
    thrust::device_vector<float> gradient = lossGradient;
    for (int i=numLayers-1; i>=0; i--){
        Base* layer = layers[i];
        layer->update(gradient, numData);
        layer->backward(gradient, numData);
        gradient = layer->gradient;
    }
}