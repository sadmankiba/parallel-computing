#include <vector>
#include <cmath>
#include <iostream>

#include "baselayer.h" 
#include "mdvector.h"
#include "imdata.h"
#include "conv.h"
#include "maxpool.h"
#include "flatten.h"
#include "dense.h" 

// sigmoid activation function
float BaseLayer::sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

void BaseLayer::update_weights() {
    std::cout << "Error: update_weights() not implemented" << std::endl;
}

float BaseLayer::activ_forward(float z) {
    return sigmoid(z);
}

float BaseLayer::activ_grad(float z) {
    return sigmoid(z) * (1 - sigmoid(z));
}

float BaseLayer::activ_grad_from_a(float a) {
    return a * (1 - a);
}

void BaseLayer::print_weights() {
    if (this->type == CONV_TYPE) {
        (dynamic_cast<Conv *>(this))->print_weights();
    } else if (this->type == POOL_TYPE) {
        (dynamic_cast<MaxPool *>(this))->print_weights();
    } else if (this->type == FLAT_TYPE) {
        (dynamic_cast<Flatten *>(this))->print_weights();
    } else if (this->type == DENSE_TYPE) {
        (dynamic_cast<Dense *>(this))->print_weights();
    } else {
        std::cout << "Error: Unknown layer type" << std::endl;
        exit(-1);
    }
}
