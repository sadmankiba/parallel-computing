#include <string>

#ifndef MNIST_H
#define MNIST_H

class MNIST{
    public:
        MNIST();
        float* readData(std::string filepath, int numData);
        float* readLabels(std::string filepath, int numData);
};

#endif