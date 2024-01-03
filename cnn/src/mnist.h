#ifndef MNIST_H
#define MNIST_H

class MNIST{
    public:
        MNIST();
        float* readData(std::string type, int numData);
        float* readLabels(std::string type, int numData);
        int readClass(std::string type, unsigned int label, unsigned int num, float *images);
        void reduceImage(float *image, float *new_image, int size, int new_size);
        void printImage(float* image, int size);
};

#endif