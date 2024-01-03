# CNN

This project implements fully connected layers and convolution layers of a CNN in CPU and GPU. The CUDA implementation parallelizes various operations including matrix multiplication, column sum, RMSProp optimization and convolution operation. The GPU implementation of the network takes less than a minute to train for 20 epochs on full MNIST train dataset and achieves more than 98% test accuracy. 

## Build and Run

CPU implementation is in `CPU/` directory and GPU implementation is in `GPU/` directory. CPU implementation can be built with `CPU/tests/Makefile`. GPU implementation can be built with `GPU-MLP/test/Makefile` directory.
