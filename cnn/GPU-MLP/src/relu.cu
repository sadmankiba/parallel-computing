#include "relu.cuh"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <thrust/transform.h>

ReLU::ReLU(int n) {
    this->numOut = n;
}

struct reluFunctor {
    __host__ __device__ float operator()(const float& x) {
        return fmaxf(0.0f, x);
    }
};

struct reluDFunctor {
    __host__ __device__ float operator()(const float& x, const float& y) {
        return x > FLT_EPSILON ? y : 0;
    }
};

void ReLU::forward(thrust::device_vector<float> &inp, int numData) {
    this->input = inp;
    this->output = thrust::device_vector<float>(numData * numOut);
    thrust::transform(input.begin(), input.end(), output.begin(), reluFunctor());
}    

void ReLU::backward(thrust::device_vector<float> &nextGradient, int numData) {
    gradient = thrust::device_vector<float>(input.size());
    thrust::transform(input.begin(), input.end(), nextGradient.begin(), gradient.begin(), reluDFunctor());
}
    