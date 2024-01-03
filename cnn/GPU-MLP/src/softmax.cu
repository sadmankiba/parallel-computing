#include "softmax.cuh"
#include "utils.cuh"

Softmax::Softmax(int n) {
    this->numOut = n;
}

__global__ void rowMaxKernel(float* input, float* maxVals, int n, int p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float maxVal = input[idx * p];
        for (int i = 1; i < p; i++) {
            maxVal = max(maxVal, input[idx * p + i]);
        }
        maxVals[idx] = maxVal;
    }
}

__global__ void rowSumExpKernel(float* input, float* sumExp, float* maxVals, int n, int p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < p; i++) {
            sum += expf(input[idx * p + i] - maxVals[idx]);
        }
        sumExp[idx] = sum;
    }
}

__global__ void logSoftmaxKernel(float* input, float* maxVals, float* sumExp, float* output, int n, int p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < p; i++) {
            output[idx * p + i] = input[idx * p + i] - maxVals[idx] - logf(sumExp[idx]);
        }
    }
}

void logsoftmax(thrust::device_vector<float> &input, thrust::device_vector<float> &output, int n, int p){
    // calculate max val for each row
    thrust::device_vector<float> maxVals(n);
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    rowMaxKernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(maxVals.data()), n, p);
    CUDA_POST_KERNEL_CHECK;

    // calculate sum of exp for each row
    thrust::device_vector<float> sumExp(n);
    rowSumExpKernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(sumExp.data()), thrust::raw_pointer_cast(maxVals.data()), n, p);
    CUDA_POST_KERNEL_CHECK;

    // calculate logsoftmax
    logSoftmaxKernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(maxVals.data()), thrust::raw_pointer_cast(sumExp.data()), thrust::raw_pointer_cast(output.data()), n, p);
    CUDA_POST_KERNEL_CHECK;
}

__global__ void logSoftmaxDKernel(float* nextGradient, float* gradRowSum, float* output, float* gradient, int n, int p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < p; i++) {
            gradient[idx * p + i] = nextGradient[idx * p + i] - expf(output[idx * p + i]) * gradRowSum[idx];
        }
    }
}

void logsoftmaxD(thrust::device_vector<float> &nextGradient, thrust::device_vector<float> &gradRowSum, thrust::device_vector<float> &output, thrust::device_vector<float> &gradient, int n, int p){
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    logSoftmaxDKernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(nextGradient.data()), thrust::raw_pointer_cast(gradRowSum.data()), thrust::raw_pointer_cast(output.data()), thrust::raw_pointer_cast(gradient.data()), n, p);
    CUDA_POST_KERNEL_CHECK;
}

void Softmax::forward(thrust::device_vector<float> &inp, int numData){
    this->input = inp;
    this->output = thrust::device_vector<float>(numData * numOut);
    logsoftmax(input, output, numData, numOut);
}

void Softmax::backward(thrust::device_vector<float> &nextGradient, int numData){
    gradient = thrust::device_vector<float>(input.size());
    thrust::device_vector<float> gradRowSum(numData);
    rowSum(nextGradient, gradRowSum, numData, numOut);
    logsoftmaxD(nextGradient, gradRowSum, output, gradient, numData, numOut);
}