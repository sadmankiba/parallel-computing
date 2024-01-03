#include "linear.cuh"
#include "utils.cuh"

Linear::Linear(int nIn, int nOut, float lr, float beta){
    this->numIn = nIn;
    this->numOut = nOut;
    this->lr = lr;
    this->beta = beta;

    this->weights = thrust::device_vector<float>(nIn * nOut);
    this->wtSquareGrad = thrust::device_vector<float>(nIn * nOut, SQUARE_GRAD_DEFAULT);
    kaiming_init(this->weights, nIn, nOut);

    this->bias = thrust::device_vector<float>(nOut, 0.0f);
    this->biasSquareGrad = thrust::device_vector<float>(nOut, SQUARE_GRAD_DEFAULT);
}

void Linear::forward(thrust::device_vector<float> &inp, int batchSize){
    this->input = inp;
    output = thrust::device_vector<float>(batchSize * numOut);

    matmul(input, weights, output, batchSize, numIn, numOut);
    addBias(output, bias, batchSize, numOut);
}

void Linear::backward(thrust::device_vector<float> &nextGradient, int batchSize){
    // calculate product of gradient and (old) weights
    thrust::device_vector<float> weightsT(weights.size());
    transpose(weights, weightsT, numIn, numOut);
    gradient = thrust::device_vector<float>(input.size());
    matmul(nextGradient, weightsT, gradient, batchSize, numOut, numIn);

    weights = newWeights;
}

void Linear::update(thrust::device_vector<float> &nextGradient, int batchSize){
    // calculate column sum of gradient
    thrust::device_vector<float> gradientSum(numOut, 0.0f);
    columnSum(nextGradient, gradientSum, batchSize, numOut);

    // update bias
    rmsPropUpdate(bias, gradientSum, biasSquareGrad, lr, beta);

    // calculate product of gradient and input
    thrust::device_vector<float> inputT(input.size());
    transpose(input, inputT, batchSize, numIn);
    thrust::device_vector<float> prodGrad(weights.size());
    matmul(inputT, nextGradient, prodGrad, numIn, batchSize, numOut);

    // calculate new weights
    newWeights = weights;
    rmsPropUpdate(newWeights, prodGrad, wtSquareGrad, lr, beta);
}