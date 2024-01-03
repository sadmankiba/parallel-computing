#include "nllloss.cuh"
#include "utils.cuh"

NLLLoss::NLLLoss(int n) {
    this->numOut = n;
}

float negLogLhoodLoss(thrust::device_vector<float> &input, thrust::device_vector<float> &labels, int n, int p) {
    // calculate product of log probabilities and labels
    thrust::device_vector<float> product(n * p);
    thrust::transform(input.begin(), input.end(), labels.begin(), product.begin(), thrust::multiplies<float>());
    
    // calculate row sum
    thrust::device_vector<float> lossPerItem(n);
    rowSum(product, lossPerItem, n, p);

    // calculate total loss for batch
    float loss = thrust::reduce(lossPerItem.begin(), lossPerItem.end());
    return -1.0f*loss;
}

void NLLLoss::forward(thrust::device_vector<float> &inp, thrust::device_vector<float> &targets, int numData) {
    this->input = inp;
    this->labels = targets;
    this->loss = negLogLhoodLoss(input, labels, numData, numOut);
}

struct scalarMultFunctor {
    const float scale;
    scalarMultFunctor(float _scale) : scale(_scale) {}

    __host__ __device__
    float operator()(float label) const {
        return label * scale;
    }
};

void NLLLoss::backward(int numData){
    this->gradient = thrust::device_vector<float>(numData * numOut);
    float scale = -1.0f / static_cast<float>(numData);
    thrust::transform(labels.begin(), labels.end(), gradient.begin(), scalarMultFunctor(scale));
}