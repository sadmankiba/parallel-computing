#include "../src/linear.cuh"
#include "../src/relu.cuh"
#include "../src/softmax.cuh"
#include "../src/nllloss.cuh"
#include "../src/mnist.cuh"
#include "../src/base.cuh"
#include "../src/model.cuh"
#include "../src/utils.cuh"
#include <iostream>
#include <thrust/inner_product.h>
#include <ctime>
using namespace std;

struct MaxIndexFunctor {
    int numClasses;
    thrust::device_vector<float>::iterator logProbsStart;

    MaxIndexFunctor(int numClasses, thrust::device_vector<float>::iterator logProbsStart)
        : numClasses(numClasses), logProbsStart(logProbsStart) {}

    __device__ int operator()(int imageIndex) {
        auto start = logProbsStart + imageIndex * numClasses;
        auto end = start + numClasses;
        auto maxElement = thrust::max_element(thrust::seq, start, end);
        return thrust::distance(start, maxElement);
    }
};

float calcAccuracy(thrust::device_vector<float> &logProbs, thrust::device_vector<float> &labels, int numImages){
    int numClasses = 10;
    thrust::device_vector<int> predictions(numImages);
    thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(numImages), predictions.begin(), MaxIndexFunctor(numClasses, logProbs.begin()));

    int correctPredictions = thrust::inner_product(predictions.begin(), predictions.end(), labels.begin(), 0, thrust::plus<int>(), thrust::equal_to<int>());

    return (float)correctPredictions / numImages;
}

int main(int argc, char* argv[]){
     // timer setup
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // read data
    MNIST* mnist = new MNIST();
    int numImages = 60000;
    float* data = mnist->readData("train");
    thrust::device_vector<float> dataVector(data, data + numImages * 784);

    float* labels = mnist->readLabels("train");
    thrust::device_vector<float> labelsVector(numImages*10);
    oneHotEncodeLabels(labels, labelsVector, numImages, 10);

    // create model
    float lr = 1e-4;
    float beta = 0.99f;

    int numLayers = 8;
    std::vector<Base*> layers = std::vector<Base*>(numLayers);
    layers[0] = new Linear(784, 1024, lr, beta);
    layers[1] = new ReLU(1024);
    layers[2] = new Linear(1024, 512, lr, beta);
    layers[3] = new ReLU(512);
    layers[4] = new Linear(512, 256, lr, beta);
    layers[5] = new ReLU(256);
    layers[6] = new Linear(256, 10, lr, beta);
    layers[7] = new Softmax(10);

    Model model = Model(numLayers, layers);
    NLLLoss nllll = NLLLoss(10);

    // train model
    int numEpochs = 20;
    int batchSize = 64;
    int numBatches = (numImages + batchSize - 1) / batchSize;

    cudaEventRecord(start);
    for (int i=0; i<numEpochs; i++){
        float cumBatchLoss = 0.0f;
        for (int j=0; j<numBatches; j++){
            int batchNumImages = min(batchSize, numImages - j*batchSize);
            thrust::device_vector<float> batchData(batchNumImages*784);
            thrust::device_vector<float>::iterator firstData = dataVector.begin() + j*batchSize*784;
            thrust::copy(firstData, firstData + batchNumImages*784, batchData.begin());

            thrust::device_vector<float> batchLabels(batchNumImages*10);
            thrust::device_vector<float>::iterator firstLabels = labelsVector.begin() + j*batchSize*10;
            thrust::copy(firstLabels, firstLabels + batchNumImages*10, batchLabels.begin());

            model.forward(batchData, batchNumImages);
            nllll.forward(model.output, batchLabels, batchNumImages);
            cumBatchLoss += nllll.loss;
            nllll.backward(batchNumImages);
            model.backward(nllll.gradient, batchNumImages);
        }
        cout << "epoch " << i << " avg batch loss: " << cumBatchLoss / numBatches << endl;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "training time for " << numEpochs << " epochs: " << ms << " ms" << endl;

    // test model
    int numTestImages = 10000;
    float* testData = mnist->readData("test");
    thrust::device_vector<float> testDataVector(testData, testData + numTestImages * 784);

    float* testLabels = mnist->readLabels("test");
    thrust::device_vector<float> testLabelsVector(testLabels, testLabels + numTestImages);

    cudaEventRecord(start);
    model.forward(testDataVector, numTestImages);
    float acc = calcAccuracy(model.output, testLabelsVector, numTestImages);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    cout << "test accuracy: " << acc << endl;
    cout << "test time: " << ms << " ms" << endl;

    return 0;
}