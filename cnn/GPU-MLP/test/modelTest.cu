#include "../src/linear.cuh"
#include "../src/relu.cuh"
#include "../src/softmax.cuh"
#include "../src/nllloss.cuh"
#include "../src/mnist.cuh"
#include "../src/base.cuh"
#include "../src/model.cuh"
#include <iostream>
#include "../src/utils.cuh"
using namespace std;


int main(int argc, char* argv[]){
    MNIST* mnist = new MNIST();
    int numImages = 1000;
    float* data = mnist->readData("train", numImages);
    thrust::device_vector<float> dataVector(data, data + numImages * 784);
    float* labels = mnist->readLabels("train", numImages);
    thrust::device_vector<float> labelsVector(numImages*10);
    oneHotEncodeLabels(labels, labelsVector, numImages, 10);
    float lr = 1e-3;
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

    int numEpochs = 30;
    int batchSize = 64;
    int numBatches = (numImages + batchSize - 1) / batchSize;

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
    return 0;
}