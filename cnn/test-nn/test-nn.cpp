#include "../src/layer.h" 
#include "../src/relu.h"
#include "../src/softmax.h"
#include "../src/celoss.h"
#include "../src/mnist.h"
#include "../src/model.h"
#include "../src/base.h"
#include "../utils/utils.h"
#include <iostream>
#include <chrono>
using namespace std;

#include <iostream>
#include <random>

// Function to generate random dummy data for MLP testing in row-major order
std::pair<float*, float*> generateDummyData(int numSamples, int numFeatures) {
    std::mt19937 gen(1);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    float* X = new float[numSamples * numFeatures];
    float* y = new float[numSamples];

    for (int i = 0; i < numSamples; ++i) {
        for (int j = 0; j < numFeatures; ++j) {
            X[i * numFeatures + j] = dist(gen);  // Generating random features in row-major order
        }
        y[i] = float(gen() % 3);  // Generating random labels (0, 1, or 2)
    }

    return std::make_pair(X, y);
}

int dummy() {
    int numSamples = 20;
    int numFeatures = 5;

    auto dum = generateDummyData(numSamples, numFeatures);

    float* data = dum.first;
    float* labels = dum.second;

    // Displalabelsing generated data (for demonstration purposes)
    std::cout << "Generated Data:" << std::endl;
    for (int i = 0; i < numSamples; ++i) {
        std::cout << "Sample " << i + 1 << ": ";
        for (int j = 0; j < numFeatures; ++j) {
            std::cout << data[i * numFeatures + j] << " ";
        }
        std::cout << " Label: " << labels[i] << std::endl;
    }

    Base** layers = new Base*[6];
    layers[0] = new Layer(5, 15, 0.001);
    layers[1] = new ReLU(15);
    layers[2] = new Layer(15, 10, 0.001);
    layers[3] = new ReLU(10);
    layers[4] = new Layer(10, 3, 0.001);
    layers[5] = new Softmax(3);
    Model model = Model(layers, 6);
    CELoss* celoss = new CELoss(3);
    
    int numEpochs = 20;
    float* input = new float[numSamples*5];
    float* output = new float[numSamples*3];

    for (int i=0; i<numEpochs; i++){
        cout << "epoch " << i << endl;
        setEqual(data, input, numSamples*5);
        
        model.forward(input, output, numSamples);

        // calc error
        celoss->forward(output, labels, numSamples);
        float loss = celoss->loss;
        cout << "epoch " << i << " loss: " << loss << endl;

        celoss->backward(numSamples);
        model.update(numSamples);
    }

    delete[] layers;
    delete[] data;
    delete[] labels;

    return 0;
}

float calcAccuracy(float* output, float* labels, int numData){
    int numCorrect = 0;
    for (int i=0; i<numData; i++){
        int label = int(labels[i]);
        float maxProb = 0.0f;
        int maxIdx = 0;
        for (int j=0; j<10; j++){
            float prob = output[i*10 + j];
            if (prob > maxProb){
                maxProb = prob;
                maxIdx = j;
            }
        }
        if (maxIdx == label){
            numCorrect++;
        }
    }
    return float(numCorrect)/float(numData);
}

int main (int argc, char *argv[]){
    chrono::duration<double, std::milli> duration; //timer

    MNIST* mnist = new MNIST();
    int numImages = 200;
    float* data = mnist->readData("train", numImages);
    float* labels = mnist->readLabels("train", numImages);
    float lr = 5e-5;

    int numLayers = 8;
    Base** layers = new Base*[numLayers];
    layers[0] = new Layer(784, 1024, lr);
    layers[1] = new ReLU(1024);
    layers[2] = new Layer(1024, 512, lr);
    layers[3] = new ReLU(512);
    layers[4] = new Layer(512, 256, lr);
    layers[5] = new ReLU(256);
    layers[6] = new Layer(256, 10, lr);
    layers[7] = new Softmax(10);
    Model model = Model(layers, numLayers);
    CELoss* celoss = new CELoss(10);
    
    int numEpochs = 15;
    int batchSize = 50;
    int numBatches = numImages/batchSize;
    float* input = new float[batchSize*784];
    float* output = new float[batchSize*10];
    
    auto start = chrono::high_resolution_clock::now();
    for (int i=0; i<numEpochs; i++){
        float loss = 0.0f;
        for (int j=0; j<numBatches; j++){
            float* batchData = data + j*batchSize*784;
            float* batchLabels = labels + j*batchSize;

            setEqual(batchData, input, batchSize*784);
            model.forward(input, output, batchSize);

            // calc error
            celoss->forward(output, batchLabels, batchSize);
            loss += celoss->loss;
            
            celoss->backward(batchSize);
            model.update(batchSize);
        }
        cout << "epoch " << i << " avg batch loss: " << loss/numBatches << endl;
        
    }
    auto stop = chrono::high_resolution_clock::now();
    duration = chrono::duration<double, std::milli>(stop - start);
    cout << "training time for " << numEpochs << " epochs: " << duration.count() << " ms" << endl;

    delete[] layers;
    delete[] data;
    delete[] labels;
    delete[] input;
    delete[] output;

    // dummy();

    // test model accuracy
    int numTestImages = 100;
    float* testData = mnist->readData("test", -1);
    float* testLabels = mnist->readLabels("test", -1);

    start = chrono::high_resolution_clock::now();
    model.forward(testData, output, numTestImages);
    float acc = calcAccuracy(output, testLabels, numTestImages);
    stop = chrono::high_resolution_clock::now();
    duration = chrono::duration<double, std::milli>(stop - start);

    cout << "test accuracy: " << acc << endl;
    cout << "test time: " << duration.count() << " ms" << endl;

    return 0;
}


