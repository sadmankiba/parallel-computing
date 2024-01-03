#include "mnist.cuh"
#include <iostream>
#include <fstream>

using namespace std;

MNIST::MNIST(){}

int reverse(int i){
    unsigned char c1,c2,c3,c4;
    c1 = i & 255;
    c2 = (i>>8) & 255;
    c3 = (i>>16) & 255;
    c4 = (i>>24) & 255;
    return ((int)c1<<24) + ((int)c2<<16) + ((int) c3<<8) + c4;
}

void oneHotEncodeLabels(float* labels, thrust::device_vector<float> &labelsVector, int numData, int numClasses){
    for (int i=0; i<numData; i++){
        int label = int(labels[i]);
        for (int j=0; j<numClasses; j++){
            if (j == label){
                labelsVector[i*numClasses + j] = 1.0f;
            }
            else{
                labelsVector[i*numClasses + j] = 0.0f;
            }
        }
    }
}

float* MNIST::readData(string type, int num){
    string filePath = "../data/MNIST/";
    if (type == "train"){
        filePath += "train-images-idx3-ubyte";
    } else if (type == "test"){
        filePath += "t10k-images-idx3-ubyte";
    }

    ifstream file(filePath, ios::binary);
    if (!file.is_open()){
        cout << "Error opening " + filePath << endl;
        return nullptr;
    }

    // read header
    int magicNumber, numImages, rows, cols;
    file.read(reinterpret_cast<char*> (&magicNumber), sizeof(magicNumber));
    file.read(reinterpret_cast<char*> (&numImages), sizeof(numImages));
    file.read(reinterpret_cast<char*> (&rows), sizeof(rows));
    file.read(reinterpret_cast<char*> (&cols), sizeof(cols));

    magicNumber = reverse(magicNumber);
    numImages = reverse(numImages);
    rows = reverse(rows);
    cols = reverse(cols);

    // cout << magicNumber << endl;
    // cout << numImages << endl;
    // cout << rows << " x " << cols << endl;

    // read data
    int n = rows*cols;
    int numPixels = (num>0) ? num*n : numImages*n;
    // cout << numPixels << endl;
    float *data = new float[numPixels];
    unsigned char temp;
    for (int i=0; i<numPixels; i++){
        file.read(reinterpret_cast<char*>(&temp), sizeof(temp));
        float tempFloat = static_cast<float>(temp);
        data[i] = tempFloat/255;
    }

    file.close();

    // for (int i = 0; i < 500; ++i) {
    //     if (i%28 ==0) cout << endl;
    //     cout << data[i] << " ";
    // }
    // cout << endl;
    return data;
}

float* MNIST::readLabels(string type, int num){
    string filePath = "../data/MNIST/";
    if (type == "train"){
        filePath += "train-labels-idx1-ubyte";
    } else if (type == "test"){
        filePath += "t10k-labels-idx1-ubyte";
    }

    ifstream file(filePath, ios::binary);
    if (!file.is_open()){
        cout << "Error opening " + filePath << endl;
        return nullptr;
    }

    // read header
    int magicNumber, numLabels;
    file.read(reinterpret_cast<char*> (&magicNumber), sizeof(magicNumber));
    file.read(reinterpret_cast<char*> (&numLabels), sizeof(numLabels));

    magicNumber = reverse(magicNumber);
    numLabels = reverse(numLabels);

    // cout << magicNumber << endl;
    // cout << numLabels << endl;

    // read data
    int numVals = (num>0) ? num : numLabels;

    float *labels = new float[numVals];
    unsigned char temp;
    for (int i=0; i<numVals; i++){
        file.read(reinterpret_cast<char*>(&temp), sizeof(temp));
        labels[i] = static_cast<float>(temp);
    }

    file.close();

    // for (int i = 0; i < numVals; ++i) {
    //     cout << labels[i] << " ";
    // }
    return labels;
}

