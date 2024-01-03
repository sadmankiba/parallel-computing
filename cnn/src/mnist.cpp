#include <string>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstring>
#include <iomanip>

#include "mnist.h"

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

float* MNIST::readData(string type, int num){
    string filePath = "../data/MNIST/raw/";
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
    string filePath = "../data/MNIST/raw/";
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

/*
Read num images of a given label from the MNIST dataset

Expects that classImages is already allocated
*/
int MNIST::readClass(string type, unsigned int label, unsigned int num, float *classImages){
    assert (label >= 0 && label <= 9);
    unsigned int num_images = num * 10 * 2;
    float *images = readData(type, num_images); /* Expecting that there will be at least num images of this label in first num * 10 * 2 images */
    float *labels = readLabels(type, num_images);
    unsigned int class_found = 0;

    for (int i = 0; i < num_images; ++i) {
        if (((unsigned int) labels[i]) == label) {
            memcpy(classImages + class_found * 28 * 28, images + i * 28 * 28, 28 * 28 * sizeof(float));
            class_found++;
        }
        if (class_found == num) {
            break;
        }
    }

    return 0;
}

void MNIST::printImage(float* image, int size){
    for (int i=0; i < size; i++){
        for (int j=0; j < size; j++){
            cout << std::fixed << std::setprecision(1) << image[i * size + j] << " ";
        }
        cout << endl;
    }
}

/*
Reduce image size by taking average of (size / new_size) x (size / new_size) pixel blocks

Expects that new_image is already allocated
*/
void MNIST::reduceImage(float *image, float *new_image, int size, int new_size) {
    assert (size % new_size == 0);
    int factor = size / new_size;
    for (int i = 0; i < new_size; ++i) {
        for (int j = 0; j < new_size; ++j) {
            float sum = 0;
            for (int k = 0; k < factor; ++k) {
                for (int l = 0; l < factor; ++l) {
                    sum += image[(i * factor + k) * size + (j * factor + l)];
                }
            }
            new_image[i * new_size + j] = sum / (factor * factor);
        }
    }
}
