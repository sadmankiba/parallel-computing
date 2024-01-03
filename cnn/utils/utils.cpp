#include <cmath>
#include <random>
#include "utils.h"

using namespace std;

void kaiming_init(float *weights, int nIn, int nOut){
    float sdev = sqrt(2/float(nIn));
    random_device rd; //init random device
    mt19937 gen(rd()); //init random num gen
    // mt19937 gen(10); //init random num gen
    normal_distribution<float> dist(0.0f, sdev);

    for (int i=0; i<nIn*nOut; i++){
        weights[i] = dist(gen);
    }
}

void zeroes(float *arr, int n){
    for (int i=0; i<n; i++) arr[i] = 0.0f;
}

void ones(float *arr, int n){
    for (int i=0; i<n; i++) arr[i] = 1.0f;
}

// float* deepCopy(float* arr, int size){
//     float* copy = new float[size];
//     memcpy(copy, arr, size*sizeof(float));
//     return cpoy;
// }

void setEqual(float* src, float* dest, int size){
    for (int i=0; i<size; i++)
        dest[i] = src[i];
}

float arrMax(float* arr, int start, int end){
    float tempMax = arr[start];
    for (int i=start+1; i<=end; i++){
        float x = arr[i];
        if (x>tempMax) tempMax = x;
    }
    return tempMax;
}