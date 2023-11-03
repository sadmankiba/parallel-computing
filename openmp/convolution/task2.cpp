#include <string>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <omp.h>
#include <iostream>
#include "convolution.h"
using namespace std;
using chrono::high_resolution_clock;
using chrono::duration;

float randomFloat(float min, float max){
    return min + (max-min)*((float)rand())/(float)RAND_MAX;
}

int main(int argc, char* argv[]){
    int n = stoi(argv[1]);
    int t = stoi(argv[2]);

    srand(time(0));
    float *image = (float*) malloc(n*n*sizeof(float));
    for (int i=0; i<n*n; i++) {
        image[i] = randomFloat(-10.0,10.0);
    }

    float *mask = (float*) malloc(9*sizeof(float));
     for (int i=0; i<9; i++) {
        mask[i] = randomFloat(-1.0,1.0);
    }

    float *output = (float*) malloc(n*n*sizeof(float));

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    omp_set_num_threads(t);
    start = high_resolution_clock::now();
    convolve(image, output, n, mask, 3);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    
    cout << output[0] << "\n";
    cout << output[n*n-1] << "\n";
    cout << duration_sec.count() << "\n";

    // start = high_resolution_clock::now();
    // convolveNC(image, output, n, mask, 3);
    // end = high_resolution_clock::now();
    // duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    // // cout << output[0] << "\n";
    // // cout << output[n*n-1] << "\n";
    // cout << duration_sec.count() << "\n\n";

    free(image);
    free(output);
    free(mask);
    return 0;
}