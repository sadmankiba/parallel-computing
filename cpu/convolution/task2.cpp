#include <cstdio>
#include <cstdlib>
#include <time.h>
#include "convolution.h"

#include <chrono>
#include <ratio>
#include <iostream>
#include <cmath>

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

float randomFloat(float min, float max){
    return min + (max-min)*((float)rand())/(float)RAND_MAX;
}

int main(int argc, char* argv[]){
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    srand(time(0));

    int n = std::atoi(argv[1]);
    int m = std::atoi(argv[2]);

    float *image = new float[n*n];
    for (int i=0; i<n*n; i++) {
        image[i] = randomFloat(-10.0,10.0);
    }

    float *mask = new float[m*m];
    for (int i=0; i<m*m; i++) {
        mask[i] = randomFloat(-1.0,1.0);
    }

    float *output = new float[n*n];
    
    start = high_resolution_clock::now();
    convolve(image, output, n, mask, m);
    end = high_resolution_clock::now();

    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    
    cout << duration_sec.count() << "\n";
    printf("%f\n", output[0]);
    printf("%f\n", output[n*n-1]);

    delete[] image;
    delete[] mask;
    delete[] output;
}