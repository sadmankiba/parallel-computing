#include "matmul.h"
#include <chrono>
#include <iostream>
#include <string>
#include <time.h>
#include <omp.h>

using namespace std;
using chrono::high_resolution_clock;
using chrono::duration;

float randomFloat(float min, float max){
    return min + (max-min)*((float)rand())/(float)RAND_MAX;
}

int main(int argc, char* argv[]){
    int n = stoi(argv[1]);
    int t = stoi(argv[2]);

    float* A = (float *) malloc(n*n*sizeof(float));
    float* B = (float *) malloc(n*n*sizeof(float));
    float* C = (float *) malloc(n*n*sizeof(float));

    srand(time(0));
    for (int i=0; i<n*n; i++){
        A[i] = randomFloat(-10.0, 10.0);
        B[i] = randomFloat(-10.0, 10.0);
    }

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    omp_set_num_threads(t);
    start = high_resolution_clock::now();
    mmul(A,B,C,n);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    
    cout << C[0] << "\n";
    cout << C[n*n-1] << "\n";
    cout << duration_sec.count() << "\n";

    free(A);
    free(B);
    free(C);

    return 0;
}