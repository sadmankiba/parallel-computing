#include <cstdlib>
#include <ctime>
#include <cuda.h>
#include <iostream>
#include "matmul.cuh"


float randomFloat(float min, float max){
    return min + (max-min)*(((float)rand())/(float)RAND_MAX);
}

int main(int argc, char* argv[]){
    srand(time(0));
    int n = std::atoi(argv[1]);
    int N = n*n;
    int tpb = std::atoi(argv[2]);

    // init host matrices
    float *A = new float[N];
    float *B = new float[N];
    float *C = new float[N];
    for (int i=0; i<N; i++) {
        A[i] = randomFloat(-1.0,1.0);
        B[i] = randomFloat(-1.0,1.0);
    }

    // init device matrices
    float *devA;
    float *devB;
    float *devC;
    cudaMalloc((void**)&devA, sizeof(float)*(N));
    cudaMalloc((void**)&devB, sizeof(float)*(N));
    cudaMalloc((void**)&devC, sizeof(float)*(N));
    cudaMemcpy(devA, A, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, sizeof(float)*N, cudaMemcpyHostToDevice);

    // timer setup
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // execute
    cudaEventRecord(start);
    matmul(devA, devB, devC, n, tpb);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // copy output back to host
    cudaMemcpy(C, devC, sizeof(float)*N, cudaMemcpyDeviceToHost);

    std::cout << C[N-1] << std::endl;
    std::cout << ms << std::endl;

    // cleanup
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}