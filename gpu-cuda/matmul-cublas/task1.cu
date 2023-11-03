#include <cstdlib>
#include <ctime>
#include <cuda.h>
#include <iostream>
#include "mmul.h"
#include <cublas_v2.h>

float randomFloat(float min, float max){
    return min + (max-min)*(((float)rand())/(float)RAND_MAX);
}

int main(int argc, char* argv[]){
    srand(time(0));
    int n = std::atoi(argv[1]);
    int N = n*n;
    int n_tests = std::atoi(argv[2]);

    // init matrices
    float *A;
    float *B;
    float *C;
    cudaMallocManaged((void **)&A, N*sizeof(float));
    cudaMallocManaged((void **)&B, N*sizeof(float));
    cudaMallocManaged((void **)&C, N*sizeof(float));
    for (int i=0; i<N; i++) {
        A[i] = randomFloat(-1,1);
        B[i] = randomFloat(-1,1);
    }

    // init cuBLAS handle
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    // timer setup
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // execute
    float totalTime = 0;
    for (int i=0; i<n_tests; i++){
        cudaEventRecord(start);
        mmul(handle, A, B, C, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        totalTime += ms;
    }

    float avg = totalTime/n_tests;
    std::cout << avg << std::endl;

    // cleanup
    cudaFree(C);
    cudaFree(A);
    cudaFree(B);
    cublasDestroy(handle);
    
    return 0;
}