#include <cuda.h>
#include "matmul.cuh"
#include <iostream>
#include <cstdlib>
#include <ctime>

int main(int argc, char* argv[]){
    srand(time(0));
    
    unsigned int n = std::atoi(argv[1]);
    int N = n * n;
    unsigned int nBlock = std::atoi(argv[2]);

    // init int matrices
    int *Aint;
    int *Bint;
    int *Cint;
    cudaMallocManaged((void **)&Aint, N * sizeof(int));
    cudaMallocManaged((void **)&Bint, N * sizeof(int));
    cudaMallocManaged((void **)&Cint, N * sizeof(int));
    for (int i=0; i< N; i++) {
        Aint[i] = (rand() % 3) - 0;
        Bint[i] = (rand() % 3) - 0;
    }

    // timer setup
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // execute matmul 1
    cudaEventRecord(start);
    matmul_1(Aint, Bint, Cint, n, nBlock);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << Cint[0] << std::endl;
    std::cout << Cint[N-1] << std::endl;
    std::cout << ms << std::endl;
    cudaFree(Cint);

    // init float matrices
    float *Afloat;
    float *Bfloat;
    float *Cfloat;
    cudaMallocManaged((void **)&Afloat, N* sizeof(float));
    cudaMallocManaged((void **)&Bfloat, N* sizeof(float));
    cudaMallocManaged((void **)&Cfloat, N* sizeof(float));
    for (int i=0; i<N; i++) {
        Afloat[i] = static_cast<float>(Aint[i]);
        Bfloat[i] = static_cast<float>(Bint[i]);
    }

    // execute matmul 2
    cudaEventRecord(start);
    matmul_2(Afloat, Bfloat, Cfloat, n, nBlock);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << Cfloat[0] << std::endl;
    std::cout << Cfloat[N-1] << std::endl;
    std::cout << ms << std::endl;
    cudaFree(Cfloat);
    cudaFree(Aint);
    cudaFree(Bint);


    // init double matrices
    double *Adouble;
    double *Bdouble;
    double *Cdouble;
    cudaMallocManaged((void **)&Adouble, N* sizeof(double));
    cudaMallocManaged((void **)&Bdouble, N* sizeof(double));
    cudaMallocManaged((void **)&Cdouble, N* sizeof(double));
    for (int i=0; i<N; i++) {
        Adouble[i] = static_cast<double>(Afloat[i]);
        Bdouble[i] = static_cast<double>(Bfloat[i]);
    }

    // execute matmul 3
    cudaEventRecord(start);
    matmul_3(Adouble, Bdouble, Cdouble, n, nBlock);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << Cdouble[0] << std::endl;
    std::cout << Cdouble[N-1] << std::endl;
    std::cout << ms << std::endl;

    // cleanup
    cudaFree(Cdouble);
    cudaFree(Afloat);
    cudaFree(Bfloat);
    cudaFree(Adouble);
    cudaFree(Bdouble);

    return 0;
}
