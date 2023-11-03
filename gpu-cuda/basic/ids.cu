#include <cstdio>
#include <stdlib.h>
#include <time.h>

__global__ void add(int *A, int c) {
    A[blockIdx.x * 8 + threadIdx.x] = c * threadIdx.x + blockIdx.x;
}

int main(void) {
    int A[16];
    int *dA;
    
    cudaMalloc((void**)&dA, sizeof(int) * 16);
    
    srand(time(NULL));
    add<<<2,8>>>(dA, rand() % 10);
    
    cudaMemcpy(A, dA, sizeof(int) * 16, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < 16; i++) {
        std::printf("%d ", A[i]);
    }
    std::printf("\n");
    cudaFree(dA);
    return 0;
}