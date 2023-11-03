#include "matmul.cuh"
#include <cuda.h>

// Computes the matrix product of A and B, storing the result in C.
// Each thread should compute _one_ element of output.
// Does not use shared memory for this problem.
//
// A, B, and C are row major representations of nxn matrices in device memory.
//
// Assumptions:
// - 1D kernel configuration
__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n*n){
        int row = i/n;
        int col = i%n;
        float output = 0.0;

        for(int j=0; j<n; j++)
            output += A[row*n + j]*B[j*n + col];

        C[i] = output;
    }

}

// Makes one call to matmul_kernel with threads_per_block threads per block.
// You can consider following the kernel call with cudaDeviceSynchronize (but if you use 
// cudaEventSynchronize to time it, that call serves the same purpose as cudaDeviceSynchronize).
void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block){
    dim3 dimBlock(threads_per_block);
    dim3 dimGrid((n*n + threads_per_block - 1)/threads_per_block);
    matmul_kernel<<<dimGrid, dimBlock>>>(A,B,C,n);
}