#include "matmul.cuh"
#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <string>
#include <cstdio>


// You should implement Tiled Matrix Multiplication discussed in class
// Computes the matrix product C = AB by making 'one' call to 'matmul_kernel'.
// A, B, and C are row-major representations of nxn matrices in managed memory.
// Configures the kernel call using a 2D configuration with blocks of dimensions
// block_dim x block_dim. The function should end in a call to
// cudaDeviceSynchronize for timing purposes.

// Use template to formulate your answer

// template <typename T>
// __device__ T* shared_memory_proxy(unsigned int p, unsigned int q)
// {
//     extern __shared__ unsigned char memory[p][q];
//     return reinterpret_cast<T*>(memory);
// }

using namespace std;

template <typename T>
__device__ T* shared_memory_proxy()
{
    extern __shared__ unsigned char memory[];
    return reinterpret_cast<T*>(memory);
}


template<typename TYPE>
__global__ void matmul_kernel(const TYPE* A, const TYPE* B, TYPE* C, unsigned int n){
    unsigned int bdim = blockDim.x;

    auto smem = shared_memory_proxy<TYPE>();
    TYPE *shared_A = smem;
    TYPE *shared_B = (TYPE *)&shared_A[(bdim * bdim)];

    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int tid = ty * bdim + tx;

    unsigned int aBegin = n * bdim * by;
    unsigned int bBegin = bdim * bx;
    int aEnd = aBegin + n - 1;
    int aStep = bdim;
    int bStep = bdim * n;
    unsigned int cIdx = aBegin + ty * n + bBegin + tx;

    TYPE Csum = 0;
    for (unsigned int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        shared_A[tid] = (a + ty * n + tx) < (n * n)? A[a + ty * n + tx]: 0;
        shared_B[tid] = (b + ty * n + tx) < (n * n)? B[b + ty * n + tx]: 0;
        __syncthreads();

        if (cIdx < (n * n)) {
            for (unsigned int k = 0; k < bdim; k++) {
                Csum += shared_A[ty * bdim + k] * shared_B[k * bdim + tx];
            }
        }
        __syncthreads();
    }
    
    if (cIdx < (n * n))
        C[cIdx] = Csum;
}

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim){
    if (block_dim > 32) block_dim = 32;

    unsigned int blockNum = (n + block_dim - 1) / block_dim;
    dim3 dimGrid(blockNum, blockNum);
    dim3 dimBlock(block_dim, block_dim);
    
    matmul_kernel<int><<<dimGrid, dimBlock, 2 * block_dim * block_dim * sizeof(int)>>>(A, B, C, n);
    cudaDeviceSynchronize();
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim){
    if (block_dim > 32) block_dim = 32;

    unsigned int blockNum = (n + block_dim - 1) / block_dim;
    dim3 dimGrid(blockNum, blockNum);
    dim3 dimBlock(block_dim, block_dim);

    matmul_kernel<float><<< dimGrid, dimBlock, 2 * block_dim * block_dim * sizeof(float)>>>(A, B, C, n);
    cudaDeviceSynchronize();
}

__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim){
    if (block_dim > 32) block_dim = 32;

    unsigned int blockNum = (n + block_dim - 1) / block_dim;
    dim3 dimGrid(blockNum, blockNum);
    dim3 dimBlock(block_dim, block_dim);
    
    matmul_kernel<double><<< dimGrid, dimBlock,  2 * block_dim * block_dim * sizeof(double)>>>(A, B, C, n);
    cudaDeviceSynchronize();
}