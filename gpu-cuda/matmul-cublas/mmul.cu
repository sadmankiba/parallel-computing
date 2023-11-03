#include "mmul.h"
#include <cuda.h>
#include <cublas_v2.h>
#include <cstdio>

// Uses a single cuBLAS call to perform the operation C := A B + C
// handle is a handle to an open cuBLAS instance
// A, B, and C are matrices with n rows and n columns stored in column-major
// NOTE: The cuBLAS call should be followed by a call to cudaDeviceSynchronize() for timing purposes
void mmul(cublasHandle_t handle, const float* A, const float* B, float* C, int n){
    cublasStatus_t stat;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, B, n, &beta, C, n);

    cudaDeviceSynchronize();

    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error: %d\n", stat);
    }
}