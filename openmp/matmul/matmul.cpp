#include <stdlib.h>
#include <omp.h>

// This function produces a parallel version of matrix multiplication C = A B using OpenMP. 
// The resulting C matrix should be stored in row-major representation. 
// Use mmul2 from HW02 task3. You may recycle the code from HW02.

// The matrices A, B, and C have dimension n by n and are represented as 1D arrays.

void mmul(const float* A, const float* B, float* C, const std::size_t n){
    #pragma omp parallel for
    for (unsigned int i = 0; i < n; i++) {
        bool setZero = true;
        for (unsigned int k = 0; k < n; k++){
            for (unsigned int j = 0; j < n; j++) {
                if (setZero) {
                    C[i * n + j] = 0;
                }
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
            setZero = false;
        }
    }
}
