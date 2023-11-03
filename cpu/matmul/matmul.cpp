#include "matmul.h"

void mmul1(const double* A, const double* B, double* C, const unsigned int n) {
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++){
            C[i * n + j] = 0;
            for (unsigned int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}
void mmul2(const double* A, const double* B, double* C, const unsigned int n) { 
    bool setZero;
    for (unsigned int i = 0; i < n; i++) {
        setZero = true;
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
void mmul3(const double* A, const double* B, double* C, const unsigned int n){
    bool setZero;
    for (unsigned int j = 0; j < n; j++) {
        setZero = true;
        for (unsigned int k = 0; k < n; k++){
            for (unsigned int i=0; i < n; i++) {
                if (setZero) {
                    C[i * n + j] = 0;
                }
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
            setZero = false;
        }
    }
}
void mmul4(const std::vector<double>& A, const std::vector<double>& B, double* C, const unsigned int n) { 
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++){
            C[i * n + j] = 0;
            for (unsigned int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}