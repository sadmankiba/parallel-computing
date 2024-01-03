#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <iostream>
#include <curand_kernel.h>

#define SQUARE_GRAD_DEFAULT 0.001
#define RMS_EPSILON 1e-10
#define TILE_SIZE 16

#define CHECK_EQ(val1, val2, message)                              \
  do {                                                             \
    if (val1 != val2) {                                            \
      std::cerr << __FILE__ << "(" << __LINE__ << "): " << message \
                << std::endl;                                      \
      exit(1);                                                     \
    }                                                              \
  } while (0)

#define CUDA_CHECK(condition)                                \
  do {                                                       \
    cudaError_t error = condition;                           \
    CHECK_EQ(error, cudaSuccess, cudaGetErrorString(error)); \
  } while (0)

#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

void device_vector_print(const thrust::device_vector<float>& d_vec, const std::string& name = "");

void kaiming_init(thrust::device_vector<float> &weights, int nIn, int nOut);

void matmul(thrust::device_vector<float> &A, thrust::device_vector<float> &B, thrust::device_vector<float> &C, int n, int p, int q);

void addBias(thrust::device_vector<float> &A, thrust::device_vector<float> &B, int n, int p);

void columnSum(thrust::device_vector<float> &A, thrust::device_vector<float> &B, int n, int p);

void rowSum(thrust::device_vector<float> &A, thrust::device_vector<float> &B, int n, int p);

void rmsPropUpdate(thrust::device_vector<float> &params, thrust::device_vector<float> &grads, thrust::device_vector<float> &sqrGrad, float lr, float beta);

void transpose(thrust::device_vector<float> &A, thrust::device_vector<float> &B, int n, int p);