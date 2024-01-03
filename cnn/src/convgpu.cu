#include <cuda.h>
#include <cstdio>
#include <iostream>
#include <string>

#define DEBUG 0

/*
 * CUDA kernel for convolution
 * 1 block. Each thread -> 1 output element.
 * blockDim.x = blockDim.y = m - n + 1
 * 
 * Arguments:
 *  input: m x m input matrix, row-major order
 *  filter: n x n filter matrix, row-major order (n < m)
 *  output: m - n + 1 x m - n + 1 output matrix, row-major order
 *  input_size: m
 *  filter_size: n
 */
__global__ void conv_kernel(const float *input, const float *filter, float *output, unsigned int input_size, unsigned int filter_size) {
    unsigned int row_out = threadIdx.y;
    unsigned int col_out = threadIdx.x;
    unsigned int output_size = input_size - filter_size + 1;
    unsigned int tid = row_out * output_size + col_out;

    if (DEBUG) {
        printf("row_out %u, col_out %u, output_size %u, tid: %u\n", row_out, col_out, output_size, tid);
    }

    if (row_out * output_size + col_out >= output_size * output_size)
        return;

    if (filter_size > input_size || row_out >= output_size || col_out >= output_size)
        return;
     
    float result = 0.0f;
    for (unsigned int i = 0; i < filter_size; i++) {
        for (unsigned int j = 0; j < filter_size; j++) {
            result += input[(row_out + i) * input_size + (col_out + j)] * filter[i * filter_size + j];
        }
    }
    if (DEBUG) {
        printf("result: %f\n", result);
    }

    output[row_out * output_size + col_out] = result;
}

/*
 * CUDA kernel for convolution with shared memory
 * 1 block. 1 thread for each input element.
 * SM size = input_size * input_size * sizeof(float) + filter_size * filter_size * sizeof(float)
 * Arguments same as conv_kernel
 */
__global__ void conv_kernel_sm(const float *input, const float *filter, float *output, unsigned int input_size, unsigned int filter_size) {
    unsigned int output_size = input_size - filter_size + 1;
    unsigned int tid = threadIdx.y * input_size + threadIdx.x;

    if (DEBUG) {
        printf("threadIdx.x %u, threadIdx.y %u, output_size %u, tid: %u\n", threadIdx.x, threadIdx.y, output_size, tid);
    }
    /* 
     * Each thread brings in one input and one filter element from global memory  
     */
    extern __shared__ float shared_input_filter[]; 
    
    shared_input_filter[threadIdx.y * input_size + threadIdx.x] = input[(threadIdx.y * input_size + threadIdx.x)];
    shared_input_filter[input_size * input_size + threadIdx.y * filter_size + threadIdx.x] = filter[threadIdx.y * filter_size + threadIdx.x];
    
    __syncthreads();

    /* If threadId < output_size, the thread calculates an output element */
    if (threadIdx.y * output_size + threadIdx.x >= output_size * output_size)
        return;

    if (filter_size > input_size || threadIdx.y >= output_size || threadIdx.x >= output_size)
        return;

    float result = 0.0f;
    for (unsigned int i = 0; i < filter_size; i++) {
        for (unsigned int j = 0; j < filter_size; j++) {
            result += shared_input_filter[(threadIdx.y + i) * input_size + (threadIdx.x + j)] 
                * shared_input_filter[input_size * input_size + i * filter_size + j];
        }
    }
    if (DEBUG) {
        printf("result: %f\n", result);
    }

    output[threadIdx.y * output_size + threadIdx.x] = result;
}

/*
 * method = "gpu-naive", "gpu-sm"
 */
int conv_gpu(float * input_h, float * filter_h, float * output_h, unsigned int input_size, unsigned int filter_size, char *method) {
    float *input, *filter, *output;

    unsigned int output_size = input_size - filter_size + 1;

    // Allocate memory
    cudaMallocManaged(&input, input_size * input_size * sizeof(float));
    cudaMallocManaged(&filter, filter_size * filter_size * sizeof(float));
    cudaMallocManaged(&output, output_size * output_size * sizeof(float));

    for (int i = 0; i < input_size * input_size; i++) {
        input[i] = input_h[i];
    }

    for (int i = 0; i < filter_size * filter_size; i++) {
        filter[i] = filter_h[i];
    }
    
    if (DEBUG)
	    std::cout << method << std::endl;

    if (strcmp(method, "gpu-sm") == 0) {	
conv_kernel_sm<<<1, dim3(input_size, input_size), 
            (input_size * input_size + filter_size * filter_size) * sizeof(float)>>>
            (input, filter, output, input_size, filter_size);
    } else if (strcmp(method, "gpu-naive") == 0) {
	      conv_kernel<<<1, dim3(output_size, output_size)>>>(input, filter, output, input_size, filter_size);
    } else {
        std::cerr << "Invalid method" << std::endl;
        return 1;
    }

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    for(int i = 0; i < output_size * output_size; i++) {
        output_h[i] = output[i];
    }

    /*
    Good print point
    */
    if (DEBUG) {
        std::cout << "GPU conv input: " << std::endl;
        for (int i = 0; i < input_size * input_size; i++) {
            std::cout << input[i] << " ";
            if ((i + 1) % input_size == 0) {
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;

        std::cout << "GPU conv filter: " << std::endl;
        for (int i = 0; i < filter_size * filter_size; i++) {
            std::cout << filter[i] << " ";
            if ((i + 1) % filter_size == 0) {
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;

        std::cout << "GPU conv output: " << std::endl;
        for (int i = 0; i < output_size * output_size; i++) {
            std::cout << output[i] << " ";
            if ((i + 1) % output_size == 0) {
                std::cout << std::endl;
            }
        }

        std::cout << std::endl;

        std::cout << "GPU conv output_h: " << std::endl;
        for (int i = 0; i < output_size * output_size; i++) {
            std::cout << output_h[i] << " ";
            if ((i + 1) % output_size == 0) {
                std::cout << std::endl;
            }
        }
    }

    // Free memory
    cudaFree(input);
    cudaFree(filter);
    cudaFree(output);

    return 0;
}


