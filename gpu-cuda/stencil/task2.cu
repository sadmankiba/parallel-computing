#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include "stencil.cuh"

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cout << "Usage: ./task2 <n> <R> <threads_per_block>\n";
        exit(1);
    }
    int n = atoi(argv[1]);
    int R = atoi(argv[2]);
    int threads_per_block = atoi(argv[3]);
    
    float *image, *mask, *output;
    float *d_image, *d_mask, *d_output;

    // Allocate memory for each array on host
    image = (float *) malloc(n * sizeof(float));
    mask = (float *) malloc((2 * R + 1) * sizeof(float));
    output = (float *) malloc(n * sizeof(float));

    // Allocate memory for each array on GPU
    cudaMalloc((void **)&d_image, n * sizeof(float));
    cudaMalloc((void **)&d_mask, (2 * R + 1) * sizeof(float));
    cudaMalloc((void **)&d_output, n * sizeof(float));

    srand(time(NULL));

    // Set image and mask array with random values in range [-1 ,1]
    for (int i = 0; i < n; i++){
        image[i] = (float) rand() / RAND_MAX * 2.0 - 1.0;
    }

    for (int i = 0; i < (2 * R + 1); i++) {
        mask[i] = (float) rand() / RAND_MAX * 2.0 - 1.0;
    }

    // Copy host arrays to device
    cudaMemcpy(d_image, image, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, (2 * R + 1) * sizeof(float), cudaMemcpyHostToDevice);
    
    // timer setup
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Run kernel
    cudaEventRecord(start);
    stencil(d_image, d_mask, d_output, n, R, threads_per_block);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << output[n-1] << std::endl;
    std::cout << ms << std::endl;

    // Cleanup
    free(image);
    free(mask);
    free(output);

    cudaFree(d_image);
    cudaFree(d_mask);
    cudaFree(d_output);

    return 0;
}