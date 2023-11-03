#include <iostream>
#include <ctime>
#include <cmath>
#include "reduce.cuh"

int main(int argc, char **argv) {
    const unsigned int size = atoi(argv[1]);
    const unsigned int n_threads = atoi(argv[2]);  // threads per block
    float *input;
    float output;
    float *d_input, *d_output;

    srand(time(NULL));
    
    input = (float *) malloc(size * sizeof(float));

    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, ceil(size * 1.0 / n_threads) * sizeof(float));

    for (unsigned int i = 0; i < size; i++) {
        input[i] = rand() * 2.0 / RAND_MAX - 1.0;
    }

    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEvent_t start;
    cudaEvent_t stop;
    float ms;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    reduce(&d_input, &d_output, size, n_threads);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(&output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << output << std::endl;
    std::cout << ms << std::endl;
}