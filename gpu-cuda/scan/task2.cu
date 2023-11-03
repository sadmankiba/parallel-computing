#include <iostream>
#include <ctime>
#include <cmath>
#include "scan.cuh"

int main(int argc, char **argv) {
    const unsigned int size = atoi(argv[1]);
    const unsigned int n_threads = atoi(argv[2]);  // threads per block
    float *in, *out;

    srand(time(NULL));
    // srand(0);

    cudaMallocManaged(&in, size * sizeof(float));
    cudaMallocManaged(&out, size * sizeof(float));

    for (unsigned int i = 0; i < size; i++) {
        in[i] = rand() * 2.0 / RAND_MAX - 1.0;
        // in[i] = rand() % 3;
    }

    // std::cout << "in\n";
    // for (unsigned int i = 0; i < size; i++) {
    //     std::cout << in[i] << " ";
    // }
    // std::cout << std::endl;
    
    cudaEvent_t start;
    cudaEvent_t stop;
    float ms;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    scan(in, out, size, n_threads);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    // std::cout << "out\n";
    // for (unsigned int i = 0; i < size; i++) {
    //     std::cout << out[i] << " ";
    // }
    // std::cout << std::endl;

    std::cout << out[size - 1] << std::endl;
    std::cout << ms << std::endl;
}