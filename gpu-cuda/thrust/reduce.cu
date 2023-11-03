#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <cuda_runtime.h>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <vector length>" << std::endl;
        return 1;
    }

    const int n = std::atoi(argv[1]);

    // Create a host vector of length n and fill it with random float numbers in the range [-1.0, 1.0]
    thrust::host_vector<float> h_vec(n);
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    thrust::generate(h_vec.begin(), h_vec.end(), [&](){ return dist(rng); });

    // Copy the host vector to a device vector
    thrust::device_vector<float> d_vec(n);
    thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());

    // Perform a reduction on the device vector and print the result
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    float result = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>());
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << result << std::endl;
    std::cout << milliseconds << std::endl;

    return 0;
}
