#include "count.cuh"
#include <cstdlib>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <cuda.h>

int main (int argc, char* argv[]){
    int n = std::atoi(argv[1]);
    srand(time(0));

    // init arrays
    thrust::host_vector<int> v(n);
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist(0,500);
    thrust::generate(v.begin(), v.end(), [&]() {return dist(rng);});

    thrust::device_vector<int> d = v;
    thrust::device_vector<int> counts(n);
    thrust::device_vector<int> values(n);

    // timer setup
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // execute
    cudaEventRecord(start);
    count(d, values, counts);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << values.back() << std::endl;
    std::cout << counts.back() << std::endl;
    std::cout << ms << std::endl;

    return 0;
}