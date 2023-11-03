#define CUB_STDERR // print CUDA runtime errors to console
#include <stdio.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include "cub/util_debug.cuh"
using namespace cub;

// Caching allocator for device memory
CachingDeviceAllocator  g_allocator(true);  

int main(int argc, char** argv) {
    const int num_items = std::atoi(argv[1]);

    srand(time(NULL));
    
    // Set up host arrays
    float h_arr[num_items];

    for (int i = 0; i < num_items; ++i) {
        h_arr[i] = rand() * 2.0 / RAND_MAX  - 1.0;
    }

    // Set up device arrays
    float* d_arr = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_arr, sizeof(float) * num_items));

    // Initialize device input
    CubDebugExit(cudaMemcpy(d_arr, h_arr, sizeof(float) * num_items, cudaMemcpyHostToDevice));
    
    // Setup device output 
    float* d_sum = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_sum, sizeof(float) * 1));

    // Request and allocate temporary storage
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_arr, d_sum, num_items));    
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Do the actual reduce operation
    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_arr, d_sum, num_items);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float h_sum;
    CubDebugExit(cudaMemcpy(&h_sum, d_sum, sizeof(float) * 1, cudaMemcpyDeviceToHost));

    std::cout << h_sum << std::endl;
    std::cout << milliseconds << std::endl;
    
    // Cleanup
    if (d_arr) CubDebugExit(g_allocator.DeviceFree(d_arr));
    if (d_sum) CubDebugExit(g_allocator.DeviceFree(d_sum));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    
    return 0;
}
