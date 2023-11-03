#include <cstdio>

#define MAX_BLOCK_SIZE 1024

// Assumption: threads_per_block >= 2 * R + 1
__global__ void stencil_kernel(const float* image, const float* mask, 
    float* output, unsigned int n, unsigned int R) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float imgS[MAX_BLOCK_SIZE * 2]; // R cannot be used for size
    __shared__ float maskS[MAX_BLOCK_SIZE]; 

    if (i >= n)
        return;

    // load image elements directly corresponing to output
    imgS[R + threadIdx.x] = image[i];
    
    // load image flanks
    if (threadIdx.x < R) {
        imgS[threadIdx.x] = ((i - (int) R) < 0)? 1 : image[i - R];
    }
    if (threadIdx.x >= ((n % blockDim.x == 0)? blockDim.x - R : n % blockDim.x - R)) {
        imgS[R + threadIdx.x + R] = ((i + R) >= n)? 1 : image[i + R];
    }

    //load mask
    if (threadIdx.x < 2 * R + 1) {
        maskS[threadIdx.x] = mask[threadIdx.x];
    }
    __syncthreads();
    
    // calculate output
    float result = 0.0f;
    int j = (int) (-1 * R);
    for (; j <= (int) R; j++) {
        result += imgS[R + threadIdx.x + j] * 1.0 * maskS[j + R];
    }
    output[i] = result;
}

__host__ void stencil(const float* image, const float* mask,
                      float* output, unsigned int n, unsigned int R,
                      unsigned int threads_per_block) {
    unsigned int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    stencil_kernel<<<blocks, threads_per_block>>>(image, mask, output, n, R);
}

__global__ void stencil_kernel_full(const float* image, const float* mask, 
    float* output, unsigned int n, unsigned int R) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float result = 0.f;
        for (int j = -R; j <= R; j++) {
            if (i + j >= 0 && i + j < n) {
                result += image[i + j] * mask[j + R];
            } else {
                result += mask[j + R];
            }
        }
        output[i] = result;
    }
}

__host__ void stencil_full(const float* image, const float* mask,
                      float* output, unsigned int n, unsigned int R,
                      unsigned int threads_per_block) {
    unsigned int blocks = (n + threads_per_block - 1) / threads_per_block;
    float *d_image, *d_mask, *d_output;

    // Allocate memory for each array on GPU
    cudaMalloc((void **)&d_image, n * sizeof(float));
    cudaMalloc((void **)&d_mask, 2 * R + 1);
    cudaMalloc((void **)&d_output, n * sizeof(float));

    // Copy host arrays to device
    cudaMemcpy(d_image, image, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, 2 * R + 1, cudaMemcpyHostToDevice);

    stencil_kernel<<<blocks, threads_per_block>>>(d_image, d_mask, d_output, n, R);
    cudaDeviceSynchronize();

    std::printf("Device output: \n");
    for (unsigned int i = 0; i < n; i++) {
        std::printf("%f ", d_output[i]);
    }
    std::printf("\n");
    // Copy result back to host
    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_mask);
    cudaFree(d_output);
}