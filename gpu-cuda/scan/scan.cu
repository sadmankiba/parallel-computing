#include <cstdio>

// scan call
// hillis-steele algorithm
/*
for j = 1 .. N
    forall k do in parallel 
        if k - 2^j >= 0
            x[out][k] = x[in][k] + x[in][k - 2^j]
        else
            x[out][k] = x[in][k]
        end
    end forall
    swap(in, out)
end 
*/

/* 
Multi block 
- 1st round: Block 0 - K sums threads_per_block elements. K = ceil(n / threads_per_block)
- 2nd round: Add the last element of each block to all the elements of the next blocks in Hills & Steele fashion
*/


__global__ void scanBlock(float *out, const float *in, unsigned int bn) {
    extern volatile __shared__ float temp[];

    unsigned int tid = threadIdx.x;
    unsigned int pout = 0, pin = 1;

    // if (tid == 0)
    //     std::printf("scanBlock - bn: %d, blockDim.x: %d\n", bn, blockDim.x);

    if (tid >= bn) 
        return;

    temp[tid] = in[tid];
    __syncthreads();

    for (int offset = 1; offset < bn; offset *= 2) {
        pout = 1 - pout;
        pin = 1 - pout;
        if (tid >= offset) {
            temp[pout * blockDim.x + tid] = temp[pin * blockDim.x + tid] + temp[pin * blockDim.x + tid - offset];
        } else {
            temp[pout * blockDim.x + tid] = temp[pin * blockDim.x + tid];
        }
        // std::printf("thread %d: temp[%d] = %f\n", tid, pout * blockDim.x + tid, temp[pout * blockDim.x + tid]);
        __syncthreads();
    }

    // if (tid == 0) {
    //     std::printf("Block %d shared mem\n", blockIdx.x);
    //     for(int i = 0; i < 2 * blockDim.x; i++)
    //         std::printf("temp[%d] = %f\n", i, temp[i]);
    // }
    out[tid] = temp[pout * blockDim.x + tid];
}

// block tid = out[tid * threads_per_block .. (tid + 1) * threads_per_block - 1]
// thread tid adds the last element of block (tid - offset) to all the elements of block tid    
__global__ void scanArray(float *out, unsigned int threads_per_block, unsigned int n) {
    unsigned int tid = threadIdx.x;

    // if (tid == 0)
    //     std::printf("scanArray - n: %d, blockDim.x: %d\n", n, blockDim.x);

    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        if (tid >= offset && tid < n) {
            for (int i = 0; i < threads_per_block; i++) {
                out[tid * threads_per_block + i] += out[(tid - offset + 1) * threads_per_block - 1];
            }
        }
    }
}

__host__ void scan(const float* input, float* output, unsigned int n, unsigned int threads_per_block) {
    unsigned int n_blocks = ceil(n / (float) threads_per_block);
    // std::printf("n_blocks: %d\n", n_blocks);
    for (unsigned int i = 0; i < n_blocks; i++) {
        int start = i * threads_per_block;
        int end = min(start + threads_per_block, n);
        scanBlock<<<n_blocks, threads_per_block, threads_per_block * sizeof(float) * 2>>>
            (output + start, input + start, end - start);
        cudaDeviceSynchronize();
    }

    scanArray<<<1, n_blocks>>>(output, threads_per_block, n);
    cudaDeviceSynchronize();
}