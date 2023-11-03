

/* Each thread performs first level of reduction. 
Loads two elements from global memory and writes to shared memory */
__device__ void load_shared_add(float* sdata, float* g_in, int size) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    if (i < size)
        sdata[tid] = g_in[i] + (((i + blockDim.x) < size)? g_in[i + blockDim.x] : 0);
    else 
        sdata[tid] = 0;
    __syncthreads();
}

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];

    load_shared_add(sdata, g_idata, n);

    unsigned int tid = threadIdx.x;

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__host__ void reduce(float **input, float **output, unsigned int N,
                     unsigned int threads_per_block) {
    unsigned int rem;
    unsigned int n_blks;

    rem = N;
    do {
        n_blks = ceil(rem * 1.0 / (2.0 * threads_per_block));
        reduce_kernel<<<n_blks, threads_per_block, threads_per_block * sizeof(float)>>>
            (*input, *output, rem);
        cudaMemcpy(*input, *output, n_blks * sizeof(float), cudaMemcpyDeviceToDevice);
        rem = n_blks;
    } while (n_blks > 1);
}