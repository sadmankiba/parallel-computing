#include "utils.cuh"

void device_vector_print(const thrust::device_vector<float>& d_vec, const std::string& name) {
    std::cout << name << ": ";
    thrust::copy(d_vec.begin(), d_vec.end(), std::ostream_iterator<float>(std::cout, ", "));
    std::cout << std::endl;
}

__global__ void kaiming_init_kernel(float *weights, int size, curandState* states, float sdev) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        curand_init(1234, idx, 0, &states[idx]);
        weights[idx] = curand_normal(&states[idx]) * sdev;
    }
}

void kaiming_init(thrust::device_vector<float> &weights, int nIn, int nOut){
    float sdev = sqrt(2/float(nIn));
    float* weights_ptr = thrust::raw_pointer_cast(weights.data());
    int size = nIn*nOut;

    thrust::device_vector<curandState> states(size);
    curandState *states_ptr = thrust::raw_pointer_cast(states.data());

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    kaiming_init_kernel<<<numBlocks, blockSize>>>(weights_ptr, size, states_ptr, sdev);

    CUDA_POST_KERNEL_CHECK;
}

__global__ void matmulKernel(float* A, float* B, float* C, int n, int p, int q) {
    
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float val = 0;

    for (int t = 0; t < (TILE_SIZE + p - 1)/TILE_SIZE; t++) {
        if (t*TILE_SIZE + tx < p && row < n)
            tile_A[ty][tx] = A[row*p + t*TILE_SIZE + tx];
        else
            tile_A[ty][tx] = 0.0;

        if (t*TILE_SIZE + ty < p && col < q)
            tile_B[ty][tx] = B[(t*TILE_SIZE + ty)*q + col];
        else
            tile_B[ty][tx] = 0.0;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++)
            val += tile_A[ty][i] * tile_B[i][tx];

        __syncthreads();
    }

    if (row < n && col < q)
        C[row*q + col] = val;
}

void matmul(thrust::device_vector<float> &A, thrust::device_vector<float> &B, thrust::device_vector<float> &C, int n, int p, int q) {
    float* A_ptr = thrust::raw_pointer_cast(A.data());
    float* B_ptr = thrust::raw_pointer_cast(B.data());
    float* C_ptr = thrust::raw_pointer_cast(C.data());

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((q + dimBlock.x - 1) / dimBlock.x, (n + dimBlock.y - 1) / dimBlock.y);

    matmulKernel<<<dimGrid, dimBlock>>>(A_ptr, B_ptr, C_ptr, n, p, q);

    CUDA_POST_KERNEL_CHECK;
}

__global__ void addBiasKernel(float* A, float* B, int n, int p) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        for (int i = 0; i < p; i++)
            A[idx*p + i] += B[i];
    }
}

void addBias(thrust::device_vector<float> &A, thrust::device_vector<float> &B, int n, int p) {
    float* A_ptr = thrust::raw_pointer_cast(A.data());
    float* B_ptr = thrust::raw_pointer_cast(B.data());

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    addBiasKernel<<<numBlocks, blockSize>>>(A_ptr, B_ptr, n, p);

    CUDA_POST_KERNEL_CHECK;
}

__global__ void columnSumKernel(float* A, float* B, int n, int p) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < p) {
        float sum = 0;
        for (int i = 0; i < n; i++)
            sum += A[i*p + idx];
        B[idx] = sum;
    }
}

/*
    A: n x p (input)
    B: 1 x p (output)
*/
void columnSum(thrust::device_vector<float> &A, thrust::device_vector<float> &B, int n, int p) {
    float* A_ptr = thrust::raw_pointer_cast(A.data());
    float* B_ptr = thrust::raw_pointer_cast(B.data());
    
    int blockSize = 256;
    int numBlocks = (p + blockSize - 1) / blockSize;

    columnSumKernel<<<numBlocks, blockSize>>>(A_ptr, B_ptr, n, p);

    CUDA_POST_KERNEL_CHECK;
}

void rowSum(thrust::device_vector<float> &A, thrust::device_vector<float> &B, int n, int p) {
    float* A_ptr = thrust::raw_pointer_cast(A.data());
    float* B_ptr = thrust::raw_pointer_cast(B.data());

    thrust::device_vector<float> A_transpose(n*p);
    float* A_transpose_ptr = thrust::raw_pointer_cast(A_transpose.data());

    transpose(A, A_transpose, n, p);

    columnSum(A_transpose, B, p, n);
}

struct rmsPropUpdateFunctor {
    // state
    const float lr;
    const float beta;
    
    // constructor
    rmsPropUpdateFunctor(float lr, float beta) : lr(lr), beta(beta) {}

    // call operator
    __host__ __device__ void operator()(const thrust::tuple<float&, float&, float&> &t) {
        float &param = thrust::get<0>(t);
        float &grad = thrust::get<1>(t);
        float &sqrGrad = thrust::get<2>(t);

        float gradSquare = grad*grad;
        float meanSquareGrad = beta*sqrGrad + (1 - beta)*gradSquare;
        sqrGrad = meanSquareGrad;
        param -= lr * grad/(sqrtf(sqrGrad) + RMS_EPSILON);
    }
};

void rmsPropUpdate(thrust::device_vector<float> &params, thrust::device_vector<float> &grads, thrust::device_vector<float> &sqrGrad, float lr, float beta) {
    CHECK_EQ(params.size(), grads.size(), "params and grads should have same size");
    CHECK_EQ(params.size(), sqrGrad.size(), "params and sqrGrad should have same size");
    
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(params.begin(), grads.begin(), sqrGrad.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(params.end(), grads.end(), sqrGrad.end())),
                     rmsPropUpdateFunctor(lr, beta));
}

/*
    A: n x p (input)
    B: p x n (transpose of input)
*/
__global__ void transposeKernel(float* A, float* B, int n, int p) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // 1 added for paddin

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    // load A into shared memory
    int col = bx * TILE_SIZE + tx;
    int row = by * TILE_SIZE + ty;
    
    for (int i = 0; i < TILE_SIZE; i += blockDim.y) {
        if (row + i < n && col < p) {
            tile[ty + i][tx] = A[(row + i) * p + col];
        }
    }
    __syncthreads();

    // store B from shared memory
    col = by * TILE_SIZE + tx;
    row = bx * TILE_SIZE + ty;

    for (int i = 0; i < TILE_SIZE; i += blockDim.y) {
        if (row < p && col + i < n) {
            B[(row + i) * n + col] = tile[tx][ty + i];
        }
    }
}

/*
    A: n x p (input)
    B: p x n (transpose of input)
*/
void transpose(thrust::device_vector<float> &A, thrust::device_vector<float> &B, int n, int p) {
    float* A_ptr = thrust::raw_pointer_cast(A.data());
    float* B_ptr = thrust::raw_pointer_cast(B.data());

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((p + dimBlock.x - 1) / dimBlock.x, (n + dimBlock.y - 1) / dimBlock.y);

    transposeKernel<<<dimGrid, dimBlock>>>(A_ptr, B_ptr, n, p);

    CUDA_POST_KERNEL_CHECK;
}

