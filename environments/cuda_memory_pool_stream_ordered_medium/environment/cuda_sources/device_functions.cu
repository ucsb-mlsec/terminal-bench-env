__device__ float computeDistance(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx * dx + dy * dy);
}

__device__ void matrixMultiplyElement(float* result, const float* A, const float* B, 
                                       int row, int col, int K) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * col];
    }
    result[row * col] = sum;
}

__global__ void vectorAdd(float* c, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void matrixTranspose(float* output, const float* input, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < cols && y < rows) {
        output[x * rows + y] = input[y * cols + x];
    }
}

__device__ float applyActivation(float value, int activationType) {
    switch (activationType) {
        case 0: // ReLU
            return fmaxf(0.0f, value);
        case 1: // Sigmoid
            return 1.0f / (1.0f + expf(-value));
        case 2: // Tanh
            return tanhf(value);
        default:
            return value;
    }
}

__global__ void reduceSum(float* output, const float* input, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}