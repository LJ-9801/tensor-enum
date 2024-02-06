#include "distribution.cuh"


__global__ void setup_curandState(curandState *state, unsigned long long seed){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

template <>
__global__ void generate_uniform_kernel(curandState *state,
                                size_t size, float low, float high,
                                float *result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_state = state[idx];
    if (idx < size)
    {
        result[idx] = curand_uniform(&local_state) * (high - low) + low;
    }   
}

template <>
__global__ void generate_uniform_kernel(curandState *state,
                                size_t size, double low, double high,
                                double *result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_state = state[idx];
    if (idx < size)
    {
        result[idx] = curand_uniform_double(&local_state) * (high - low) + low;
    }                    
}

template <>
__global__ void generate_multi_uniform_kernel(curandState *state, size_t size, float *low, float *high, float *result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_state = state[idx];
    if (idx < size)
    {
       result[idx] = curand_uniform(&local_state) * (high[idx] - low[idx]) + low[idx]; 
    }
}

template <>
__global__ void generate_multi_uniform_kernel(curandState *state, size_t size, double *low, double *high, double *result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_state = state[idx];
    if (idx < size)
    {
         result[idx] = curand_uniform_double(&local_state) * (high[idx] - low[idx]) + low[idx]; 
    }
}

template <>
__global__ void generate_normal_kernel(curandState *state, size_t size, float mean, float stddev, float *result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_state = state[idx];
    if (idx < size)
    {
        result[idx] = curand_normal(&local_state) * stddev + mean;
    }
}

template <>
__global__ void generate_normal_kernel(curandState *state, size_t size, double mean, double stddev, double *result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_state = state[idx];
    if (idx < size)
    {
        result[idx] = curand_normal_double(&local_state) * stddev + mean;
    }
}

template <>
__global__ void generate_multi_normal_kernel(curandState *state, size_t size, float *mean, float *stddev, float *result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_state = state[idx];
    if (idx < size)
    {
        result[idx] = curand_normal(&local_state) * stddev[idx] + mean[idx];
    }
}

template <>
__global__ void generate_multi_normal_kernel(curandState *state, size_t size, double *mean, double *stddev, double *result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_state = state[idx];
    if (idx < size)
    {
        result[idx] = curand_normal_double(&local_state) * stddev[idx] + mean[idx];
    }
}


template <>
void generateUniform(float **data, size_t size, unsigned long long seed, float low, float high){
    size_t blocksPerGrid = (size + CURAND_MAX_THREADS - 1) / CURAND_MAX_THREADS;
    curandState *devStates;
    CHECK_CUDA(cudaMalloc((void **)&devStates, size * sizeof(curandState)));
    setup_curandState<<<blocksPerGrid, CURAND_MAX_THREADS>>>(devStates, seed);
    generate_uniform_kernel<<<blocksPerGrid, CURAND_MAX_THREADS>>>(devStates, size, low, high, *data);
    CHECK_CUDA(cudaFree(devStates));
}

template <>
void generateUniform(double **data, size_t size, unsigned long long seed, double low, double high){
    size_t blocksPerGrid = (size + CURAND_MAX_THREADS - 1) / CURAND_MAX_THREADS;
    curandState *devStates;
    CHECK_CUDA(cudaMalloc((void **)&devStates, size * sizeof(curandState)));
    setup_curandState<<<blocksPerGrid, CURAND_MAX_THREADS>>>(devStates, seed);
    generate_uniform_kernel<<<blocksPerGrid, CURAND_MAX_THREADS>>>(devStates, size, low, high, *data);
    CHECK_CUDA(cudaFree(devStates));
}

template <>
void generateMultiUniform(float **data, size_t size, unsigned long long seed, float *low, float *high){
    size_t blocksPerGrid = (size + CURAND_MAX_THREADS - 1) / CURAND_MAX_THREADS;
    curandState *devStates;
    CHECK_CUDA(cudaMalloc((void **)&devStates, size * sizeof(curandState)));
    setup_curandState<<<blocksPerGrid, CURAND_MAX_THREADS>>>(devStates, seed);
    generate_multi_uniform_kernel<<<blocksPerGrid, CURAND_MAX_THREADS>>>(devStates, size, low, high, *data);
    CHECK_CUDA(cudaFree(devStates));
}

template <>
void generateMultiUniform(double **data, size_t size, unsigned long long seed, double *low, double *high){
    size_t blocksPerGrid = (size + CURAND_MAX_THREADS - 1) / CURAND_MAX_THREADS;
    curandState *devStates;
    CHECK_CUDA(cudaMalloc((void **)&devStates, size * sizeof(curandState)));
    setup_curandState<<<blocksPerGrid, CURAND_MAX_THREADS>>>(devStates, seed);
    generate_multi_uniform_kernel<<<blocksPerGrid, CURAND_MAX_THREADS>>>(devStates, size, low, high, *data);
    CHECK_CUDA(cudaFree(devStates));
}

template <>
void generateNormal(float **data, size_t size, unsigned long long seed, float mean, float stddev){
    size_t blocksPerGrid = (size + CURAND_MAX_THREADS - 1) / CURAND_MAX_THREADS;
    curandState *devStates;
    CHECK_CUDA(cudaMalloc((void **)&devStates, size * sizeof(curandState)));
    setup_curandState<<<blocksPerGrid, CURAND_MAX_THREADS>>>(devStates, seed);
    generate_normal_kernel<<<blocksPerGrid, CURAND_MAX_THREADS>>>(devStates, size, mean, stddev, *data);
    CHECK_CUDA(cudaFree(devStates));
}

template <>
void generateNormal(double **data, size_t size, unsigned long long seed, double mean, double stddev){
    size_t blocksPerGrid = (size + CURAND_MAX_THREADS - 1) / CURAND_MAX_THREADS;
    curandState *devStates;
    CHECK_CUDA(cudaMalloc((void **)&devStates, size * sizeof(curandState)));
    setup_curandState<<<blocksPerGrid, CURAND_MAX_THREADS>>>(devStates, seed);
    generate_normal_kernel<<<blocksPerGrid, CURAND_MAX_THREADS>>>(devStates, size, mean, stddev, *data);
    CHECK_CUDA(cudaFree(devStates));
}

template <>
void generateMultiNormal(float **data, size_t size, unsigned long long seed, float *mean, float *stddev){
    size_t blocksPerGrid = (size + CURAND_MAX_THREADS - 1) / CURAND_MAX_THREADS;
    curandState *devStates;
    CHECK_CUDA(cudaMalloc((void **)&devStates, size * sizeof(curandState)));
    setup_curandState<<<blocksPerGrid, CURAND_MAX_THREADS>>>(devStates, seed);
    generate_multi_normal_kernel<<<blocksPerGrid, CURAND_MAX_THREADS>>>(devStates, size, mean, stddev, *data);
    CHECK_CUDA(cudaFree(devStates));
}

template <>
void generateMultiNormal(double **data, size_t size, unsigned long long seed, double *mean, double *stddev){
    size_t blocksPerGrid = (size + CURAND_MAX_THREADS - 1) / CURAND_MAX_THREADS;
    curandState *devStates;
    CHECK_CUDA(cudaMalloc((void **)&devStates, size * sizeof(curandState)));
    setup_curandState<<<blocksPerGrid, CURAND_MAX_THREADS>>>(devStates, seed);
    generate_multi_normal_kernel<<<blocksPerGrid, CURAND_MAX_THREADS>>>(devStates, size, mean, stddev, *data);
    CHECK_CUDA(cudaFree(devStates));
}
