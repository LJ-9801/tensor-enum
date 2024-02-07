#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <assert.h>
#include "common.cuh"


#define CURAND_MAX_THREADS 256


#define assert_float_type(T) \
    static_assert(std::is_same<T, float32>::value || std::is_same<T, float64>::value, \
                    "Only float32 and float64 are supported");

__global__ void setup_curandState(curandState *state, unsigned long long seed){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

template <typename T>
__global__ void generate_uniform_kernel(curandState *state, size_t size, T low, T high, T *result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_state = state[idx];
    if (idx < size)
    {   
        if(std::is_same<T, float32>::value)
            result[idx] = curand_uniform(&local_state) * (high - low) + low;
        else if(std::is_same<T, float64>::value)
            result[idx] = curand_uniform_double(&local_state) * (high - low) + low;
        else
            assert_float_type(T)
    }  
}


template <typename T>
__global__ void generate_multi_uniform_kernel(curandState *state, size_t size, T *lows, T *highs, T *result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_state = state[idx];

    if (idx < size)
    {   
        if(std::is_same<T, float32>::value)
            result[idx] = curand_uniform(&local_state) * (highs[idx] - lows[idx]) + lows[idx];
        else if(std::is_same<T, float64>::value)
            result[idx] = curand_uniform_double(&local_state) * (highs[idx] - lows[idx]) + lows[idx];
        else
            assert_float_type(T)
    } 
}


template <typename T>
__global__ void generate_normal_kernel(curandState *state, size_t size, T mean, T stddev, T *result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_state = state[idx];
    if (idx < size)
    {
        if(std::is_same<T, float32>::value)
            result[idx] = curand_normal(&local_state) * stddev + mean;
        else if(std::is_same<T, float64>::value)
            result[idx] = curand_normal_double(&local_state) * stddev + mean;
        else
            assert_float_type(T)
        
    } 
}

template <typename T>
__global__ void generate_multi_normal_kernel(curandState *state, size_t size, T *mean, T *stddev, T *result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_state = state[idx];
    if (idx < size)
    {
        if(std::is_same<T, float32>::value)
            result[idx] = curand_normal(&local_state) * stddev[idx] + mean[idx];
        else if(std::is_same<T, float64>::value)
            result[idx] = curand_normal_double(&local_state) * stddev[idx] + mean[idx];
        else
            assert_float_type(T)
    } 

}

//https://github.com/pytorch/pytorch/blob/77721ee318d6785010144aa4569efb98199e7162/torch/nn/init.py#L278
template <typename T>
__global__ void generate_kaiming_uniform_kernel(curandState *state, size_t size, T* data){
}

template <typename T>
inline void setup_random_generator(cudaStream_t *stream, curandState **devStates, T** data, size_t size){
    assert(*data == NULL && "Data should be NULL");
    CHECK_CUDA(cudaStreamCreate(stream));
    CHECK_CUDA(cudaMallocAsync(devStates, size * sizeof(curandState), *stream));
    CHECK_CUDA(cudaMallocAsync(data, size * sizeof(T), *stream));
}


inline void finish_random_generator(cudaStream_t *stream, curandState **devState){
    CHECK_CUDA(cudaStreamSynchronize(*stream)); 
    CHECK_CUDA(cudaFree(*devState));
    CHECK_CUDA(cudaStreamDestroy(*stream));
}
#endif // DISTRIBUTION_H