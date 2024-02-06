#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <assert.h>
#include "common.cuh"


#define CURAND_MAX_THREADS 256


#define assert_float_type() \
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
            assert_float_type()
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
            assert_float_type()
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
            assert_float_type()
        
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
            assert_float_type()
    } 

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


/**
 * Generate random numbers from a uniform distribution
 * @param data: pointer to the data
 * @param size: size of the data
 * @param seed: seed for the random number generator
 * @param low: lower bound of the distribution
 * @param high: upper bound of the distribution
*/
template <typename T>
void generateUniform(T **data, size_t size, unsigned long long seed, T low, T high){
    cudaStream_t stream; curandState *devStates;
    setup_random_generator(&stream, &devStates, data, size);
    int blocks = (size + CURAND_MAX_THREADS - 1) / CURAND_MAX_THREADS;
    setup_curandState<<<blocks, CURAND_MAX_THREADS, 0, stream>>>(devStates, seed);
    generate_uniform_kernel<<<blocks, CURAND_MAX_THREADS, 0, stream>>>(devStates, size, low, high, *data);
    finish_random_generator(&stream, &devStates); 
}


/**
 * Generate random numbers from a multiuniform distribution
 * @param data: pointer to the data
 * @param size: size of the data
 * @param seed: seed for the random number generator
 * @param lows: lower bound of the distribution, this should be an array of size size
 * @param highs: upper bound of the distribution, this should be an array of size size
*/
template <typename T>
void generateMultiUniform(T **data, size_t size, unsigned long long seed, T *lows, T *highs){
    assert(*data == NULL && "Data should be NULL");
    cudaStream_t stream; curandState *devStates;
    setup_random_generator(&stream, &devStates, data, size);
    int blocks = (size + CURAND_MAX_THREADS - 1) / CURAND_MAX_THREADS;
    setup_curandState<<<blocks, CURAND_MAX_THREADS, 0, stream>>>(devStates, seed);
    generate_multi_uniform_kernel<<<blocks, CURAND_MAX_THREADS, 0, stream>>>(devStates, size, lows, highs, *data);
    finish_random_generator(&stream, &devStates);

}

/**
 * Generate random numbers from a normal distribution
 * @param data: pointer to the data
 * @param size: size of the data
 * @param seed: seed for the random number generator
 * @param mean: mean of the distribution
 * @param stddev: standard deviation of the distribution
*/
template <typename T>
void generateNormal(T **data, size_t size, unsigned long long seed, T mean, T stddev){
    assert(*data == NULL && "Data should be NULL");
    cudaStream_t stream; curandState *devStates;
    setup_random_generator(&stream, &devStates, data, size);
    int blocks = (size + CURAND_MAX_THREADS - 1) / CURAND_MAX_THREADS;
    setup_curandState<<<blocks, CURAND_MAX_THREADS, 0, stream>>>(devStates, seed);
    generate_normal_kernel<<<blocks, CURAND_MAX_THREADS, 0, stream>>>(devStates, size, mean, stddev, *data);
    finish_random_generator(&stream, &devStates);

}

/**
 * Generate random numbers from a multinormal distribution
 * @param data: pointer to the data
 * @param size: size of the data
 * @param seed: seed for the random number generator
 * @param means: mean of the distribution, this should be an array of size size
 * @param stddevs: standard deviation of the distribution, this should be an array of size size
*/
template <typename T>
void generateMultiNormal(T **data, size_t size, unsigned long long seed, T *means, T *stddevs){
    assert(*data == NULL && "Data should be NULL");
    cudaStream_t stream; curandState *devStates;
    setup_random_generator(&stream, &devStates, data, size);
    int blocks = (size + CURAND_MAX_THREADS - 1) / CURAND_MAX_THREADS;
    setup_curandState<<<blocks, CURAND_MAX_THREADS, 0, stream>>>(devStates, seed);
    generate_multi_normal_kernel<<<blocks, CURAND_MAX_THREADS, 0, stream>>>(devStates, size, means, stddevs, *data);
    finish_random_generator(&stream, &devStates);
}
#endif // DISTRIBUTION_H