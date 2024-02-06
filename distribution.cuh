#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "helper.cuh"


#define CURAND_MAX_THREADS 256

__global__ void setup_curandState(curandState *state, unsigned long long seed);

template <typename T>
__global__ void generate_uniform_kernel(curandState *state, size_t size, T low, T high, T *result);

template <typename T>
__global__ void generate_multi_uniform_kernel(curandState *state, size_t size, T *low, T *high, T *result);

template <typename T>
__global__ void generate_normal_kernel(curandState *state, size_t size, T mean, T stddev, T *result);

template <typename T>
__global__ void generate_multi_normal_kernel(curandState *state, size_t size, T *mean, T *stddev, T *result);

template <typename T>
void generateUniform(T **data, size_t size, unsigned long long seed, T low, T high);

template <typename T>
void generateMultiUniform(T **data, size_t size, unsigned long long seed, T *low, T *high);

template <typename T>
void generateNormal(T **data, size_t size, unsigned long long seed, T mean, T stddev);

template <typename T>
void generateMultiNormal(T **data, size_t size, unsigned long long seed, T *mean, T *stddev);
#endif // DISTRIBUTION_H