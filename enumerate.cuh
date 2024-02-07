#ifndef ENUMERATE_H
#define ENUMERATE_H
#include "distribution.cuh"

#define THREADS_PER_BLOCK 1024

template <typename T>
__global__ void _fill(T *output, T value, size_t size){

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        output[idx] = value;
    }
}

template <typename T>
__global__ void _arange(T *output, T start, T step, size_t size){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        output[idx] = start + idx * step;
    }
}

template <typename T>
__global__ void _linspace(T *output, T start, T stop, size_t size){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        output[idx] = start + idx * (stop - start) / (size - 1);
    }
}

template <typename T>
inline void setup_fill_generator(cudaStream_t *stream, T **data, size_t size){
    CHECK_CUDA(cudaStreamCreate(stream));
    CHECK_CUDA(cudaMallocAsync(data, size * sizeof(T), *stream));
}

inline void finish_fill_generator(cudaStream_t *stream){
    CHECK_CUDA(cudaStreamSynchronize(*stream));
    CHECK_CUDA(cudaStreamDestroy(*stream));
}
#endif // KERNELS_H