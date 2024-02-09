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
__global__ void _logspace(T *output, T start, T stop, T step, T base){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t size = (stop - start) / step;

    if(idx < size){
        output[idx] = pow(base, start + idx * step);
    }
}

template <typename T>
__global__ void _eye(T *output, size_t n, size_t m){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    /**
     * @todo: division is very slow
    */
    if(idx < n * m){
        size_t i = idx / m;
        size_t j = idx % m;
        output[idx] = i == j ? 1 : 0;
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