#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H
#include <stdio.h>

#define CHECK_CURAND(call) \
do { \
    curandStatus_t status = (call); \
    if (status != CURAND_STATUS_SUCCESS) { \
        fprintf(stderr, "curand error:%d: %s: %d\n", status, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CHECK_CUDA(call) \
do { \
    cudaError_t status = (call); \
    if (status != cudaSuccess) { \
        fprintf(stderr, "cuda error:%d: %s: %d\n", status, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)



#endif