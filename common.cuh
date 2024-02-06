#ifndef DTYPE_H
#define DTYPE_H
#include <cuComplex.h>
#include <vector>
#include <numeric>
#include <stdio.h>

typedef float float32;
typedef double float64;
typedef int64_t int64;
typedef int32_t int32;
typedef int16_t int16;
typedef int8_t int8;
typedef u_int32_t uint32;
typedef u_int64_t uint64;
typedef u_int16_t uint16;
typedef u_int8_t uint8;
typedef cuFloatComplex complex32;
typedef cuDoubleComplex complex64;


typedef std::vector<size_t> shape_t;

#define get_size(shape) std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>())


/**
 * @brief Tensor struct
 * a simple strunt for letting
 * the user define a tensor with
 * a shape and a data pointer
*/
template <typename T>
struct Tensor{
    T* data;
    shape_t shape;
};


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

#endif // DTYPE_H