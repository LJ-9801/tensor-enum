#ifndef DTYPE_H
#define DTYPE_H
#include <cuComplex.h>
#include <vector>
#include <numeric>

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

template <typename T>
struct Tensor{
    T* data;
    shape_t shape;
};


#endif // DTYPE_H