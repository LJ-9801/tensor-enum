#ifndef DTYPE_H
#define DTYPE_H
#include <cuComplex.h>
#include <vector>
#include <numeric>
#include <stdio.h>
#include <ostream>

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

/**
 * @brief get_size
 * @param shape
 * @return the size of the tensor
*/
#define get_size(shape) std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>())


/**
 * @brief Tensor struct
 * a simple struct for letting
 * the user define a tensor with
 * a shape and a data pointer
*/
template <typename T>
struct Tensor{
    T* data;
    shape_t shape;

    ~Tensor(){
        if(data != nullptr){
            CHECK_CUDA(cudaFree(data));
        }
    }
};

std::ostream& operator<<(std::ostream& os, const shape_t& shape){
    os << "(";
    for(size_t i = 0; i < shape.size(); i++){
        os << shape[i];
        if(i != shape.size() - 1){
            os << ", ";
        }
    }
    os << ")";
    return os;
}

template<typename T>
uint32_t check_length(T number, uint32_t max_len){
    std::string str = std::to_string(number);
    return str.length() > max_len ? str.length() : max_len;
    
}

typedef std::vector<std::string> string_format_t; 
// this function will loop over all the elements inside the
// tensor and automatically format them to fit the longest
// element to be printed.
template<typename T>
void tensor_str_walk(std::vector<T> tensor, shape_t& shape, 
                        size_t& index, int depth, uint32_t& max_real, uint32_t& max_imag, 
                        string_format_t& result){
        if(depth == shape.size() - 1){
            result.push_back("[");
            for (int i = 0; i < shape[depth]; i++){
                max_real = check_length(tensor[index++], max_real);
                result.push_back("");
                if(i < shape[depth] - 1) 
                    result.push_back(", "); 
            }
            result.push_back("]"); 
            return;
        }
        
        result.push_back("["); 
        for (int i = 0; i < shape[depth]; i++){
            tensor_str_walk(tensor, shape, index, depth + 1, max_real, max_imag, result);
            if(i < shape[depth] - 1) {
                result.push_back(",\n" + std::string(depth+1, ' '));
            }
        }
        result.push_back("]"); 
}

template<typename T>
void formatter(std::ostream& os, string_format_t& result, uint32_t max_real, uint32_t max_imag, std::vector<T>& tmp_data){
    size_t counter = 0;
    for(auto& s: result){
        if(s.size() == 0){
            s = std::string(max_real + max_imag, ' ');
            std::string snum = std::to_string(tmp_data[counter]);
            s.replace(s.size() - snum.size(), snum.size(), snum); 
            counter++;
        }
        os << s; 
    } 
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor){
    CHECK_CUDA(cudaDeviceSynchronize());
    string_format_t result;
    shape_t shape = tensor.shape;
    std::vector<T> tmp_data = std::vector<T>(get_size(shape));
    CHECK_CUDA(cudaMemcpy(tmp_data.data(), tensor.data, get_size(shape) * sizeof(T), cudaMemcpyDeviceToHost));

    // walk the tensor and get the max length of the all numbers
    size_t index = 0; uint32_t max_real = 0; uint32_t max_imag = 0;
    tensor_str_walk<T>(tmp_data, shape, index, 0, max_real, max_imag, result);
    // now we need to iterate over the string formatter and format all
    // the empty string with the number
    formatter(os, result, max_real, max_imag, tmp_data); 
    return os;
}

#endif // DTYPE_H