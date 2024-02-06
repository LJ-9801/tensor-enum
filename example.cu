#include "tensor-enum.cuh"


int main(){
    Tensor<float> a = {NULL, {2, 3, 4}};
    Tensor<float> b = {NULL, {2, 3, 4}}; 

    cudaMalloc(&a.data, get_size(a.shape) * sizeof(float));
    cudaMalloc(&b.data, get_size(b.shape) * sizeof(float));



    cudaFree(a.data);
    cudaFree(b.data);
    return 0;
}