#include "tensor-enum.cuh"
#define SEED 1234

int main(){
    Tensor<float32> a = {NULL, {100, 1024, 1024}};
    Tensor<float32> b = {NULL, {100, 1024, 1024}}; 

    generateUniform<float32>(&a.data, get_size(a.shape), SEED, -5.0, 5.0);
    generateNormal<float32>(&b.data, get_size(b.shape), SEED, 0, 1.5);

    return 0;
}