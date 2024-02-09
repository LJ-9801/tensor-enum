#include "tensor-enum.cuh"
#define SEED 1234

int main(){

    // random number generation
    Tensor<float32> a = {NULL, {3, 1024, 1024}};
    Tensor<float32> b = {NULL, {3, 1024, 1024}};
    generateUniform<float32>(&a.data, get_size(a.shape), SEED, -5.0, 5.0);
    generateNormal<float32>(&b.data, get_size(b.shape), SEED, 0, 1.5);


    // 1D array generation
    Tensor<float32> c = {NULL, {1024}};
    Tensor<float32> d = {NULL, {1024}};
    linspace<float32>(&c.data, 0, 100, get_size(c.shape));
    arange<float32>(&d.data, 0, 0.5, get_size(d.shape));

    float32 start = 1; float32 end = 100; float32 step = 0.5;
    unsigned int size = (end - start) / step;
    Tensor<float32> e = {NULL, {size}};
    logspace<float32>(&e.data, start, end, step);


    // eye
    unsigned int n = 512; unsigned int m = 512;
    Tensor<float32> f = {NULL, {n, m}};
    eye<float32>(&f.data, n, m); 

    return 0;
}