#include <iostream>
#include <chrono>
#include "tensor-enum.cuh"


int main(){
    unsigned long long seed = std::chrono::system_clock::now().time_since_epoch().count();

    // random number generation
    Tensor<float32> a = {NULL, {3, 4, 4}};
    Tensor<float32> b = {NULL, {3, 4, 4}};
    generateUniform<float32>(&a.data, get_size(a.shape), seed, -2, 2);
    generateNormal<float32>(&b.data, get_size(b.shape), seed, 0, 1.5);

    std::cout << "a is: \n" << a << std::endl; 
    std::cout << "b is: \n" << b << std::endl;

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
    unsigned int n = 5; unsigned int m = 5;
    Tensor<float32> f = {NULL, {n, m}};
    eye<float32>(&f.data, n, m);

    return 0;
}