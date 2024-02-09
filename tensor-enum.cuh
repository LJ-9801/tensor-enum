#include "common.cuh"
#include "distribution.cuh"
#include "enumerate.cuh"


/**
 * Generate random numbers from a uniform distribution
 * @param data: pointer to the data
 * @param size: size of the data
 * @param seed: seed for the random number generator
 * @param low: lower bound of the distribution
 * @param high: upper bound of the distribution
*/
template <typename T>
void generateUniform(T **data, size_t size, unsigned long long seed, T low, T high){
    cudaStream_t stream; curandState *devStates;
    setup_random_generator(&stream, &devStates, data, size);
    int blocks = (size + CURAND_MAX_THREADS - 1) / CURAND_MAX_THREADS;
    setup_curandState<<<blocks, CURAND_MAX_THREADS, 0, stream>>>(devStates, seed);
    generate_uniform_kernel<<<blocks, CURAND_MAX_THREADS, 0, stream>>>(devStates, size, low, high, *data);
    finish_random_generator(&stream, &devStates); 
}


/**
 * Generate random numbers from a multiuniform distribution
 * @param data: pointer to the data
 * @param size: size of the data
 * @param seed: seed for the random number generator
 * @param lows: lower bound of the distribution, this should be an array of size size
 * @param highs: upper bound of the distribution, this should be an array of size size
*/
template <typename T>
void generateMultiUniform(T **data, size_t size, unsigned long long seed, T *lows, T *highs){
    assert(*data == NULL && "Data should be NULL");
    cudaStream_t stream; curandState *devStates;
    setup_random_generator(&stream, &devStates, data, size);
    int blocks = (size + CURAND_MAX_THREADS - 1) / CURAND_MAX_THREADS;
    setup_curandState<<<blocks, CURAND_MAX_THREADS, 0, stream>>>(devStates, seed);
    generate_multi_uniform_kernel<<<blocks, CURAND_MAX_THREADS, 0, stream>>>(devStates, size, lows, highs, *data);
    finish_random_generator(&stream, &devStates);

}

/**
 * Generate random numbers from a normal distribution
 * @param data: pointer to the data
 * @param size: size of the data
 * @param seed: seed for the random number generator
 * @param mean: mean of the distribution
 * @param stddev: standard deviation of the distribution
*/
template <typename T>
void generateNormal(T **data, size_t size, unsigned long long seed, T mean, T stddev){
    assert(*data == NULL && "Data should be NULL");
    cudaStream_t stream; curandState *devStates;
    setup_random_generator(&stream, &devStates, data, size);
    int blocks = (size + CURAND_MAX_THREADS - 1) / CURAND_MAX_THREADS;
    setup_curandState<<<blocks, CURAND_MAX_THREADS, 0, stream>>>(devStates, seed);
    generate_normal_kernel<<<blocks, CURAND_MAX_THREADS, 0, stream>>>(devStates, size, mean, stddev, *data);
    finish_random_generator(&stream, &devStates);

}

/**
 * Generate random numbers from a multinormal distribution
 * @param data: pointer to the data
 * @param size: size of the data
 * @param seed: seed for the random number generator
 * @param means: mean of the distribution, this should be an array of size size
 * @param stddevs: standard deviation of the distribution, this should be an array of size size
*/
template <typename T>
void generateMultiNormal(T **data, size_t size, unsigned long long seed, T *means, T *stddevs){
    assert(*data == NULL && "Data should be NULL");
    cudaStream_t stream; curandState *devStates;
    setup_random_generator(&stream, &devStates, data, size);
    int blocks = (size + CURAND_MAX_THREADS - 1) / CURAND_MAX_THREADS;
    setup_curandState<<<blocks, CURAND_MAX_THREADS, 0, stream>>>(devStates, seed);
    generate_multi_normal_kernel<<<blocks, CURAND_MAX_THREADS, 0, stream>>>(devStates, size, means, stddevs, *data);
    finish_random_generator(&stream, &devStates);
}

template <typename T>
void fill(T **data, T value, size_t size){
    assert(*data == NULL && "Data should be NULL");
    cudaStream_t stream;
    setup_fill_generator(&stream, data, size); 
    int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    _fill<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(*data, value, size);
    finish_fill_generator(&stream); 
}

template <typename T>
void ones(T **data, size_t size){
    assert(*data == NULL && "Data should be NULL");
    cudaStream_t stream;
    setup_fill_generator(&stream, data, size); 
    int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    _fill<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(*data, 1, size);
    finish_fill_generator(&stream);
}

template <typename T>
void zeros(T **data, size_t size){
    assert(*data == NULL && "Data should be NULL");
    cudaStream_t stream;
    setup_fill_generator(&stream, data, size); 
    int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    _fill<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(*data, 0, size);
    finish_fill_generator(&stream);
}

template <typename T>
void arange(T **data, T start, T step, size_t size){
    assert(*data == NULL && "Data should be NULL");
    cudaStream_t stream;
    setup_fill_generator(&stream, data, size);
    int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    _arange<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(*data, start, step, size);
    finish_fill_generator(&stream);
}

template <typename T>
void linspace(T **data, T start, T end, size_t size){
    assert(*data == NULL && "Data should be NULL");
    cudaStream_t stream;
    setup_fill_generator(&stream, data, size); 
    int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    _linspace<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(*data, start, end, size);
    finish_fill_generator(&stream);
}

template <typename T>
void logspace(T **data, T start, T end, T step, T base = 10){
    assert(*data == NULL && "Data should be NULL");
    cudaStream_t stream;
    size_t size = (end - start) / step + 1;
    setup_fill_generator(&stream, data, size); 
    int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    _logspace<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(*data, start, end, step, base);
    finish_fill_generator(&stream);
}

template <typename T>
void eye(T **data, size_t n, size_t m){
    assert(*data == NULL && "Data should be NULL");
    cudaStream_t stream;
    setup_fill_generator(&stream, data, n * m); 
    int blocks = (n * m + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    _eye<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(*data, n, m);
    finish_fill_generator(&stream);
}