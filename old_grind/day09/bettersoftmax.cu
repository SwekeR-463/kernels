#include<iostream>
#include<cmath>
#include<cuda_runtime.h>

#define BLOCK_SIZE 256

// better softmax kernel
__global__ void softmaxKernel(float* input, float* output, int n) {
    extern __shared__ float shared_data[]; // shared memory for intermediate results

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // load input into shared memory
    float val = (idx < n) ? input[idx] : -INFINITY;
    shared_data[tid] = val;
    __syncthreads();

    // parallel reduction to find the maximum value in the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }

    float max_val = shared_data[0];
    __syncthreads();

    // compute exponentials and store them in shared memory 
    shared_data[tid] = (idx < n) ? expf(val - max_val) : 0.0f;
    __syncthreads();

    // parallel reduction to compute the sum of exponentials
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    float sum_exp = shared_data[0];
    __syncthreads();

    // compute softmax
    if (idx < n) {
        output[idx] = shared_data[tid] / sum_exp;
    }
}

void softmax(float* input, float* output, int n) {
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, n * sizeof(float));
    cudaMalloc((void**)&d_output, n * sizeof(float));

    cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    softmaxKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_input, d_output, n);

    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time = 0.0f;
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "Kernel Execution Time: " << time << "ms" << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    const int n = 10;
    float input[n] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    float output[n];

    softmax(input, output, n);

    std::cout << "Softmax output: ";
    for (int i = 0; i < n; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
// Kernel Execution Time: 0.237184ms