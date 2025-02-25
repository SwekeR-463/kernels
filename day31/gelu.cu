#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

// gelu kernel
__global__ void gelu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float cdf = 0.5f * (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
        output[idx] = x * cdf;
    }
}

void gelu(const float* input, float* output, int n) {

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gelu_kernel<<<numBlocks, blockSize>>>(input, output, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time = 0.0f;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Kernel Execution Time: " << time << "ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int n = 1000000;
    float* h_input = new float[n];
    float* h_output = new float[n];

    for (int i = 0; i < n; ++i) {
        h_input[i] = static_cast<float>(i) - n / 2;
    }

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, n * sizeof(float));
    cudaMalloc((void**)&d_output, n * sizeof(float));

    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    gelu(d_input, d_output, n);

    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;

    return 0;
}
// Kernel Execution Time: 2.90659ms