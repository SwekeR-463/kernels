#include <iostream>
#include <cuda_runtime.h>

// naive rmsnorm kernel
__global__ void sum_of_squares_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread computes the square of its element
    if (idx < n) {
        output[idx] = input[idx] * input[idx];
    } else {
        output[idx] = 0.0f; // pad with zeros if out of bounds
    }
}

float rms_norm(const float* input, int n) {
    float* d_input, *d_output;
    float h_output = 0.0f;

    cudaMalloc((void**)&d_input, n * sizeof(float));
    cudaMalloc((void**)&d_output, n * sizeof(float));

    cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    sum_of_squares_kernel<<<gridSize, blockSize>>>(d_input, d_output, n);

    float* h_partial_sums = new float[n];
    cudaMemcpy(h_partial_sums, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // perform the final reduction on the host
    for (int i = 0; i < n; i++) {
        h_output += h_partial_sums[i];
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time = 0.0f;
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "Kernel Execution Time: " << time << "ms" << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_partial_sums;

    return sqrtf(h_output / n);
}

int main() {
    const int n = 1000000; // million inputs
    float h_input[n];

    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>(i + 1);
    }

    float rms = rms_norm(h_input, n);
    std::cout << "RMS Norm: " << rms << std::endl;

    return 0;
}
// Kernel Execution Time: 7.53171ms