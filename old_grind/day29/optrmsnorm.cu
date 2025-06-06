#include <iostream>
#include <cuda_runtime.h>

// optimized rms norm kernel
__global__ void rmsNormKernel(const float* input, float* output, int n) {
    // shared memory for partial sums
    extern __shared__ float sharedData[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // load data into shared memory
    if (idx < n) {
        sharedData[tid] = input[idx] * input[idx]; // square each element
    } else {
        sharedData[tid] = 0.0f; // pad with zeros if out of bounds
    }
    __syncthreads();

    // reduce within the block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }

    // write the block's partial sum to the output
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

float rmsNorm(const float* input, int n) {
    float* d_input, *d_output;
    float h_output = 0.0f;

    cudaMalloc((void**)&d_input, n * sizeof(float));
    cudaMalloc((void**)&d_output, sizeof(float));

    cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    rmsNormKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, n);

    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time = 0.0f;
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "Kernel Execution Time: " << time << "ms" << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    // compute the final RMS norm
    return sqrtf(h_output / n);
}

int main() {
    const int n = 1000000;
    float h_input[n];

    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>(i + 1);
    }

    float rms = rmsNorm(h_input, n);
    std::cout << "RMS Norm: " << rms << std::endl;

    return 0;
}
// Kernel Execution Time: 0.28368ms