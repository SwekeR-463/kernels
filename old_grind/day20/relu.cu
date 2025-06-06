#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// optimized ReLU Kernel
__global__ void reluKernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = fmaxf(0.0f, input[idx]); // ReLU operation
    }
}

void reluCUDA(float* h_input, float* h_output, int N) {
    float *d_input, *d_output;
    size_t size = N * sizeof(float);

    CHECK_CUDA(cudaMalloc((void**)&d_input, size));
    CHECK_CUDA(cudaMalloc((void**)&d_output, size));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    int blockSize = 256; 
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    reluKernel<<<gridSize, blockSize>>>(d_input, d_output, N);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float time = 0;
    CHECK_CUDA(cudaEventElapsedTime(&time, start, stop));
    std::cout << "Kernel execution time: " << time << " ms" << std::endl;

    CHECK_CUDA(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main() {
    const int N = 1 << 20; // 1M elements
    float *h_input = new float[N];
    float *h_output = new float[N];

    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; 
    }

    reluCUDA(h_input, h_output, N);

    for (int i = 0; i < N; i++) {
        if (h_output[i] != fmaxf(0.0f, h_input[i])) {
            std::cerr << "Error at index " << i << std::endl;
            break;
        }
    }

    delete[] h_input;
    delete[] h_output;

    return 0;
}
// Kernel execution time: 0.383968 ms