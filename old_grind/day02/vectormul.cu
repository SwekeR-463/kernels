#include<iostream>
#include<cuda_runtime.h>

#define N 1024 * 1024 // size of the vectors

// vector multiplication kernel
__global__ void mul(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        C[i] = A[i] * B[i];
    }
}

int main() {
    float *h_A, *h_B, *h_C;

    float *d_A, *d_B, *d_C;

    // allocation of memory for host vectors
    h_A = (float*)malloc(N * sizeof(float));
    h_B = (float*)malloc(N * sizeof(float));
    h_C = (float*)malloc(N * sizeof(float));

    for(int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i*3);
    }

    // allocation of memory for device vectors
    cudaMalloc((void**)&d_A, N * (sizeof(float)));
    cudaMalloc((void**)&d_B, N * (sizeof(float)));
    cudaMalloc((void**)&d_C, N * (sizeof(float)));

    // copy host vectors to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 512;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // cuda events for getting timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    mul<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time = 0;
    cudaEventElapsedTime(&time, start, stop);

    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Kernel execution time: " << time << " ms" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;

}
// Kernel execution time: 0.330592 ms