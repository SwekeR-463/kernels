#include<iostream>
#include<cuda_runtime.h>


// mat mul kernel
__global__ void matmulkernel(float* A, float* B, float* C, int N) {
    // compute row & col indices for the current thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // ensure thread indices are within matrix bounds
    if (row < N && col < N) {
        float value = 0;
        // dot prod for A row & B col
        for (int i = 0; i < N; ++i) {
            value += A[row * N + i] * B[i * N + col]; 
        }
        // store the computed value in C
        C[row * N + col] = value;
    }
}

// host function
void matmul(float* h_A, float* h_B, float* h_C, int N) {
    int size = N * N * sizeof(float);
    float *d_A, *d_B, *d_C;

    // allocate memory on gpu
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // copy A & B from host to device -> CPU to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // thread block size 
    dim3 dimBlock(32, 32);
    // grid size to cover all elements of the matrix
    dim3 dimGrid((N + 32 - 1) / 32, (N + 32 - 1) / 32);
    // why? -> even if N is not divisible by 32 it will launch extra blocks as buffer to handle remaining elements

    // for kernel execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matmulkernel<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Kernel Execution Time: " << time << "ms" << std::endl;

    // copy C back to CPU from GPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // free allocated memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int N = 1024;
    int size = N * N * sizeof(float);

    // allocate pinned memory on the host for better data transfer performance
    float *h_A, *h_B, *h_C;
    cudaMallocHost(&h_A, size);
    cudaMallocHost(&h_B, size);
    cudaMallocHost(&h_C, size);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    matmul(h_C, h_A, h_B, N);

    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    return 0;
}
// Kernel Execution Time: 3.74483ms