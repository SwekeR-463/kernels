#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mma.h> // for tensor cores
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define WARP_SIZE 32 // to execute threads in groups of warps, and each warp consists of 32 threads

// MMA matrix tile dimensions
#define M 16
#define N 16
#define K 16

// GEMM configuration.
#define M_TILES 256
#define N_TILES 256
#define K_TILES 256

#define M_TOTAL (M * M_TILES)
#define N_TOTAL (N * N_TILES)
#define K_TOTAL (K * K_TILES)

// the above define -> M,N,K & M TILES, N TILES, K TILES get together to make a matrix of size 4096 * 4096

using namespace nvcuda;

// init matrix
__host__ void InitMatrix(half *A, half *B, float *C) {
    for (int i = 0; i < M_TOTAL * K_TOTAL; i++)
        A[i] = __float2half(rand() % 1000 / 1000.0f);
    for (int i = 0; i < K_TOTAL * N_TOTAL; i++)
        B[i] = __float2half(rand() % 1000 / 1000.0f);
    for (int i = 0; i < M_TOTAL * N_TOTAL; i++)
        C[i] = rand() % 1000 / 1000.0f;
}

// wmma(warp-synchronous matrix multiply-accumulate) kernel for fp16
__global__ void WMMAF16TensorCore(half *A, half *B, float *C, float *D) {
    int ix = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    // declares small tile-sized matrices to fit into Tensor Cores
    // initializes the accumulator to zero before multiplication
    // prepares for efficient mma_sync operations using Tensor Cores
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> ab_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

    wmma::fill_fragment(ab_frag, 0.0f);

    // AB = A * B
    int a_row = ix * M;
    int b_row = iy * N;
    for (int k = 0; k < K_TOTAL; k += K) {
        int a_col = k;
        int b_col = k;

        if (a_row < M_TOTAL && a_col < K_TOTAL && b_row < K_TOTAL && b_col < N_TOTAL) {
            // load the inputs
            wmma::load_matrix_sync(a_frag, A + a_col + a_row * K_TOTAL, K_TOTAL);
            wmma::load_matrix_sync(b_frag, B + b_col + b_row * N_TOTAL, N_TOTAL);

            // perform the matrix multiplication
            wmma::mma_sync(ab_frag, a_frag, b_frag, ab_frag);
        }
    }

    // D = AB + C
    int c_row = a_row;
    int c_col = b_row;
    if (c_row < M_TOTAL && c_col < N_TOTAL) {
        wmma::load_matrix_sync(c_frag, C + c_col + c_row * N_TOTAL, N_TOTAL, wmma::mem_row_major);

        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = ab_frag.x[i] + c_frag.x[i];
        }

        // store the output
        wmma::store_matrix_sync(D + c_col + c_row * N_TOTAL, c_frag, N_TOTAL, wmma::mem_row_major);
    }
}

cudaError_t CalcWMMA(half *A, half *B, float *C, float *D) {
    cudaError_t cuda_status;
    dim3 gridDim, blockDim;

    // 16 warps in one block
    blockDim.x = 4 * WARP_SIZE;
    blockDim.y = 4;

    gridDim.x = (M_TOTAL + (M * blockDim.x / WARP_SIZE - 1)) / (M * blockDim.x / WARP_SIZE);
    gridDim.y = (N_TOTAL + N * blockDim.y - 1) / (N * blockDim.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    WMMAF16TensorCore<<<gridDim, blockDim>>>(A, B, C, D);
    cuda_status = cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time = 0;
    cudaEventElapsedTime(&time, start, stop);

    printf("[+] GPU (with Tensor Cores) Elapsed Time: %f ms\n", time);
    printf("[+] TFLOPS: %.2f\n", ((double)M_TOTAL * N_TOTAL * K_TOTAL * 2) / time / 1e9);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return cuda_status;
}

int main() {
    cudaError_t cuda_status;
    cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) {
        printf("cudaSetDevice failed! ");
        return 1;
    }

    // matrix on device
    half *A;
    half *B;
    float *C;
    float *D;

    // CUDA Unified Memory
    cudaMallocManaged((void **)&A, sizeof(half) * M_TOTAL * K_TOTAL);
    cudaMallocManaged((void **)&B, sizeof(half) * K_TOTAL * N_TOTAL);
    cudaMallocManaged((void **)&C, sizeof(float) * M_TOTAL * N_TOTAL);
    cudaMallocManaged((void **)&D, sizeof(float) * M_TOTAL * N_TOTAL);

    // initialize matrices
    printf("[*] Initializing Matrix...\n");
    InitMatrix(A, B, C);
    printf("[+]   A: %d x %d\n", M_TOTAL, K_TOTAL);
    printf("[+]   B: %d x %d\n", K_TOTAL, N_TOTAL);
    printf("[+]   C: %d x %d\n", M_TOTAL, N_TOTAL);

    // compute D = A * B + C using Tensor Cores
    printf("[*] Computing D = A * B + C with Tensor Cores...\n");
    cuda_status = CalcWMMA(A, B, C, D);

    if (cuda_status != cudaSuccess) {
        printf("Kernel execution failed! ");
        return 1;
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(D);

    cuda_status = cudaDeviceReset();
    if (cuda_status != cudaSuccess) {
        printf("cudaDeviceReset failed! ");
        return 1;
    }

    return 0;
}
// [+] GPU (with Tensor Cores) Elapsed Time: 95.950851 ms
// [+] TFLOPS: 1.43