#include<iostream>
#include<cuda_runtime.h>

#define TILE_WIDTH 32

// tiled mat mul kernel
__global__ void matMulKernel(float* d_M, float* d_N, float* d_P, int width) {
    // shared memory tiles for M and N
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    // compute row & col index of the element
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    // loop over required tiles
    for (int ph = 0; ph < (width + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        // load d_M and d_N tiles into shared memory
        if (row < width && (ph * TILE_WIDTH + tx) < width) {
            Mds[ty][tx] = d_M[row * width + ph * TILE_WIDTH + tx];
        } else {
            Mds[ty][tx] = 0.0f;
        }

        if (col < width && (ph * TILE_WIDTH + ty) < width) {
            Nds[ty][tx] = d_N[(ph * TILE_WIDTH + ty) * width + col];
        } else {
            Nds[ty][tx] = 0.0f;
        }

        __syncthreads();

        // multiply & accumulate results
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }

        __syncthreads();
    }

    if (row < width && col < width) {
        d_P[row * width + col] = Pvalue;
    }
}

void matMul(float* h_P, float* h_M, float* h_N, int width) {
    int size = width * width * sizeof(float);
    float *d_M, *d_N, *d_P;

    cudaMalloc((void**)&d_M, size);
    cudaMalloc((void**)&d_N, size);
    cudaMalloc((void**)&d_P, size);

    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (width + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, width);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time = 0.0;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Kernel Execution Time: " << time << "ms" << std::endl;

    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

int main() {
    int Width = 1024; 
    int size = Width * Width * sizeof(float);

    float *h_M = (float*)malloc(size);
    float *h_N = (float*)malloc(size);
    float *h_P = (float*)malloc(size);

    for (int i = 0; i < Width * Width; ++i) {
        h_M[i] = static_cast<float>(rand()) / RAND_MAX;
        h_N[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    matMul(h_P, h_M, h_N, Width);

    free(h_M);
    free(h_N);
    free(h_P);

    return 0;
}
// Kernel Execution Time: 2.65376ms