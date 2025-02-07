#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 128
#define MASK_WIDTH 7
#define RADIUS (MASK_WIDTH / 2)

// 1d cached tiled conv kernel
__global__ void convolution_1D_tiled_caching_kernel(float *N, float *M, float *P, int Width) {
    __shared__ float N_ds[TILE_SIZE + 2 * RADIUS];
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // load main elements into shared memory
    if (i < Width) {
        N_ds[threadIdx.x + RADIUS] = N[i];
    }

    // load halo elements into shared memory
    if (threadIdx.x < RADIUS) {
        int left_halo_index = blockIdx.x * blockDim.x - RADIUS + threadIdx.x;
        int right_halo_index = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
        if (left_halo_index >= 0) {
            N_ds[threadIdx.x] = (left_halo_index < Width) ? N[left_halo_index] : 0.0f;
        }
        if (right_halo_index < Width) {
            N_ds[threadIdx.x + blockDim.x + RADIUS] = N[right_halo_index];
        }
    }

    __syncthreads();

    float Pvalue = 0;
    if (i < Width) {
        for (int j = 0; j < MASK_WIDTH; j++) {
            int N_index = threadIdx.x + j;
            Pvalue += N_ds[N_index] * M[j];
        }
        P[i] = Pvalue;
    }
}

int main() {
    const int num_elements = 1000000; 
    const int array_size = num_elements * sizeof(float);
    const int mask_size = MASK_WIDTH * sizeof(float);
    const int block_size = TILE_SIZE;
    const int grid_size = (num_elements + block_size - 1) / block_size;

    float *h_N = new float[num_elements];
    float *h_P = new float[num_elements];
    float *h_M = new float[MASK_WIDTH];

    float *d_N, *d_P, *d_M;
    cudaMalloc((void **)&d_N, array_size);
    cudaMalloc((void **)&d_P, array_size);
    cudaMalloc((void **)&d_M, mask_size);

    for (int i = 0; i < num_elements; ++i) {
        h_N[i] = static_cast<float>(i);  
    }

    for (int j = 0; j < MASK_WIDTH; ++j) {
        h_M[j] = 1.0f / MASK_WIDTH;  
    }

    cudaMemcpy(d_N, h_N, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, mask_size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    convolution_1D_tiled_caching_kernel<<<grid_size, block_size>>>(d_N, d_M, d_P, num_elements);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Kernel execution time: " << time << " ms" << std::endl;

    cudaMemcpy(h_P, d_P, array_size, cudaMemcpyDeviceToHost);

    cudaFree(d_N);
    cudaFree(d_P);
    cudaFree(d_M);

    delete[] h_N;
    delete[] h_P;
    delete[] h_M;

    return 0;
}
// Kernel execution time: 0.082944 ms