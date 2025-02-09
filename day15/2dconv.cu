#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

#define TILE_SIZE 16
#define MAX_MASK_WIDTH 5
#define O_TILE_WIDTH (TILE_SIZE - MAX_MASK_WIDTH + 1)

typedef struct {
    int width;       
    int height;      
    int pitch;       // (number of bytes per row, including padding)
    int channels;    
    float* data;     
} wbImage_t;

// 2d conv kernel
__global__ void convolution_2D_tiled_kernel(wbImage_t output, const wbImage_t input, int Mask_Width, const float *M) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_o = blockIdx.y * O_TILE_WIDTH + ty;
    int col_o = blockIdx.x * O_TILE_WIDTH + tx;

    int row_i = row_o - Mask_Width / 2;
    int col_i = col_o - Mask_Width / 2;

    __shared__ float N_ds[TILE_SIZE + MAX_MASK_WIDTH - 1][TILE_SIZE + MAX_MASK_WIDTH - 1];

    if ((row_i >= 0) && (row_i < input.height) && (col_i >= 0) && (col_i < input.width)) {
        N_ds[ty][tx] = input.data[row_i * input.pitch + col_i];
    } else {
        N_ds[ty][tx] = 0.0f; 
    }
    __syncthreads();

    float output_value = 0.0f;
    if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH) {
        for (int i = 0; i < Mask_Width; i++) {
            for (int j = 0; j < Mask_Width; j++) {
                output_value += M[i * Mask_Width + j] * N_ds[i + ty][j + tx];
            }
        }

        if (row_o < output.height && col_o < output.width) {
            output.data[row_o * output.width + col_o] = output_value;
        }
    }
}

void convolution_2D_tiled(wbImage_t output, const wbImage_t input, int Mask_Width, const float *d_M) {
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((input.width + O_TILE_WIDTH - 1) / O_TILE_WIDTH,
                 (input.height + O_TILE_WIDTH - 1) / O_TILE_WIDTH);

    convolution_2D_tiled_kernel<<<dimGrid, dimBlock>>>(output, input, Mask_Width, d_M);

}

int main() {
    int width = 1024;
    int height = 1024;
    int channels = 3; 
    const int pitch = width; 

    int Mask_Width = 5; 

    wbImage_t h_input, h_output;
    h_input.width = width;
    h_input.height = height;
    h_input.pitch = pitch;
    h_input.channels = channels;
    h_input.data = (float *)malloc(width * height * sizeof(float));

    h_output.width = width;
    h_output.height = height;
    h_output.pitch = pitch;
    h_output.channels = channels;
    h_output.data = (float *)malloc(width * height * sizeof(float)); 

    float *h_M = (float *)malloc(Mask_Width * Mask_Width * sizeof(float)); 

    for (int i = 0; i < width * height; i++) {
        h_input.data[i] = static_cast<float>(rand()) / RAND_MAX; 
    }
    for (int i = 0; i < Mask_Width * Mask_Width; i++) {
        h_M[i] = static_cast<float>(rand()) / RAND_MAX; 
    }

    wbImage_t d_input, d_output;
    cudaMalloc((void **)&d_input.data, width * height * sizeof(float));
    cudaMalloc((void **)&d_output.data, width * height * sizeof(float));
    float *d_M;
    cudaMalloc((void **)&d_M, Mask_Width * Mask_Width * sizeof(float));

    cudaMemcpy(d_input.data, h_input.data, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, Mask_Width * Mask_Width * sizeof(float), cudaMemcpyHostToDevice);

    d_input.width = width;
    d_input.height = height;
    d_input.pitch = pitch;
    d_input.channels = channels;

    d_output.width = width;
    d_output.height = height;
    d_output.pitch = pitch;
    d_output.channels = channels;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    convolution_2D_tiled(d_output, d_input, Mask_Width, d_M);

    cudaMemcpy(h_output.data, d_output.data, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);

    float time = 0.0f;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Kernel Execution Time: " << time << "ms" << std::endl;

    cudaFree(d_input.data);
    cudaFree(d_output.data);
    cudaFree(d_M);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(h_input.data);
    free(h_output.data);
    free(h_M);

    return 0;
}
// Kernel Execution Time 3.85766ms