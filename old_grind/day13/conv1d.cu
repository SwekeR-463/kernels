#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

#define n 1000000          // size of the input array
#define MASK_WIDTH 7       // size of the mask(conv kernel)
#define BLOCK_SIZE 256     

__constant__ float d_M[MASK_WIDTH];

// 1d conv kernel
__global__ void convolution_1D_basic_kernel(float *N, float *P, int Mask_Width, int Width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < Width) {
        float Pvalue = 0;
        int N_start_point = i - (Mask_Width / 2);
        
        for (int j = 0; j < Mask_Width; j++) {
            if (N_start_point + j >= 0 && N_start_point + j < Width) {
                Pvalue += N[N_start_point + j] * d_M[j];
            }
        }
        
        P[i] = Pvalue;
    }
}

void convolution_1D(float *h_N, float *h_M, float *h_P, int mask_width, int width) {
    float *d_N, *d_P;
    
    cudaMalloc((void**)&d_N, width * sizeof(float));
    cudaMalloc((void**)&d_P, width * sizeof(float));
    
    cudaMemcpy(d_N, h_N, width * sizeof(float), cudaMemcpyHostToDevice);
    
    // copy mask to constant memory on the device
    cudaMemcpyToSymbol(d_M, h_M, mask_width * sizeof(float));
    
    int gridSize = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    convolution_1D_basic_kernel<<<gridSize, BLOCK_SIZE>>>(d_N, d_P, mask_width, width);
    
    
    cudaMemcpy(h_P, d_P, width * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);

    float time = 0.0f;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Kernel Execution Time: " << time << "ms" << std::endl;
    
    cudaFree(d_N);
    cudaFree(d_P);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    float *h_N = new float[n];      // input array
    float *h_M = new float[MASK_WIDTH]; // mask
    float *h_P = new float[n];      // output array

    // initialize input array h_N with some values 
    for (int i = 0; i < n; i++) {
        h_N[i] = static_cast<float>(i % 10); 
    }

    // initialize mask h_M with some values 
    for (int i = 0; i < MASK_WIDTH; i++) {
        h_M[i] = 1.0f / MASK_WIDTH; // averaging mask
    }
    
    convolution_1D(h_N, h_M, h_P, MASK_WIDTH, n);

    delete[] h_N;
    delete[] h_M;
    delete[] h_P;

    return 0;
}
// Kernel Execution Time: 3.52842ms