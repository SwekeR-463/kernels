#include <stdio.h>
#include <cuda.h>
#include<cuda_runtime.h>

// Naïve prefix sum kernel (inclusive scan)
__global__ void naive_prefix_sum(int *d_in, int *d_out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < n) {
        int sum = 0;
        for (int i = 0; i <= tid; i++) {
            sum += d_in[i];
        }
        d_out[tid] = sum;
    }
}

int main() {
    const int N = 1024;
    int h_in[N], h_out[N];

    
    for (int i = 0; i < N; i++) {
        h_in[i] = 1;  
    }

    int *d_in, *d_out;
    cudaMalloc((void**)&d_in, N * sizeof(int));
    cudaMalloc((void**)&d_out, N * sizeof(int));

    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    naive_prefix_sum<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time = 0;
    cudaEventElapsedTime(&time, start, stop);

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);


    printf("Kernel execution time: %f ms\n", time);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
// Kernel execution time: 55.084927 ms