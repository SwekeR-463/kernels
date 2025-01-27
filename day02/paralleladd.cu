#include<iostream>
#include<math.h>

// addition kernel
__global__ void add(int n, float *x, float *y) {
    int index = threadIdx.x + blockIdx.x + blockDim.x; // global thread index
    int stride = blockDim.x + gridDim.x; // total threads in the grid
    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}

int main(void) {
    int N = 1<<20; // 1 million
    float *x, *y;

    // allocate unified memory - accessible from GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    for(int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // launch the kernel with multiple blocks
    int blockSize = 512; // threads per block
    int numBlocks = (N + blockSize - 1) / blockSize; // total no of blocks
    // run the kernel 
    add<<<numBlocks, blockSize>>>(N, x, y);

    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for(int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i]-2.9f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    // free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}