#include <iostream>
#include <cuda_runtime.h>

#define SIZE 1000000

// partial sum kernel
__global__ void reductionKernel(float* X, float* output) {
    extern __shared__ float partialSum[]; // dynamic shared memory allocation

    unsigned int t = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // load elements into shared memory
    if (idx < SIZE) {
        partialSum[t] = X[idx];
    } else {
        partialSum[t] = 0.0f; // pad with zeros if out of bounds
    }
    __syncthreads();

    // perform reduction in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            partialSum[t] += partialSum[t + stride];
        }
        __syncthreads();
    }

    // write the result of this block to the output array
    if (t == 0) {
        output[blockIdx.x] = partialSum[0];
    }
}

int main() {
    const int arraySize = SIZE;
    const int blockSize = 256; // threads per block
    const int gridSize = (arraySize + blockSize - 1) / blockSize; // number of blocks

    float* h_X = new float[arraySize];
    float* h_output = new float[gridSize];

    // initialize input array with random values
    for (int i = 0; i < arraySize; i++) {
        h_X[i] = static_cast<float>(rand()) / RAND_MAX; 
    }

    float* d_X;
    float* d_output;

    cudaMalloc((void**)&d_X, arraySize * sizeof(float));
    cudaMalloc((void**)&d_output, gridSize * sizeof(float));

    cudaMemcpy(d_X, h_X, arraySize * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    reductionKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_X, d_output);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Kernel execution time: " << time << " ms" << std::endl;

    cudaMemcpy(h_output, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

    float finalSum = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        finalSum += h_output[i];
    }

    std::cout << "Final sum of all elements: " << finalSum << std::endl;

    cudaFree(d_X);
    cudaFree(d_output);
    delete[] h_X;
    delete[] h_output;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
// Kernel execution time: 54.7347 ms