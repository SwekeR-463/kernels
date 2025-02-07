#include<iostream>
#include<cmath>
#include<random>
#include<cuda_runtime.h>

// sigmoid kernel
__global__ void sigmoidKernel(const float* input, float* output, int n) {
    extern __shared__ float shared_input[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        shared_input[threadIdx.x] = input[idx];
        __syncthreads();
        output[idx] = 1.0f / (1.0f + expf(-shared_input[threadIdx.x]));
    }
}

extern "C" void sigmoid(const float* input, float* output, int n) {
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, n * sizeof(float));
    cudaMalloc((void**)&d_output, n * sizeof(float));

    cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 128; // smaller block size for higher occupancy
    int numBlocks = (n + blockSize - 1) / blockSize;

    float time = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    sigmoidKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, n);

    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);

    cudaFree(d_input);
    cudaFree(d_output);

    cudaEventElapsedTime(&time, start, stop);
    // std::cout << "Kernel Execution Time: " << time << "ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void generate_random_data(float* data, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f); // Range of input values
    for (int i = 0; i < n; i++) {
        data[i] = dist(gen);
    }
}


int main() {
    int n = 1 << 20; // 1 million elements
    float *input = new float[n];
    float *output = new float[n];

    generate_random_data(input, n);

    sigmoid(input, output, n);
   
    delete[] input;
    delete[] output;

    return 0;
}
