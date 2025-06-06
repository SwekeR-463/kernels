#include<iostream>
#include<cmath>
#include<cuda_runtime.h>

// softmax kernel
__global__ void softmaxKernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float max_val = input[0];
        for (int i = 1; i < n; ++i) {
            if (input[i] > max_val) {
                max_val = input[i];
            }
        }
        
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += expf(input[i] - max_val);
        }

        output[idx] = expf(input[idx] - max_val) / sum;
    }
}

void softmax(float* input, float* output, int n) {
    float *d_input, *d_output;
    
    cudaMalloc((void**)&d_input, n * sizeof(float));
    cudaMalloc((void**)&d_output, n * sizeof(float));

    cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    softmaxKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);

    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time = 0.0f;
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "Kernel Execution Time: " << time << "ms" << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int num_elements = 10;
    float input[num_elements] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    float output[num_elements];

    softmax(input, output, num_elements);

    std::cout << "Softmax output: ";
    for (int i = 0; i < num_elements; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
// Kernel Execution Time: 1.51315ms