#include <iostream>
#include <cuda_runtime.h>

// layernorm kernel
__global__ void layer_norm_kernel(const float* input, float* output, const int batch_size, const int feature_size, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        // mean
        float mean = 0.0f;
        for (int i = 0; i < feature_size; ++i) {
            mean += input[idx * feature_size + i];
        }
        mean /= feature_size;

        // variance
        float variance = 0.0f;
        for (int i = 0; i < feature_size; ++i) {
            float diff = input[idx * feature_size + i] - mean;
            variance += diff * diff;
        }
        variance /= feature_size;

        // normalize
        float inv_std = 1.0f / sqrtf(variance + epsilon);
        for (int i = 0; i < feature_size; ++i) {
            output[idx * feature_size + i] = (input[idx * feature_size + i] - mean) * inv_std;
        }
    }
}

void layer_norm(const float* input, float* output, int batch_size, int feature_size, float epsilon = 1e-5) {
    // allocate device memory
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, batch_size * feature_size * sizeof(float));
    cudaMalloc((void**)&d_output, batch_size * feature_size * sizeof(float));

    // copy input data to device
    cudaMemcpy(d_input, input, batch_size * feature_size * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;

    int blocks_per_grid = (batch_size + threads_per_block - 1) / threads_per_block;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    layer_norm_kernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, batch_size, feature_size, epsilon);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time = 0.0f;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Kernel Execution Time: " << time << "ms" << std::endl;

    // copy output data back to host
    cudaMemcpy(output, d_output, batch_size * feature_size * sizeof(float), cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    const int batch_size = 4096; // large batch size
    const int feature_size = 4096; // large feature size
    float* input = new float[batch_size * feature_size];
    float* output = new float[batch_size * feature_size];

    // initialize input with random values
    for (int i = 0; i < batch_size * feature_size; ++i) {
        input[i] = static_cast<float>(rand()) / RAND_MAX; // random values between 0 and 1
    }

    layer_norm(input, output, batch_size, feature_size);

    delete[] input;
    delete[] output;

    return 0;
}
// Kernel Execution Time: 4.07101ms