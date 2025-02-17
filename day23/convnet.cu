#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

#define BLOCK_SIZE 16

// kernel for forward convolution
__global__ void convolutionForwardKernel(const float* input, const float* weights, float* output,
                                         int input_channels, int input_height, int input_width,
                                         int num_filters, int kernel_size, int output_height, int output_width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int filter_idx = blockIdx.z;

    if (row < output_height && col < output_width) {
        float sum = 0.0f;
        for (int c = 0; c < input_channels; c++) {
            for (int k_r = 0; k_r < kernel_size; k_r++) {
                for (int k_c = 0; k_c < kernel_size; k_c++) {
                    int in_r = row + k_r;
                    int in_c = col + k_c;
                    int input_idx = c * input_height * input_width + in_r * input_width + in_c;
                    int weight_idx = filter_idx * input_channels * kernel_size * kernel_size + 
                                     c * kernel_size * kernel_size + k_r * kernel_size + k_c;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
        int output_idx = filter_idx * output_height * output_width + row * output_width + col;
        output[output_idx] = sum;
    }
}

// kernel for backward convolution (dLdW)
__global__ void convolutionBackwardKernel_dLdW(const float* input, const float* dLdY, float* dLdW,
                                               int input_channels, int input_height, int input_width,
                                               int num_filters, int kernel_size, int output_height, int output_width) {
    int f = blockIdx.x;
    int c = blockIdx.y;
    int k_r = threadIdx.y;
    int k_c = threadIdx.x;

    float grad = 0.0f;
    for (int out_r = 0; out_r < output_height; out_r++) {
        for (int out_c = 0; out_c < output_width; out_c++) {
            int in_r = out_r + k_r;
            int in_c = out_c + k_c;
            int input_idx = c * input_height * input_width + in_r * input_width + in_c;
            int dLdY_idx = f * output_height * output_width + out_r * output_width + out_c;
            grad += input[input_idx] * dLdY[dLdY_idx];
        }
    }
    int weight_idx = f * input_channels * kernel_size * kernel_size + c * kernel_size * kernel_size + k_r * kernel_size + k_c;
    dLdW[weight_idx] = grad;
}

// kernel for backward convolution (dLdX)
__global__ void convolutionBackwardKernel_dLdX(const float* dLdY, const float* weights, float* dLdX,
                                               int input_channels, int input_height, int input_width,
                                               int num_filters, int kernel_size, int output_height, int output_width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.z;

    if (row < input_height && col < input_width) {
        float grad = 0.0f;
        for (int f = 0; f < num_filters; f++) {
            for (int k_r = 0; k_r < kernel_size; k_r++) {
                for (int k_c = 0; k_c < kernel_size; k_c++) {
                    int out_r = row - k_r;
                    int out_c = col - k_c;
                    if (out_r >= 0 && out_r < output_height && out_c >= 0 && out_c < output_width) {
                        int dLdY_idx = f * output_height * output_width + out_r * output_width + out_c;
                        int weight_idx = f * input_channels * kernel_size * kernel_size + 
                                         c * kernel_size * kernel_size + k_r * kernel_size + k_c;
                        grad += dLdY[dLdY_idx] * weights[weight_idx];
                    }
                }
            }
        }
        int dLdX_idx = c * input_height * input_width + row * input_width + col;
        dLdX[dLdX_idx] = grad;
    }
}

// function for execution time
void timeKernelExecution(cudaEvent_t start, cudaEvent_t stop, const char* kernel_name) {
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << kernel_name << " Execution Time: " << milliseconds << " ms\n";
}

void testConvNet() {
    const int batch_size = 1;
    const int input_channels = 1;
    const int input_height = 4;
    const int input_width = 4;
    const int kernel_size = 3;
    const int num_filters = 2;
    const int output_height = input_height - kernel_size + 1;
    const int output_width = input_width - kernel_size + 1;
    
    float input[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    
    float weights[] = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1,
        0, 1, -1,
        0, 1, -1,
        0, 1, -1
    };
    
    float *d_input, *d_weights, *d_output, *dLdY, *dLdW, *dLdX;
    size_t input_size = input_channels * input_height * input_width * sizeof(float);
    size_t weights_size = num_filters * input_channels * kernel_size * kernel_size * sizeof(float);
    size_t output_size = num_filters * output_height * output_width * sizeof(float);
    
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_weights, weights_size);
    cudaMalloc(&d_output, output_size);
    cudaMalloc(&dLdY, output_size);
    cudaMalloc(&dLdW, weights_size);
    cudaMalloc(&dLdX, input_size);
    
    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, weights_size, cudaMemcpyHostToDevice);
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((output_width + BLOCK_SIZE - 1) / BLOCK_SIZE, (output_height + BLOCK_SIZE - 1) / BLOCK_SIZE, num_filters);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // forward convolution
    cudaEventRecord(start);
    convolutionForwardKernel<<<gridDim, blockDim>>>(d_input, d_weights, d_output, input_channels, input_height, input_width,
                                                    num_filters, kernel_size, output_height, output_width);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    timeKernelExecution(start, stop, "Forward Convolution");

    // backward convolution (dLdW)
    cudaEventRecord(start);
    convolutionBackwardKernel_dLdW<<<dim3(num_filters, input_channels), dim3(kernel_size, kernel_size)>>>(d_input, dLdY, dLdW,
                                                  input_channels, input_height, input_width,
                                                  num_filters, kernel_size, output_height, output_width);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    timeKernelExecution(start, stop, "Backward Convolution (dLdW)");

    // backward convolution (dLdX)
    cudaEventRecord(start);
    convolutionBackwardKernel_dLdX<<<gridDim, blockDim>>>(dLdY, d_weights, dLdX, input_channels, input_height, input_width,
                                                        num_filters, kernel_size, output_height, output_width);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    timeKernelExecution(start, stop, "Backward Convolution (dLdX)");
    
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
    cudaFree(dLdY);
    cudaFree(dLdW);
    cudaFree(dLdX);
}

int main() {
    testConvNet();
    return 0;
}
// Forward Convolution Execution Time: 87.7158 ms
// Backward Convolution (dLdW) Execution Time: 0.02144 ms
// Backward Convolution (dLdX) Execution Time: 0.01536 ms