#include<iostream>
#include<cuda_runtime.h>
#include<cublas_v2.h>

// im2col kernel
__global__ void im2colKernel(const float* input, float* im2col,
                             int input_channels, int input_height, int input_width,
                             int kernel_size, int stride, int output_height, int output_width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < output_width && row < output_height) {
        for (int c = 0; c < input_channels; ++c) {
            for (int k_r = 0; k_r < kernel_size; ++k_r) {
                for (int k_c = 0; k_c < kernel_size; ++k_c) {
                    int in_r = row * stride + k_r;
                    int in_c = col * stride + k_c;
                    int input_idx = c * input_height * input_width + in_r * input_width + in_c;
                    int im2col_idx = (c * kernel_size * kernel_size + k_r * kernel_size + k_c) * (output_height * output_width)
                                     + row * output_width + col;
                    im2col[im2col_idx] = input[input_idx];
                }
            }
        }
    }
}

// gemm -> Sgemm -> single precision gemm 
void convForward(const float* input, const float* weights, float* output,
                            int input_channels, int input_height, int input_width,
                            int num_filters, int kernel_size, int stride,
                            int output_height, int output_width) {
    int im2col_size = input_channels * kernel_size * kernel_size * output_height * output_width;
    float* d_im2col;
    cudaMalloc(&d_im2col, im2col_size * sizeof(float));

    // launch im2col kernel
    dim3 blockSize(16, 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    dim3 gridSize((output_width + 15) / 16, (output_height + 15) / 16);
    im2colKernel<<<gridSize, blockSize>>>(input, d_im2col, input_channels, input_height, input_width,
                                          kernel_size, stride, output_height, output_width);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float t1 = 0.0f;
    cudaEventElapsedTime(&t1, start, stop);

    float alpha = 1.0f, beta = 0.0f;
    // this part is inspired from https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLAS/Level-3/gemm/cublas_gemm_example.cu
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEventRecord(start);


    // A: weights [num_filters, input_channels * kernel_size * kernel_size]
    // B: im2col  [input_channels * kernel_size * kernel_size, output_height * output_width]
    // C: output  [num_filters, output_height * output_width]

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                output_height * output_width, num_filters, input_channels * kernel_size * kernel_size,
                &alpha,
                d_im2col, output_height * output_width,
                weights, input_channels * kernel_size * kernel_size,
                &beta,
                output, output_height * output_width);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float t2 = 0.0f;
    cudaEventElapsedTime(&t2, start, stop);

    std::cout << "Kernel Execution Time: " << t1 + t2 << "ms" << std::endl;

    cublasDestroy(handle);
    cudaFree(d_im2col);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int input_channels = 1;
    int input_height = 24, input_width = 24;
    int num_filters = 3;
    int kernel_size = 3;
    int stride = 1;
    int output_height = (input_height - kernel_size) / stride + 1;
    int output_width = (input_width - kernel_size) / stride + 1;

    int input_size = input_channels * input_height * input_width;
    int weight_size = num_filters * input_channels * kernel_size * kernel_size;
    int output_size = num_filters * output_height * output_width;

    float* h_input = new float[input_size];
    float* h_weights = new float[weight_size];
    float* h_output = new float[output_size];

    for (int i = 0; i < input_size; i++) h_input[i] = 1.0f;
    for (int i = 0; i < weight_size; i++) h_weights[i] = 0.5f;

    float *d_input, *d_weights, *d_output;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_weights, weight_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, weight_size * sizeof(float), cudaMemcpyHostToDevice);

    convForward(d_input, d_weights, d_output, input_channels, input_height, input_width,
                num_filters, kernel_size, stride, output_height, output_width);

    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    delete[] h_input;
    delete[] h_weights;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);

    return 0;
}
