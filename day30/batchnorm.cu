#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void batchnorm_forward_kernel(
    const float* input,  // input tensor (N, D)
    float* output,       // output tensor (N, D)
    const float* gamma,  // scale parameter (D,)
    const float* beta,   // shift parameter (D,)
    float* mean,         // mean computed per feature (D,)
    float* var,          // variance computed per feature (D,)
    int N, int D,        // input dimensions
    float eps            // small constant for numerical stability
) {
    // feature dimension index
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (d < D) {
        // compute mean for the feature dimension
        float sum = 0.0f;
        for (int n = 0; n < N; ++n) {
            sum += input[n * D + d];
        }
        mean[d] = sum / N;

        // compute variance for the feature dimension
        float sum_sq = 0.0f;
        for (int n = 0; n < N; ++n) {
            float diff = input[n * D + d] - mean[d];
            sum_sq += diff * diff;
        }
        var[d] = sum_sq / N;

        // normalize and apply scale/shift
        for (int n = 0; n < N; ++n) {
            float normalized = (input[n * D + d] - mean[d]) / sqrtf(var[d] + eps);
            output[n * D + d] = gamma[d] * normalized + beta[d];
        }
    }
}

void batchnorm_forward(
    const float* input, float* output,
    const float* gamma, const float* beta,
    float* mean, float* var,
    int N, int D, float eps
) {
    int block_size = 256;
    int grid_size = (D + block_size - 1) / block_size;

    batchnorm_forward_kernel<<<grid_size, block_size>>>(
        input, output, gamma, beta, mean, var, N, D, eps
    );

    CUDA_CHECK(cudaGetLastError());
}

int main() {
    int N = 32, D = 128; // Batch size and feature dimension
    int input_size = N * D;
    int param_size = D;

    float* h_input = new float[input_size];
    float* h_output = new float[input_size];
    float* h_gamma = new float[param_size];
    float* h_beta = new float[param_size];
    float* h_mean = new float[param_size];
    float* h_var = new float[param_size];

    for (int i = 0; i < input_size; ++i) h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < param_size; ++i) {
        h_gamma[i] = 1.0f;
        h_beta[i] = 0.0f;
    }

    float *d_input, *d_output, *d_gamma, *d_beta, *d_mean, *d_var;
    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, param_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, param_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean, param_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_var, param_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma, param_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta, param_size * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    batchnorm_forward(d_input, d_output, d_gamma, d_beta, d_mean, d_var, N, D, 1e-5);

    CUDA_CHECK(cudaMemcpy(h_output, d_output, input_size * sizeof(float), cudaMemcpyDeviceToHost));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time = 0.0f;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Kernel Execution Time: " << time << "ms" << std::endl;

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_var));

    delete[] h_input;
    delete[] h_output;
    delete[] h_gamma;
    delete[] h_beta;
    delete[] h_mean;
    delete[] h_var;

    return 0;
}
// Kernel Execution Time: 0.310496ms