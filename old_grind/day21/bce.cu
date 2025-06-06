#include <math.h>
#include <iostream>
#include <cuda_runtime.h>


// binary cross entropy kernel
__global__ void binaryCrossEntropyKernel(const float* predictions, const float* targets, float* losses, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float y_true = targets[idx];
        float y_pred = predictions[idx];
        // avoid log(0) which is undefined
        float epsilon = 1e-7;
        y_pred = fmaxf(epsilon, fminf(1.0f - epsilon, y_pred));
        losses[idx] = - (y_true * logf(y_pred) + (1 - y_true) * logf(1 - y_pred));
    }
}

// kernel to reduce the loss array to a single value
__global__ void reduceLossKernel(const float* losses, float* total_loss, int n) {
    extern __shared__ float shared_data[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    shared_data[tid] = (idx < n) ? losses[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        total_loss[blockIdx.x] = shared_data[0];
    }
}

float binaryCrossEntropy(const float* predictions, const float* targets, int n) {
    float *d_predictions, *d_targets, *d_losses, *d_total_loss;
    float h_total_loss = 0.0f;

    cudaMalloc((void**)&d_predictions, n * sizeof(float));
    cudaMalloc((void**)&d_targets, n * sizeof(float));
    cudaMalloc((void**)&d_losses, n * sizeof(float));
    cudaMalloc((void**)&d_total_loss, sizeof(float));

    cudaMemcpy(d_predictions, predictions, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, targets, n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    binaryCrossEntropyKernel<<<gridSize, blockSize>>>(d_predictions, d_targets, d_losses, n);

    reduceLossKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_losses, d_total_loss, n);

    cudaMemcpy(&h_total_loss, d_total_loss, sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);

    float time = 0;
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "Kernel Execution Time: " << time << "ms" << std::endl;

    cudaFree(d_predictions);
    cudaFree(d_targets);
    cudaFree(d_losses);
    cudaFree(d_total_loss);

    return h_total_loss / n; 
}

int main() {
    int n = 1000000;
    float predictions[n], targets[n];

    // initialize predictions and targets with some values
    for (int i = 0; i < n; ++i) {
        predictions[i] = static_cast<float>(rand()) / RAND_MAX;
        targets[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float loss = binaryCrossEntropy(predictions, targets, n);
    printf("Binary Cross-Entropy Loss: %f\n", loss);

    return 0;
}
// Kernel Execution Time: 63.3628ms