#include<iostream>
#include<cmath>
#include<cuda_runtime.h>

// optimized layer norm kernel
__global__ void layer_norm_kernel(const float* input, float* output, const int batch_size, const int feature_size, float epsilon) {
    extern __shared__ float shared_data[]; // shared memory for intermediate results
    int idx = blockIdx.x; // eacch block processes one sample in the batch
    int tid = threadIdx.x;

    // mean
    float mean = 0.0f;
    for (int i = tid; i < feature_size; i+= blockDim.x) {
        mean += input[idx * feature_size + i];
    }
    __syncthreads();

    // parallel reduction 
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] = mean;
        }
        __syncthreads();
        if (tid < stride) {
            mean += shared_data[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        shared_data[0] = mean / feature_size; // store mean in shared memory
    }
    __syncthreads();
    mean = shared_data[0];

    // variance
    float variance = 0.0f;
    for (int i = tid; i < feature_size; i += blockDim.x) {
        float diff = input[idx * feature_size + i] - mean;
        variance += diff * diff;
    }
    __syncthreads();

    // parallel reduction for variance
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] = variance;
        }
        __syncthreads();
        if (tid < stride) {
            variance += shared_data[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        shared_data[1] = variance / feature_size; // store variance in shared memory
    }
    __syncthreads();
    variance = shared_data[1];

    // normalize
    float inv_std = 1.0f / sqrtf(variance + epsilon);
    for (int i = tid; i < feature_size; i += blockDim.x) {
        output[idx * feature_size + i] = (input[idx * feature_size + i] - mean) * inv_std;
    }
}

void layer_norm(const float* input, float* output, int batch_size, int feature_size, float epsilon = 1e-5) {
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, batch_size * feature_size * sizeof(float));
    cudaMalloc((void**)&d_output, batch_size * feature_size * sizeof(float));

    cudaMemcpy(d_input, input, batch_size * feature_size * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int shared_memory_size = 2 * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);

    layer_norm_kernel<<<batch_size, threads_per_block, shared_memory_size>>>(d_input, d_output, batch_size, feature_size, epsilon);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(output, d_output, batch_size * feature_size * sizeof(float), cudaMemcpyDeviceToHost);

    float time = 0.0f;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Kernel Execution Time: " << time << "ms" << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    const int batch_size = 4096;
    const int feature_size = 4096;

    float* input = new float[batch_size * feature_size];
    float* output = new float[batch_size * feature_size];

    for (int i = 0; i < batch_size * feature_size; ++i) {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    layer_norm(input, output, batch_size, feature_size);

    delete[] input;
    delete[] output;

    return 0;
}
// #include<iostream>
#include<cmath>
#include<cuda_runtime.h>

// optimized layer norm kernel
__global__ void layer_norm_kernel(const float* input, float* output, const int batch_size, const int feature_size, float epsilon) {
    extern __shared__ float shared_data[]; // shared memory for intermediate results
    int idx = blockIdx.x; // eacch block processes one sample in the batch
    int tid = threadIdx.x;

    // mean
    float mean = 0.0f;
    for (int i = tid; i < feature_size; i+= blockDim.x) {
        mean += input[idx * feature_size + i];
    }
    __syncthreads();

    // parallel reduction 
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] = mean;
        }
        __syncthreads();
        if (tid < stride) {
            mean += shared_data[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        shared_data[0] = mean / feature_size; // store mean in shared memory
    }
    __syncthreads();
    mean = shared_data[0];

    // variance
    float variance = 0.0f;
    for (int i = tid; i < feature_size; i += blockDim.x) {
        float diff = input[idx * feature_size + i] - mean;
        variance += diff * diff;
    }
    __syncthreads();

    // parallel reduction for variance
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] = variance;
        }
        __syncthreads();
        if (tid < stride) {
            variance += shared_data[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        shared_data[1] = variance / feature_size; // store variance in shared memory
    }
    __syncthreads();
    variance = shared_data[1];

    // normalize
    float inv_std = 1.0f / sqrtf(variance + epsilon);
    for (int i = tid; i < feature_size; i += blockDim.x) {
        output[idx * feature_size + i] = (input[idx * feature_size + i] - mean) * inv_std;
    }
}

void layer_norm(const float* input, float* output, int batch_size, int feature_size, float epsilon = 1e-5) {
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, batch_size * feature_size * sizeof(float));
    cudaMalloc((void**)&d_output, batch_size * feature_size * sizeof(float));

    cudaMemcpy(d_input, input, batch_size * feature_size * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int shared_memory_size = 2 * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);

    layer_norm_kernel<<<batch_size, threads_per_block, shared_memory_size>>>(d_input, d_output, batch_size, feature_size, epsilon);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(output, d_output, batch_size * feature_size * sizeof(float), cudaMemcpyDeviceToHost);

    float time = 0.0f;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Kernel Execution Time: " << time << "ms" << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    const int batch_size = 4096;
    const int feature_size = 4096;

    float* input = new float[batch_size * feature_size];
    float* output = new float[batch_size * feature_size];

    for (int i = 0; i < batch_size * feature_size; ++i) {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    layer_norm(input, output, batch_size, feature_size);

    delete[] input;
    delete[] output;

    return 0;
}
// Kernel Execution Time: 0.945728ms