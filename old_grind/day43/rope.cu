#include <cuda_fp16.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

// kernel for RoPE (Rotary Positional Embedding)
__global__ void rope_kernel_optimized(__half* input, __half* output, float* theta, int seq_len, int embed_dim, int batch_size) {
    // thread indices
    int pos = blockIdx.x * blockDim.x + threadIdx.x;  // position in sequence
    int dim_pair = blockIdx.y * blockDim.y + threadIdx.y; // dimension pair index (2i)
    int batch_idx = blockIdx.z;                       // batch index

    // early exit if out of bounds
    if (pos >= seq_len || dim_pair >= embed_dim / 2 || batch_idx >= batch_size) {
        return;
    }

    // shared memory for theta
    extern __shared__ float shared_theta[];
    for (int i = threadIdx.y; i < embed_dim / 2; i += blockDim.y) {
        shared_theta[i] = theta[i];
    }
    __syncthreads();

    // compute base index for this thread
    int idx = batch_idx * seq_len * embed_dim + pos * embed_dim + 2 * dim_pair;

    // load input pair into registers (coalesced access)
    __half2 input_pair = *reinterpret_cast<__half2*>(&input[idx]);
    float x0 = __half2float(input_pair.x); // Even dimension
    float x1 = __half2float(input_pair.y); // Odd dimension

    // compute angle and trigonometric values
    float angle = pos * shared_theta[dim_pair];
    float cos_val = __cosf(angle); // hardware-accelerated cosine
    float sin_val = __sinf(angle); // hardware-accelerated sine

    // apply rotation in FP32 for precision, then convert back to FP16
    float rot_x0 = x0 * cos_val - x1 * sin_val;
    float rot_x1 = x0 * sin_val + x1 * cos_val;

    // store result as __half2 for coalesced write
    *reinterpret_cast<__half2*>(&output[idx]) = __halves2half2(__float2half(rot_x0), __float2half(rot_x1));
}

// host launch function
void launch_rope_kernel_optimized(__half* d_input, __half* d_output, float* d_theta,
                                 int seq_len, int embed_dim, int batch_size) {
    // block size: 32x16 = 512 threads
    dim3 block(32, 16);
    dim3 grid(
        (seq_len + block.x - 1) / block.x,         // cover sequence length
        (embed_dim / 2 + block.y - 1) / block.y,   // cover dimension pairs
        batch_size                                 // one block per batch
    );

    // shared memory size: embed_dim / 2 floats
    size_t shared_mem_size = (embed_dim / 2) * sizeof(float);

    rope_kernel_optimized<<<grid, block, shared_mem_size>>>(d_input, d_output, d_theta, seq_len, embed_dim, batch_size);


    // check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}

// host function to initialize data
void initialize_data(__half* h_input, float* h_theta, int seq_len, int embed_dim, int batch_size) {
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            for (int d = 0; d < embed_dim; ++d) {
                h_input[b * seq_len * embed_dim + s * embed_dim + d] = __float2half((float)(d % 10)); 
            }
        }
    }
    for (int d = 0; d < embed_dim / 2; ++d) {
        h_theta[d] = 1.0f / (10000.0f * powf(100.0f, (2.0f * d) / embed_dim)); 
    }
}

// host function to verify results
void verify_results(__half* h_output, int seq_len, int embed_dim, int batch_size) {
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            for (int d = 0; d < embed_dim; d += 2) {
                float x0 = __half2float(h_output[b * seq_len * embed_dim + s * embed_dim + d]);
                float x1 = __half2float(h_output[b * seq_len * embed_dim + s * embed_dim + d + 1]);
                printf("Batch %d, Pos %d, Dim %d: (%f, %f)\n", b+1, s, d, x0, x1);
            }
        }
    }
}

int main() {
    // parameters
    int seq_len = 512;    // sequence length
    int embed_dim = 768;  // embedding dimension
    int batch_size = 2;  // batch size

    // allocate host memory
    size_t input_size = batch_size * seq_len * embed_dim * sizeof(__half);
    size_t theta_size = (embed_dim / 2) * sizeof(float);
    __half* h_input = (__half*)malloc(input_size);
    float* h_theta = (float*)malloc(theta_size);
    __half* h_output = (__half*)malloc(input_size);

    // initialize host data
    initialize_data(h_input, h_theta, seq_len, embed_dim, batch_size);

    // allocate device memory
    __half *d_input, *d_output;
    float *d_theta;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, input_size);
    cudaMalloc(&d_theta, theta_size);

    // copy data to device
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta, h_theta, theta_size, cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    launch_rope_kernel_optimized(d_input, d_output, d_theta, seq_len, embed_dim, batch_size);

    // copy results back to host
    cudaMemcpy(h_output, d_output, input_size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time = 0.0f;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Kernel Execution Time: " << time << "ms" << std::endl;

    // verify results
    // verify_results(h_output, seq_len, embed_dim, batch_size);

    free(h_input);
    free(h_theta);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_theta);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
// Kernel Execution Time: 1.43306ms