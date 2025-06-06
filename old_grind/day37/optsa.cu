#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <iostream>

#define BLOCK_SIZE 32  // tile size for shared memory
#define WARP_SIZE 32

// warp-level reduction for max value
__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// warp-level reduction for sum
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// kernel for self-attention
__global__ void selfAttentionKernel(
    const float* __restrict__ Q, 
    const float* __restrict__ K, 
    const float* __restrict__ V, 
    float* __restrict__ output,  
    int seq_len,                 
    int d_k,                     
    float scale                  
) {
    // shared memory for caching tiles of Q, K, and V
    __shared__ float Q_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float K_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float V_tile[BLOCK_SIZE][BLOCK_SIZE];  

    // thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // row and column of the output matrix
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    // accumulator for the dot product
    float acc = 0.0f;

    // loop over tiles of Q and K
    for (int t = 0; t < seq_len; t += BLOCK_SIZE) {
        // load tiles of Q and K into shared memory
        if (row < seq_len && t + tx < d_k) {
            Q_tile[ty][tx] = Q[row * d_k + (t + tx)];
        } else {
            Q_tile[ty][tx] = 0.0f;
        }

        if (col < seq_len && t + ty < d_k) {
            K_tile[ty][tx] = K[col * d_k + (t + ty)];
        } else {
            K_tile[ty][tx] = 0.0f;
        }

        __syncthreads();

        // compute partial dot product using FMA
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            acc = __fadd_rn(acc, __fmul_rn(Q_tile[ty][k], K_tile[k][tx]));
        }

        __syncthreads();
    }

    // scale the attention scores
    acc = __fmul_rn(acc, scale);

    // compute softmax (max-reduction trick for numerical stability)
    float max_val = acc;
    max_val = warpReduceMax(max_val);

    // compute exp and sum using fast math functions
    float exp_val = __expf(__fsub_rn(acc, max_val));
    float sum_exp = warpReduceSum(exp_val);

    // normalize to get attention scores
    float attention_score = __fdiv_rn(exp_val, sum_exp);

    // loop over tiles of V
    for (int t = 0; t < seq_len; t += BLOCK_SIZE) {
        // load tile of V into shared memory
        if (t + tx < seq_len && col < d_k) {
            V_tile[ty][tx] = V[(t + tx) * d_k + col];
        } else {
            V_tile[ty][tx] = 0.0f;
        }

        __syncthreads();

        // multiply attention scores with V using FMA
        float out_val = 0.0f;
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            out_val = __fadd_rn(out_val, __fmul_rn(attention_score, V_tile[ty][k]));
        }

        // write to output using atomicAdd
        if (row < seq_len && col < d_k) {
            atomicAdd(&output[row * d_k + col], out_val);
        }

        __syncthreads();
    }
}

void selfAttention(
    const float* Q, const float* K, const float* V,
    float* output, int seq_len, int d_k
) {

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE, (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // scaling factor
    float scale = 1.0f / sqrtf(d_k);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);

    selfAttentionKernel<<<grid, block>>>(Q, K, V, output, seq_len, d_k, scale);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time = 0.0f;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Kernel Execution Time: " << time << "ms" << std::endl;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {

    const int seq_len = 4096;
    const int d_k = 4096;

    // allocate host memory for Q, K, V, and output
    float* h_Q = (float*)malloc(seq_len * d_k * sizeof(float));
    float* h_K = (float*)malloc(seq_len * d_k * sizeof(float));
    float* h_V = (float*)malloc(seq_len * d_k * sizeof(float));
    float* h_output = (float*)malloc(seq_len * d_k * sizeof(float));

    // initialize Q, K, V with sample values
    for (int i = 0; i < seq_len * d_k; ++i) {
        h_Q[i] = static_cast<float>(rand()) / RAND_MAX;
        h_K[i] = static_cast<float>(rand()) / RAND_MAX;
        h_V[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // allocate device memory for Q, K, V, and output
    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, seq_len * d_k * sizeof(float));
    cudaMalloc(&d_K, seq_len * d_k * sizeof(float));
    cudaMalloc(&d_V, seq_len * d_k * sizeof(float));
    cudaMalloc(&d_output, seq_len * d_k * sizeof(float));

    cudaMemcpy(d_Q, h_Q, seq_len * d_k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, seq_len * d_k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, seq_len * d_k * sizeof(float), cudaMemcpyHostToDevice);

    selfAttention(d_Q, d_K, d_V, d_output, seq_len, d_k);

    cudaMemcpy(h_output, d_output, seq_len * d_k * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_output);

    return 0;
}

// Kernel Execution Time: 336.068ms