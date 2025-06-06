#include<stdio.h>
#include<iostream>
#include<math.h>
#include<cuda_runtime.h>


// self attention kernel
__global__ void selfAttentionKernel(float* Q, float* K, float* V, float* output, int seq_len, int d_model) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; 
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < seq_len && col < seq_len) {
        // compute the dot product of Q[row] & K[col]
        float score = 0.0f;
        for (int i = 0; i < d_model; ++i) {
            score += Q[row * d_model + i] * K[col * d_model + i];
        }

        score /= sqrtf((float)d_model); // scale by sqrt(d_model)

        // compute softmax
        float exp_score = expf(score);

        // accumulate the denominator for softmax 
        float sum_exp = 0.0f;
        for (int k = 0; k < seq_len; ++k) {
            float tmp_score = 0.0f;
            for (int i = 0; i < d_model; ++i) {
                tmp_score += Q[row * d_model + i] * K[k * d_model + i];
            }
            tmp_score /= sqrtf((float)d_model);
            sum_exp += expf(tmp_score);
        }

        float attention_weight = exp_score / sum_exp;

        // compute the output
        for (int i = 0; i < d_model; ++i) {
            atomicAdd(&output[row * d_model + i], attention_weight * V[col * d_model + i]);
            // why atomicAdd -> to ensure correctness when multiple threads update the same memory location
            // simplest way to handle race conditions
        }
    }
}

// host function
void selfAttention(float* Q, float* K, float* V, float* output, int seq_len, int d_model) {
    dim3 block(32, 32);
    dim3 grid((seq_len + 31) / 32, (seq_len + 31) / 32);

    selfAttentionKernel<<<grid, block>>>(Q, K , V, output, seq_len, d_model);

    cudaDeviceSynchronize();
}

int main() {
    int seq_len = 128; // sequence length
    int d_model = 128; // model dimension

    // allocate unified memory for Q, K, V, and output
    float *Q, *K, *V, *output;
    cudaMallocManaged(&Q, seq_len * d_model * sizeof(float));
    cudaMallocManaged(&K, seq_len * d_model * sizeof(float));
    cudaMallocManaged(&V, seq_len * d_model * sizeof(float));
    cudaMallocManaged(&output, seq_len * d_model * sizeof(float));

    // initialize Q, K, V with random values
    for (int i = 0; i < seq_len * d_model; ++i) {
        Q[i] = static_cast<float>(rand()) / RAND_MAX;
        K[i] = static_cast<float>(rand()) / RAND_MAX;
        V[i] = static_cast<float>(rand()) / RAND_MAX;
        output[i] = 0.0f; // initialize output to 0
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    selfAttention(Q, K, V, output, seq_len, d_model);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time = 0.0f;
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "Kernel Execution Time: " << time << "ms" << std::endl;

    /* print the output
    printf("Output:\n");
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            printf("%f ", output[i * d_model + j]);
        }
        printf("\n");
    }*/

    // free memory
    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(output);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
// Kernel Execution Time: 9.53142ms