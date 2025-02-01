#include<iostream>
#include<cuda_runtime.h>

#define BLOCK_SIZE 32

// mat mul kernel
__global__ void matrixMultiply(float *A, float *B, float *C, int N, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < d) {
        float sum = 0.0;
        for (int i = 0; i < d; ++i) {
            sum += A[row * d + i] * B[i * d + col];
        }
        C[row * d + col] = sum;
    }
}

// softmax kernel
__global__ void softmax(float *mat, int N, int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N) {
        float max_val = -1e9;
        for (int i = 0; i < d; i++) {
            max_val = fmaxf(max_val, mat[row * d +i]);
        }

        float sum_exp = 0.0;
        for (int i = 0; i < d; i++) {
            mat[row * d + i] = expf(mat[row * d + i] - max_val);
            sum_exp += mat[row * d + i];
        }

        for (int i = 0; i < d; i++) {
            mat[row * d + i] /= sum_exp;
        }
    }
} 

void lightningAttention(float *Q, float *K, float *V, float *O, int N, int d, int B) {
    int T = N / B;

    float *d_Q, *d_K, *d_V, *d_O, *d_KV, *d_O_intra, *d_O_inter;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(&d_Q, N * d * sizeof(float));
    cudaMalloc(&d_K, N * d * sizeof(float));
    cudaMalloc(&d_V, N * d * sizeof(float));
    cudaMalloc(&d_O, N * d * sizeof(float));
    cudaMalloc(&d_KV, d * d * sizeof(float));
    cudaMalloc(&d_O_intra, B * d * sizeof(float));
    cudaMalloc(&d_O_inter, B * d * sizeof(float));

    cudaMemcpy(d_Q, Q, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, N * d * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((d + BLOCK_SIZE - 1) / BLOCK_SIZE, (B + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEventRecord(start);

    for (int t = 0; t < T; t++) {
        float *Q_block = d_Q + t * B * d;
        float *K_block = d_K + t * B * d;
        float *V_block = d_V + t * B * d;

        matrixMultiply<<<gridSize, blockSize>>>(Q_block, K_block, d_O_intra, B, d);
        softmax<<<B, 1>>>(d_O_intra, B, d);
        matrixMultiply<<<gridSize, blockSize>>>(d_O_intra, V_block, d_O_intra, B, d);

        matrixMultiply<<<gridSize, blockSize>>>(Q_block, d_KV, d_O_inter, B, d);
        matrixMultiply<<<gridSize, blockSize>>>(K_block, V_block, d_KV, d, d);

        cudaMemcpy(d_O + t * B * d, d_O_intra, B * d * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_O + t * B * d, d_O_inter, B * d * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time = 0.0;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Kernel Execution Time: " << time << "ms" << std::endl;

    cudaMemcpy(O, d_O, N * d * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_KV);
    cudaFree(d_O_intra);
    cudaFree(d_O_inter);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}

int main() {
    int N = 64, d = 128, B = 16;

    float *Q = new float[N * d];
    float *K = new float[N * d];
    float *V = new float[N * d];
    float *O = new float[N * d];

    for (int i = 0; i < N * d; i++) {
        Q[i] = static_cast<float>(rand()) / RAND_MAX;
        K[i] = static_cast<float>(rand()) / RAND_MAX;
        V[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    lightningAttention(Q, K, V, O, N, d, B);

    delete[] Q;
    delete[] K;
    delete[] V;
    delete[] O;

    return 0;
}
// Kernel Execution Time: 50.7443ms