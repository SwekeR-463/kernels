#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

using data_type = float; // fp32 

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const int m = 4096;
    const int n = 4096;
    const int k = 4096;
    const int lda = 4096;
    const int ldb = 4096;
    const int ldc = 4096;
    const int batch_count = 32;

    const long long int strideA = m * k;
    const long long int strideB = k * n;
    const long long int strideC = m * n;

    // getting values for the matrices
    const std::vector<data_type> A(m * k * batch_count, 1.0f);
    const std::vector<data_type> B(k * n * batch_count, 1.0f);
    std::vector<data_type> C(m * n * batch_count, 0.0f);
    const data_type alpha = 1.0f;
    const data_type beta = 0.0f;

    data_type *d_A = nullptr;
    data_type *d_B = nullptr;
    data_type *d_C = nullptr;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    cublasCreate(&cublasH);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    cublasSetStream(cublasH, stream);

    cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size());
    cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B.size());
    cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(data_type) * C.size());

    cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);

    // Sgemm -> single fp precision = fp32
    cublasSgemmStridedBatched(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, strideA, d_B, ldb, strideB, &beta, d_C, ldc, strideC, batch_count);
    
    cudaEventRecord(stop, stream);

    cudaEventSynchronize(stop);

    float time = 0;
    cudaEventElapsedTime(&time, start, stop);

    cudaMemcpyAsync(C.data(), d_C, sizeof(data_type) * C.size(), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    double flops = 2.0 * m * n * k * batch_count; // FLOPs for the entire batch
    double tflops = (flops / (time / 1000.0)) / 1e12;

    printf("Kernel Execution time: %f ms\n", time);
    printf("TFLOPs: %f\n", tflops);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cublasDestroy(cublasH);

    cudaStreamDestroy(stream);

    cudaDeviceReset();

    return EXIT_SUCCESS;
}
// for batch count = 64
// Kernel Execution time: 86.794205 ms
// TFLOPs: 101.344244

// for batch count = 32
// Kernel Execution time: 144.666046 ms
// TFLOPs: 30.401374