#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

using data_type = float;

// CUDA error check
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// cuBLAS error check
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "CUBLAS error: %d at %s:%d\n", status, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// utility function to print matrices
void print_matrix(int rows, int cols, const data_type *matrix, int ldm) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f", matrix[i + j * ldm]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const int m = 4096;
    const int n = 4096;
    const int k = 4096;
    const int lda = 4096;
    const int ldb = 4096;
    const int ldc = 4096;

    const std::vector<data_type> A(m *k, 1.0f);
    const std::vector<data_type> B(k*n, 1.0f);
    std::vector<data_type> C(m * n);
    const data_type alpha = 1.0f;
    const data_type beta = 0.0f;

    data_type *d_A = nullptr;
    data_type *d_B = nullptr;
    data_type *d_C = nullptr;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    // create cublas handle, bind a stream
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    // copy data to device
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(data_type) * C.size()));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice,stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice,stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));

    CUBLAS_CHECK(cublasSgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));
    CUDA_CHECK(cudaEventRecord(stop, stream));

    CUDA_CHECK(cudaEventSynchronize(stop));

    float time = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));

    CUDA_CHECK(cudaMemcpyAsync(C.data(), d_C, sizeof(data_type) * C.size(), cudaMemcpyDeviceToHost,stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    double flops = 2.0 * m * n * k;
    double tflops = (flops / (time / 1000.0)) / 1e12;

    printf("CUBLAS GEMM Execution Time: %f ms\n", time);
    printf("CUBLAS GEMM TFLOPs: %f\n", tflops);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;

}
// CUBLAS GEMM Execution Time: 84.292160 ms
// CUBLAS GEMM TFLOPs: 1.630507