#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

using data_type = double;

// utility function to print a vector
void print_vector(size_t n, const data_type* vec) {
    for (size_t i = 0; i < n; i++) {
        printf("%0.2f ", vec[i]);
    }
    printf("\n");
}

// error checking macro for CUDA API calls
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// error checking macro for cuBLAS API calls
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = (call); \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error: %d at %s:%d\n", status, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    /*
     *   A = | 1.0 2.0 3.0 4.0 |
     *   B = | 5.0 6.0 7.0 8.0 |
     */

    const std::vector<data_type> A = {1.0, 2.0, 3.0, 4.0};
    const std::vector<data_type> B = {5.0, 6.0, 7.0, 8.0};
    const int incx = 1;
    const int incy = 1;

    data_type result = 0.0;

    data_type *d_A = nullptr;
    data_type *d_B = nullptr;

    printf("A\n");
    print_vector(A.size(), A.data());
    printf("=====\n");

    printf("B\n");
    print_vector(B.size(), B.data());
    printf("=====\n");

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B.size()));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice,
                               stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));

    /* step 3: compute */
    CUBLAS_CHECK(cublasDdot(cublasH, A.size(), d_A, incx, d_B, incy, &result));

    CUDA_CHECK(cudaEventRecord(stop, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    float time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));

    /*
     *   result = 70.00
     */

    printf("Result\n");
    printf("%0.2f\n", result);
    printf("=====\n");

    printf("Kernel execution time: %f ms\n", time);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
// Kernel execution time: 0.982880 ms