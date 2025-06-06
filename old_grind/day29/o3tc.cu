/****************************************************************************************
*  GEMM on Tensor Cores (FP16  →  FP32 accumulation) – 4096 × 4096 × 4096
*
*      D = A × B          (α = 1 , β = 0)
*
*  – Tile: 16×16×16   (wmma::matrix_a  = row‑major)
*                    (wmma::matrix_b  = col‑major)
*  – One warp = one 16×16 tile of D
*  – 4 warps / block  →  128 threads / block
*
*  Build (Ampere example):
*      nvcc -O3 -arch=sm_80 -o gemm_tc gemm_tc.cu
****************************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

/* ---------------------------------------------------------------------------------- */
/* problem size (must be multiples of 16)                                              */
constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;

/* leading dimensions (row major)                                                     */
constexpr int LD_A = K;
constexpr int LD_B = N;
constexpr int LD_D = N;

/* WMMA tile sizes                                                                    */
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

/* ---------------------------------------------------------------------------------- */
/* CUDA kernel                                                                         */
__global__ void gemm_tensor_cores(const __half* __restrict__ A,
                                  const __half* __restrict__ B,
                                  float*        __restrict__ D)
{
    /* ----- warp coordinates (one warp = one 16×16 tile) --------------------------- */
    const int warpM = (blockIdx.y * blockDim.y + threadIdx.y);   // tile row
    const int warpN = (blockIdx.x * blockDim.x + threadIdx.x);   // tile col

    const int row = warpM * WMMA_M;
    const int col = warpN * WMMA_N;
    if (row >= M || col >= N) return;

    /* accumulator fragment (FP32)                                                    */
    wmma::fragment<wmma::accumulator,
                   WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    /* loop over K dimension --------------------------------------------------------- */
    for (int kb = 0; kb < K; kb += WMMA_K)
    {
        /* load A tile (row‑major)                                                    */
        wmma::fragment<wmma::matrix_a,
                       WMMA_M, WMMA_N, WMMA_K,
                       __half, wmma::row_major> a_frag;

        const __half* tileA = A + row * LD_A + kb;
        wmma::load_matrix_sync(a_frag, tileA, LD_A);

        /* load B tile (col‑major)                                                    */
        wmma::fragment<wmma::matrix_b,
                       WMMA_M, WMMA_N, WMMA_K,
                       __half, wmma::col_major> b_frag;

        const __half* tileB = B + kb * LD_B + col;
        wmma::load_matrix_sync(b_frag, tileB, LD_B);

        /* FMA on tensor cores                                                        */
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    /* store result                                                                   */
    wmma::store_matrix_sync(D + row * LD_D + col,
                            c_frag, LD_D, wmma::mem_row_major);
}

/* ---------------------------------------------------------------------------------- */
/* simple helper: CUDA error checking                                                  */
#define CUDA_CALL(stmt)                                                  \
    do {                                                                 \
        cudaError_t err = (stmt);                                        \
        if (err != cudaSuccess) {                                        \
            printf("CUDA error %s at %s:%d\n",                           \
                   cudaGetErrorString(err), __FILE__, __LINE__);         \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

/* ---------------------------------------------------------------------------------- */
/* host launcher                                                                      */
void launch_gemm(const __half* dA,
                 const __half* dB,
                 float*        dD)
{
    dim3 threadsPerBlock(32, 4);             // 32 threads/warp × 4 warps = 128 thr
    dim3 blocks((N + WMMA_N*threadsPerBlock.x - 1) /
                (WMMA_N*threadsPerBlock.x),
                (M + WMMA_M*threadsPerBlock.y - 1) /
                (WMMA_M*threadsPerBlock.y));

    gemm_tensor_cores<<<blocks, threadsPerBlock>>>(dA, dB, dD);
}

/* ---------------------------------------------------------------------------------- */
/* main: allocate, run once to warm‑up, then time with cudaEvent                      */
int main()
{
    printf("Matrix size: %d × %d, K = %d  (FP16 input, FP32 output)\n", M, N, K);

    size_t bytesAB = static_cast<size_t>(M) * K * sizeof(__half);
    size_t bytesBD = static_cast<size_t>(K) * N * sizeof(__half);
    size_t bytesC  = static_cast<size_t>(M) * N * sizeof(float);

    /* host buffers (pinned for faster copies)                                        */
    __half *hA, *hB;
    float  *hD;
    CUDA_CALL(cudaMallocHost(&hA, bytesAB));
    CUDA_CALL(cudaMallocHost(&hB, bytesBD));
    CUDA_CALL(cudaMallocHost(&hD, bytesC));

    /* init A & B with simple pattern                                                 */
    for (size_t i = 0; i < (size_t)M*K; ++i)
        hA[i] = __float2half(static_cast<float>((i % 3) + 1));  // 1,2,3,...
    for (size_t i = 0; i < (size_t)K*N; ++i)
        hB[i] = __float2half(static_cast<float>(((i+1) % 5)+1)); // 1..5

    /* device buffers                                                                 */
    __half *dA, *dB;
    float  *dD;
    CUDA_CALL(cudaMalloc(&dA, bytesAB));
    CUDA_CALL(cudaMalloc(&dB, bytesBD));
    CUDA_CALL(cudaMalloc(&dD, bytesC));

    CUDA_CALL(cudaMemcpy(dA, hA, bytesAB, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dB, hB, bytesBD, cudaMemcpyHostToDevice));

    /* warm‑up                                                                       */
    launch_gemm(dA, dB, dD);
    CUDA_CALL(cudaDeviceSynchronize());

    /* create events                                                                  */
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    CUDA_CALL(cudaEventRecord(start));
    launch_gemm(dA, dB, dD);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CALL(cudaEventElapsedTime(&ms, start, stop));

    double flops = 2.0 * M * N * static_cast<double>(K);   // 2MNK FP ops
    double gflops = flops / (ms * 1.0e6);

    printf("Elapsed time : %.3f ms\n", ms);
    printf("Throughput   : %.2f GFLOP/s\n", gflops);

    /* cleanup                                                                        */
    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));
    CUDA_CALL(cudaFree(dA));
    CUDA_CALL(cudaFree(dB));
    CUDA_CALL(cudaFree(dD));
    CUDA_CALL(cudaFreeHost(hA));
    CUDA_CALL(cudaFreeHost(hB));
    CUDA_CALL(cudaFreeHost(hD));
    return 0;
}