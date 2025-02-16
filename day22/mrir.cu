#include <math.h>
#include <random>
#include <iostream>
#include <cuda_runtime.h>

#define FHD_THREADS_PER_BLOCK 256
#define PI 3.14159265358979323846
#define CHUNK_SIZE 256

__constant__ float kx_c[CHUNK_SIZE], ky_c[CHUNK_SIZE], kz_c[CHUNK_SIZE];

// fourier transform of the MRI signal
__global__ void cmpFHd(float* rPhi, float* iPhi, float* phiMag,
                       float* x, float* y, float* z,
                       float* rMu, float* iMu, int M) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    float xn_r = x[n]; 
    float yn_r = y[n]; 
    float zn_r = z[n];

    float rFhDn_r = rPhi[n]; 
    float iFhDn_r = iPhi[n];

    __shared__ float rMu_shared[CHUNK_SIZE], iMu_shared[CHUNK_SIZE];
    if (threadIdx.x < CHUNK_SIZE) {
        rMu_shared[threadIdx.x] = rMu[blockIdx.x * CHUNK_SIZE + threadIdx.x];
        iMu_shared[threadIdx.x] = iMu[blockIdx.x * CHUNK_SIZE + threadIdx.x];
    }
    __syncthreads();

    for (int m = 0; m < M; m++) {
        float expFhD = 2 * PI * (kx_c[m] * xn_r + ky_c[m] * yn_r + kz_c[m] * zn_r);
        
        float cArg = __cosf(expFhD);
        float sArg = __sinf(expFhD);

        rFhDn_r += rMu_shared[m] * cArg - iMu_shared[m] * sArg;
        iFhDn_r += iMu_shared[m] * cArg + rMu_shared[m] * sArg;
    }

    rPhi[n] = rFhDn_r;
    iPhi[n] = iFhDn_r;
    phiMag[n] = sqrtf(rFhDn_r * rFhDn_r + iFhDn_r * iFhDn_r);
}

int main() {
    int N = 1024; // define problem size
    int M = 1024; // number of samples


    float *x, *y, *z, *rMu, *iMu, *rPhi, *iPhi, *phiMag;
    
    cudaError_t cudaStatus;
    cudaStatus = cudaMallocManaged(&x, N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaMallocManaged failed for x: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaStatus = cudaMallocManaged(&y, N * sizeof(float));
    cudaStatus = cudaMallocManaged(&z, N * sizeof(float));
    cudaStatus = cudaMallocManaged(&rMu, M * sizeof(float));
    cudaStatus = cudaMallocManaged(&iMu, M * sizeof(float));
    cudaStatus = cudaMallocManaged(&rPhi, N * sizeof(float));
    cudaStatus = cudaMallocManaged(&iPhi, N * sizeof(float));
    cudaStatus = cudaMallocManaged(&phiMag, N * sizeof(float));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < N; i++) {
        x[i] = dis(gen);
        y[i] = dis(gen);
        z[i] = dis(gen);
        rPhi[i] = 0.0f;
        iPhi[i] = 0.0f;
        phiMag[i] = 0.0f;
    }

    for (int i = 0; i < M; i++) {
        rMu[i] = dis(gen);
        iMu[i] = dis(gen);
    }


    // process data in chunks
    for (int i = 0; i < M / CHUNK_SIZE; i++) {
        std::cout << "\nProcessing chunk " << i + 1 << " of " << M / CHUNK_SIZE << std::endl;

        cudaStatus = cudaMemcpyToSymbol(kx_c, &rMu[i * CHUNK_SIZE], CHUNK_SIZE * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            std::cout << "cudaMemcpyToSymbol failed: " << cudaGetErrorString(cudaStatus) << std::endl;
            return 1;
        }

        cudaStatus = cudaMemcpyToSymbol(ky_c, &iMu[i * CHUNK_SIZE], CHUNK_SIZE * sizeof(float));
        cudaStatus = cudaMemcpyToSymbol(kz_c, &rMu[i * CHUNK_SIZE], CHUNK_SIZE * sizeof(float));

        int numBlocks = (N + FHD_THREADS_PER_BLOCK - 1) / FHD_THREADS_PER_BLOCK;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        cmpFHd<<<numBlocks, FHD_THREADS_PER_BLOCK>>>(rPhi, iPhi, phiMag, x, y, z, rMu, iMu, CHUNK_SIZE);
        
        cudaEventRecord(stop);

        float time = 0;
        cudaEventElapsedTime(&time, start, stop);

        std::cout << "Kernel Execution Time: " << time << "ms" << std::endl;

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            std::cout << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
            return 1;
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            std::cout << "cudaDeviceSynchronize failed: " << cudaGetErrorString(cudaStatus) << std::endl;
            return 1;
        }
    }

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    cudaFree(rMu);
    cudaFree(iMu);
    cudaFree(rPhi);
    cudaFree(iPhi);
    cudaFree(phiMag);


    return 0;
}