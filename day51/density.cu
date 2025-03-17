#include <iostream>
#include <cuda_runtime.h>

#define N 1024 // number of unit cells

__global__ void densityKernel(float *Z, float *M, float *a, float *rho, float N_A) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        rho[idx] = (Z[idx] * M[idx]) / (powf(a[idx], 3) * N_A);
    }
}

int main() {
    float *h_Z, *h_M, *h_a, *h_rho;
    float *d_Z, *d_M, *d_a, *d_rho;
    float N_A = 6.022e23; // Avogadro's number

    h_Z = new float[N];
    h_M = new float[N];
    h_a = new float[N];
    h_rho = new float[N];

    for (int i = 0; i < N; i++) {
        h_Z[i] = (i % 4) + 1; // random Z values
        h_M[i] = 50.0f + i;   // example molar mass
        h_a[i] = 0.1f + 0.01f * (i % 10); // example lattice parameter
    }

    cudaMalloc(&d_Z, N * sizeof(float));
    cudaMalloc(&d_M, N * sizeof(float));
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_rho, N * sizeof(float));

    cudaMemcpy(d_Z, h_Z, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    densityKernel<<<blocksPerGrid, threadsPerBlock>>>(d_Z, d_M, d_a, d_rho, N_A);
    
    cudaMemcpy(h_rho, d_rho, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);

    float time = 0.0f;
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "Kernel Execution Time: " << time << "ms" << std::endl;

    /*print some results
    for (int i = 0; i < 5; i++) {
        std::cout << "Density of unit cell " << i << " = " << h_rho[i] << " g/cm^3" << std::endl;
    }*/

    delete[] h_Z;
    delete[] h_M;
    delete[] h_a;
    delete[] h_rho;
    cudaFree(d_Z);
    cudaFree(d_M);
    cudaFree(d_a);
    cudaFree(d_rho);

    return 0;
}
// Kernel Execution Time: 0.204864ms