#include <iostream>
#include <cuda_runtime.h>

// spmv csr kernel
__global__ void spmvCSRKernel(int num_rows, const float* values, const int* col_indices, const int* row_ptr, const float* x, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot_product = 0.0f;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        for (int j = row_start; j < row_end; j++) {
            dot_product += values[j] * x[col_indices[j]];
        }
        y[row] = dot_product;
    }
}

void spmvCSR(int num_rows, const float* values, const int* col_indices, const int* row_ptr, const float* x, float* y) {
    float *d_values, *d_x, *d_y;
    int *d_col_indices, *d_row_ptr;

    cudaMalloc((void**)&d_values, sizeof(float) * row_ptr[num_rows]);
    cudaMalloc((void**)&d_col_indices, sizeof(int) * row_ptr[num_rows]);
    cudaMalloc((void**)&d_row_ptr, sizeof(int) * (num_rows + 1));
    cudaMalloc((void**)&d_x, sizeof(float) * num_rows);
    cudaMalloc((void**)&d_y, sizeof(float) * num_rows);

    cudaMemcpy(d_values, values, sizeof(float) * row_ptr[num_rows], cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, col_indices, sizeof(int) * row_ptr[num_rows], cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr, row_ptr, sizeof(int) * (num_rows + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, sizeof(float) * num_rows, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (num_rows + threads_per_block - 1) / threads_per_block;
    spmvCSRKernel<<<blocks_per_grid, threads_per_block>>>(num_rows, d_values, d_col_indices, d_row_ptr, d_x, d_y);

    cudaMemcpy(y, d_y, sizeof(float) * num_rows, cudaMemcpyDeviceToHost);

    cudaFree(d_values);
    cudaFree(d_col_indices);
    cudaFree(d_row_ptr);
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    int num_rows = 1000000; // no of rows(n * n)
    int nnz = 5000000; // no of non-zero elements

    float* values = new float[nnz];
    int* col_indices = new int[nnz];
    int* row_ptr = new int[num_rows + 1];
    float* x = new float[num_rows];
    float* y = new float[num_rows];

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    spmvCSR(num_rows, values, col_indices, row_ptr, x, y);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Kernel execution time: " << time << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] values;
    delete[] col_indices;
    delete[] row_ptr;
    delete[] x;
    delete[] y;

    return 0;
}
// Kernel execution time: 8.58669 ms