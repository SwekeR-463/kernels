#include<iostream>
#include<algorithm>
#include<vector>
#include<cuda_runtime.h>

// spmv jds kernel
__global__ void spmvJDSKernel(float *data, int *col_indices, int *jagged_ptr, int *perm, float *x, float *y, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        float sum = 0.0f;
        int diag_start = jagged_ptr[row];
        int diag_end = jagged_ptr[row + 1];

        for (int i = diag_start; i < diag_end; i++) {
            sum += data[i] * x[col_indices[i]];
        }  

        y[perm[row]] = sum;
    }
}

// function to generate a random sparse matrix
void randSparseMatrix(int num_rows, int num_cols, float density, std::vector<float>& data, std::vector<int>& col_indices, std::vector<int>& row_ptr) {
    data.clear();
    col_indices.clear();
    row_ptr.resize(num_rows + 1, 0);

    for(int i = 0; i < num_rows; i++) {
        for(int j = 0; j < num_cols; j++) {
            if (static_cast<float>(rand()) / RAND_MAX / density) {
                data.push_back(static_cast<float>(rand()) /  RAND_MAX);
                col_indices.push_back(j);
                row_ptr[i + 1]++;
            }
        }
    }
    for (int i = 1; i <= num_rows; i++) {
        row_ptr[i] += row_ptr[i - 1];
    }
}

// csr jds
void csr2jds(std::vector<float>& csr_data, std::vector<int>& csr_col_indices, std::vector<int>& csr_row_ptr, std::vector<float>& jds_data, std::vector<int>& jds_col_indices, std::vector<int>& jds_jagged_ptr, std::vector<int>& perm, int num_rows) {
    std::vector<std::pair<int, int>> row_nnz(num_rows);
    for (int i = 0; i < num_rows; i++) {
        row_nnz[i] = {csr_row_ptr[i + 1] - csr_row_ptr[i], i};
    }
    std::sort(row_nnz.begin(), row_nnz.end(), std::greater<std::pair<int, int>>());

    jds_data.clear();
    jds_col_indices.clear();
    jds_jagged_ptr.resize(num_rows + 1, 0);
    perm.resize(num_rows);

    for (int i = 0; i < num_rows; i++) {
        int original_row = row_nnz[i].second;
        perm[i] = original_row;

        int row_start = csr_row_ptr[original_row];
        int row_end = csr_row_ptr[original_row + 1];

        for (int j = row_start; j < row_end; j++) {
            jds_data.push_back(csr_data[j]);
            jds_col_indices.push_back(csr_col_indices[j]);
        }
        jds_jagged_ptr[i + 1] = jds_jagged_ptr[i] + (row_end - row_start);
    }
}

void spmvJDS(float *data, int *col_indices, int *jagged_ptr, int *perm, float *x, float *y, int num_rows) {
    int blockSize = 256;
    int gridSize = (num_rows + blockSize - 1) / blockSize;

    spmvJDSKernel<<<gridSize, blockSize>>>(data, col_indices, jagged_ptr, perm, x, y, num_rows);

    cudaDeviceSynchronize();
}

int main() {
    int num_rows = 10000; 
    int num_cols = 100000;
    float density = 0.1; // 1% non-zero elements

    // generate a random sparse matrix in CSR format
    std::vector<float> csr_data;
    std::vector<int> csr_col_indices, csr_row_ptr;
    randSparseMatrix(num_rows, num_cols, density, csr_data, csr_col_indices, csr_row_ptr);

    // convert CSR to JDS format
    std::vector<float> jds_data;
    std::vector<int> jds_col_indices, jds_jagged_ptr, perm;
    csr2jds(csr_data, csr_col_indices, csr_row_ptr, jds_data, jds_col_indices, jds_jagged_ptr, perm, num_rows);

    std::vector<float> x(num_cols, 1.0f); 
    std::vector<float> y(num_rows, 0.0f); 

    float *d_data, *d_x, *d_y;
    int *d_col_indices, *d_jagged_ptr, *d_perm;

    cudaMalloc((void**)&d_data, jds_data.size() * sizeof(float));
    cudaMalloc((void**)&d_col_indices, jds_col_indices.size() * sizeof(int));
    cudaMalloc((void**)&d_jagged_ptr, (num_rows + 1) * sizeof(int));
    cudaMalloc((void**)&d_perm, num_rows * sizeof(int));
    cudaMalloc((void**)&d_x, num_cols * sizeof(float));
    cudaMalloc((void**)&d_y, num_rows * sizeof(float));

    cudaMemcpy(d_data, jds_data.data(), jds_data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, jds_col_indices.data(), jds_col_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_jagged_ptr, jds_jagged_ptr.data(), (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_perm, perm.data(), num_rows * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), num_cols * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    spmvJDS(d_data, d_col_indices, d_jagged_ptr, d_perm, d_x, d_y, num_rows);

    cudaMemcpy(y.data(), d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);

    float time = 0.0f;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Kernel Execution Time: " << time << "ms" << std::endl;

    cudaFree(d_data);
    cudaFree(d_col_indices);
    cudaFree(d_jagged_ptr);
    cudaFree(d_perm);
    cudaFree(d_x);
    cudaFree(d_y);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}