#include<iostream>
#include<stdlib.h>
#include<cuda_runtime.h>

#define TILE_SIZE 4 // tile size for merge sort
#define ARRAY_SIZE 20 // size of the array to be sorted

/*
took a small array coz to see the proper
implementation and cross check manually
*/

// min function
__device__ __host__ int my_min(int a, int b) {
    return (a < b) ? a : b;
}

// sequential merge function
__device__ void merge_sequential(int* A, int m, int* B, int n, int* C) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        if (A[i] < B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    while (i < m) {
        C[k++] = A[i++];
    }
    while (j < n) {
        C[k++] = B[j++];
    }
}

// co-rank function to determine the merge boundaries
__device__ int co_rank(int k, int* A, int m, int* B, int n) {
    int i = my_min(k,m);
    int j = k - 1;
    while (i > 0 && j < n && A[i - 1] > B[j]) {
        i--;
        j++;
    }
    return i;
}

// tiled merge sort kernel
__global__ void merge_tiled_kernel(int* A, int m, int* B, int n, int* C, int tile_size) {
    extern __shared__ int shareAB[];
    int* A_S = &shareAB[0];  
    int* B_S = &shareAB[tile_size];  

    int C_curr = blockIdx.x * ceil((m + n) / gridDim.x);  // starting point of C for current block
    int C_next = my_min((blockIdx.x + 1) * ceil((m + n) / gridDim.x), (m + n));  // starting point for next block

    // compute co-ranks for the current block
    if (threadIdx.x == 0) {
        A_S[0] = co_rank(C_curr, A, m, B, n);  
        A_S[1] = co_rank(C_next, A, m, B, n); 
    }
    __syncthreads();

    int A_curr = A_S[0];
    int A_next = A_S[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;

    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;

    int total_iteration = ceil((float)C_length / tile_size);  // total iterations needed
    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;

    while (C_completed < C_length) {
        // load tiles of A and B into shared memory
        for (int i = 0; i < tile_size; i += blockDim.x) {
            if (i + threadIdx.x < A_length - A_consumed) {
                A_S[i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x];
            }
            if (i + threadIdx.x < B_length - B_consumed) {
                B_S[i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
            }
        }
        __syncthreads();

        // merge the loaded tiles
        int c_curr = threadIdx.x * (tile_size / blockDim.x);
        int c_next = (threadIdx.x + 1) * (tile_size / blockDim.x);
        c_curr = min(c_curr, C_length - C_completed);
        c_next = min(c_next, C_length - C_completed);

        int a_curr = co_rank(c_curr, A_S, min(tile_size, A_length - A_consumed), B_S, min(tile_size, B_length - B_consumed));
        int b_curr = c_curr - a_curr;
        int a_next = co_rank(c_next, A_S, min(tile_size, A_length - A_consumed), B_S, min(tile_size, B_length - B_consumed));
        int b_next = c_next - a_next;

        merge_sequential(A_S + a_curr, a_next - a_curr, B_S + b_curr, b_next - b_curr, C + C_curr + C_completed + c_curr);

        // update consumed elements
        C_completed += tile_size;
        A_consumed += co_rank(tile_size, A_S, tile_size, B_S, tile_size);
        B_consumed = C_completed - A_consumed;
        __syncthreads();
    }
}

int main() {
    int h_A[ARRAY_SIZE / 2] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};  
    int h_B[ARRAY_SIZE / 2] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20}; 
    int h_C[ARRAY_SIZE];  

    int *d_A, *d_B, *d_C;
    size_t size = ARRAY_SIZE / 2 * sizeof(int);

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, ARRAY_SIZE * sizeof(int));

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    merge_tiled_kernel<<<1, TILE_SIZE, 2 * TILE_SIZE * sizeof(int)>>>(d_A, ARRAY_SIZE / 2, d_B, ARRAY_SIZE / 2, d_C, TILE_SIZE);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);

    cudaMemcpy(h_C, d_C, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // print sorted array
    printf("Sorted Array:\n");
    for (int i = 0; i < ARRAY_SIZE; i++) {
        printf("%d ", h_C[i]);
    }
    printf("\n");

    printf("Kernel Execution Time: %.3f ms\n", time);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
// Kernel Execution Time: 100.067 ms