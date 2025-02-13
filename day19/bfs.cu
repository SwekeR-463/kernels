#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_QUEUE_SIZE 1024
#define N 1000000  // e.g. size

// bfs kernel
__global__ void BFS(unsigned int *p_frontier, unsigned int *p_frontier_tail, 
                               unsigned int *c_frontier, unsigned int *c_frontier_tail, 
                               unsigned int *edges, unsigned int *dest, 
                               unsigned int *label, unsigned int *visited) {
    __shared__ unsigned int c_frontier_s[BLOCK_QUEUE_SIZE];
    __shared__ unsigned int c_frontier_tail_s, our_c_frontier_tail;

    if (threadIdx.x == 0) c_frontier_tail_s = 0;
    __syncthreads();

    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < *p_frontier_tail) {
        const unsigned int my_vertex = p_frontier[tid];
        for (unsigned int i = edges[my_vertex]; i < edges[my_vertex + 1]; ++i) {
            const unsigned int was_visited = atomicExch(&visited[dest[i]], 1);
            if (!was_visited) {
                label[dest[i]] = label[my_vertex] + 1;
                const unsigned int my_tail = atomicAdd(&c_frontier_tail_s, 1);
                if (my_tail < BLOCK_QUEUE_SIZE)
                    c_frontier_s[my_tail] = dest[i];
                else {
                    const unsigned int my_global_tail = atomicAdd(c_frontier_tail, 1);
                    c_frontier[my_global_tail] = dest[i];
                }
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        our_c_frontier_tail = atomicAdd(c_frontier_tail, c_frontier_tail_s);
    }
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < c_frontier_tail_s; i += blockDim.x) {
        c_frontier[our_c_frontier_tail + i] = c_frontier_s[i];
    }
}

int main() {
    // Allocate and initialize data
    unsigned int *d_p_frontier, *d_p_frontier_tail, *d_c_frontier, *d_c_frontier_tail;
    unsigned int *d_edges, *d_dest, *d_label, *d_visited;

    cudaMalloc(&d_p_frontier, N * sizeof(unsigned int));
    cudaMalloc(&d_p_frontier_tail, sizeof(unsigned int));
    cudaMalloc(&d_c_frontier, N * sizeof(unsigned int));
    cudaMalloc(&d_c_frontier_tail, sizeof(unsigned int));
    cudaMalloc(&d_edges, N * sizeof(unsigned int));
    cudaMalloc(&d_dest, N * sizeof(unsigned int));
    cudaMalloc(&d_label, N * sizeof(unsigned int));
    cudaMalloc(&d_visited, N * sizeof(unsigned int));

    cudaMemset(d_p_frontier_tail, 0, sizeof(unsigned int));
    cudaMemset(d_c_frontier_tail, 0, sizeof(unsigned int));
    cudaMemset(d_visited, 0, N * sizeof(unsigned int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    BFS<<<(N + 255) / 256, 256>>>(d_p_frontier, d_p_frontier_tail, d_c_frontier, d_c_frontier_tail, d_edges, d_dest, d_label, d_visited);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "Kernel Execution Time: " << time << " ms" << std::endl;

    cudaFree(d_p_frontier);
    cudaFree(d_p_frontier_tail);
    cudaFree(d_c_frontier);
    cudaFree(d_c_frontier_tail);
    cudaFree(d_edges);
    cudaFree(d_dest);
    cudaFree(d_label);
    cudaFree(d_visited);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
// Kernel Execution Time: 64.2407 ms