#include<iostream>
#include<cuda_runtime.h>

#define BLOCK_QUEUE_SIZE 1024
#define N 1000000 // e.g. size

// bfs kernel
__global__ void BFS(unsigned int *p_frontier, unsigned int *p_frontier_tail, unsigned int *c_frontier, unsigned int *c_frontier_tail, unsigned int *edges, unsigned int *dest, unsigned int *label, unsigned int *visited) {
    // COOKING IT....
}