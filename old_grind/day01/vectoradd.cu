#include<iostream>
#include<math.h>

// addition kernel
__global__
void add(int n, float *x, float *y) {
    int index = threadIdx.x; // the index of current thread within its block
    int stride = blockDim.x; // the no. of threads in the block
    for (int i = index; i < n; i += stride) {
        // printf("thread %d", index);
        // printf("threads in block %d", stride);
        y[i] = x[i] + y[i];
    }
}

int main(void) {
    int N = 1<<20; // 1 million
    float *x, *y;

    // allocate unified memory - accessible from GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    for(int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }


    // run the kernel 
    add<<<1, 256>>>(N, x, y); // <<<numBlocks, blockSize>>>

    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for(int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i]-2.7f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    // free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}