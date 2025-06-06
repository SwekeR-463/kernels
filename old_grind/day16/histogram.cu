#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// histogram privatized kernel
__global__ void histogram_privatized_kernel(unsigned char* input, unsigned int* bins, 
                                            unsigned int num_elements, unsigned int num_bins) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ unsigned int histo_s[];

    // initialize shared memory
    for (unsigned int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x) {
        histo_s[binIdx] = 0u;
    }
    __syncthreads();

    unsigned int prev_index = -1;
    unsigned int accumulator = 0;

    // compute histogram
    for (unsigned int i = tid; i < num_elements; i += blockDim.x * gridDim.x) {
        int alphabet_position = input[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            unsigned int curr_index = alphabet_position / 4;
            if (curr_index != prev_index) {
                if (accumulator >= 0) atomicAdd(&(histo_s[curr_index]), accumulator);
                accumulator = 1;
                prev_index = curr_index;
            } else {
                accumulator++;
            }
        }
    }
    __syncthreads();

    // commit to global memory
    for (unsigned int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x) {
        atomicAdd(&(bins[binIdx]), histo_s[binIdx]);
    }
}

int main() {
    const unsigned int num_elements = 1000000; // 1 million inputs
    const unsigned int num_bins = 26; // 26 letters in the alphabet
    const unsigned int block_size = 256; 
    const unsigned int grid_size = (num_elements + block_size - 1) / block_size; 

    unsigned char* h_input = (unsigned char*)malloc(num_elements * sizeof(unsigned char));
    unsigned int* h_bins = (unsigned int*)malloc(num_bins * sizeof(unsigned int));

    for (unsigned int i = 0; i < num_elements; i++) {
        h_input[i] = 'a' + (i % 26); // fill with 'a' to 'z'
    }
    memset(h_bins, 0, num_bins * sizeof(unsigned int));

    unsigned char* d_input;
    unsigned int* d_bins;
    CUDA_CHECK(cudaMalloc((void**)&d_input, num_elements * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc((void**)&d_bins, num_bins * sizeof(unsigned int)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, num_elements * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bins, h_bins, num_bins * sizeof(unsigned int), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    histogram_privatized_kernel<<<grid_size, block_size, num_bins * sizeof(unsigned int)>>>(d_input, d_bins, num_elements, num_bins);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));

    float time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
    std::cout << "Kernel execution time: " << time << " ms" << std::endl;

    CUDA_CHECK(cudaMemcpy(h_bins, d_bins, num_bins * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_bins));
    free(h_input);
    free(h_bins);

    return 0;
}
// Kernel execution time: 60.6269 ms