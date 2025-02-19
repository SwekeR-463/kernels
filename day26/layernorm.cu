#include <iostream>
#include <cuda_runtime.h>

// layernorm kernel
__global__ void layer_norm_kernel(const float* input, float* output, const int batch_size, const int feature_size, float epsilon) {
    // COOKING ITTTTTTTTTTTTTTT
}