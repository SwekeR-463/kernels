#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

// pytorch binding for kernels
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

// fp32 kernel
__global__ void embedding_fp32_kernel(const int *idx, float *weight, float *output, int n, int emb_size) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = bx * blockDim.x + tx;
    /*
    idx[bx] selects the embedding row we want.

    offset = idx[bx] * emb_size gets the start index in the flat weight array.

    offset + tx gives the correct element within that embedding row.

    output[bx * emb_size + tx] writes that element to the corresponding position in the output.
    */
    int offset = idx[bx] * emb_size;
    output[bx * emb_size + tx] = weight[offset + tx];
}

// fp16 kernel
__global__ void embedding_fp16_kernel(const int *idx, half *weight, half *output, int n, int emb_size) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int tid = bx * blockDim.x + tx;
  int offset = idx[bx] * emb_size;
  output[bx * emb_size + tx] = weight[offset + tx];
}

#define TORCH_BINDING_EMBEDDING(packed_type, th_type, element_type,            \
                                n_elements)                                    \
  void embedding_##packed_type(torch::Tensor a, torch::Tensor weight,          \
                               torch::Tensor o) {                              \
    CHECK_TORCH_TENSOR_DTYPE(a, (torch::kInt32));                              \
    CHECK_TORCH_TENSOR_DTYPE(weight, (th_type));                               \
    CHECK_TORCH_TENSOR_DTYPE(o, (th_type));                                    \
                                                                               \
    const int N = a.size(0);                                                   \
    const int emb_size = weight.size(1);                                       \
    dim3 block(emb_size / n_elements);                                         \
    dim3 grid(N);                                                              \
    embedding_##packed_type##_kernel<<<grid, block>>>(                         \
        reinterpret_cast<int *>(a.data_ptr()),                                 \
        reinterpret_cast<element_type *>(weight.data_ptr()),                   \
        reinterpret_cast<element_type *>(o.data_ptr()), N, emb_size);          \
  }

TORCH_BINDING_EMBEDDING(fp32, torch::kFloat32, float, 1)
TORCH_BINDING_EMBEDDING(fp16, torch::kHalf, half, 1)


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(embedding_fp32);
  TORCH_BINDING_COMMON_EXTENSION(embedding_fp16);
}