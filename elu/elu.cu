#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define WARP_SIZE 32
// reinterpret_cast is a type of casting operator that converts a pointer of one data type into a pointer of another data type, even if the data types are unrelated
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat16 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])



// global value alpha
#define ALPHA 1.0f

// macro for checking torch tensor dtype
#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                                      \
 if (((T).options().dtype() != (th_type))) {                                                      \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                                      \
    throw std::runtime_error("Tensor dtype must be " #th_type);                                      \
  }

#define STRINGIFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                                       \
  m.def(STRINGIFY(func), &func, STRINGIFY(func));

// fp32 elu kernel
__global__ void elu_fp32_kernel(float *x, float *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    y[idx] = elu(x[idx]);
  }
}

// fp16
__global__ void elu_fp16_kernel(half *x, half *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    y[idx] = elu_half(x[idx]);
  }
}

// batched fp16 where two fp16 values are packed into a single half2 register and processed in one go
__global__ void elu_fp16x2_kernel(half *x, half *y, int N) {
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    half2 reg_x = HALF2(x[idx]);
    half2 reg_y;
    reg_y.x = elu_half(reg_x.x);
    reg_y.y = elu_half(reg_x.y);
    HALF2(y[idx]) = reg_y;
  }
}

// batched fp16 that processes 8 half values at a time and stores in a packed 128bit vectorized store
__global__ void elu_fp16x8_pack_kernel(half *x, half *y, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  half pack_x[8], pack_y[8];
  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

  #pragma unroll // controls loop unrolling -> when this in effect the optimizer determines and applies the best unrolling factor for each loop
  for (int i = 0; i < 8; i++) {
    pack_y[i] = elu_half(pack_x[i]);
  }
  if ((idx + 7) < N) {
    LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
  }
}

// pytorch binding
#define TORCH_BINDING_ELU(packed_type, th_type, element_type, n_elements)      \
  void elu_##packed_type(torch::Tensor x, torch::Tensor y) {                   \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
    const int ndim = x.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= x.size(i);                                                        \
      }                                                                        \
      dim3 block(256 / (n_elements));                                          \
      dim3 grid((N + 256 - 1) / 256);                                          \
      elu_##packed_type##_kernel<<<grid, block>>>(                             \
          reinterpret_cast<element_type *>(x.data_ptr()),                      \
          reinterpret_cast<element_type *>(y.data_ptr()), N);                  \
    } else {                                                                   \
      const int S = x.size(0);                                                 \
      const int K = x.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        dim3 block(K / (n_elements));                                          \
        dim3 grid(S);                                                          \
        elu_##packed_type##_kernel<<<grid, block>>>(                           \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= x.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        elu_##packed_type##_kernel<<<grid, block>>>(                           \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      }                                                                        \
    }                                                                          \
  }



TORCH_BINDING_ELU(fp32, torch::kFloat32, float, 1)
TORCH_BINDING_ELU(fp16, torch::kHalf, half, 1)
TORCH_BINDING_ELU(fp16x2, torch::kHalf, half, 2)
TORCH_BINDING_ELU(fp16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(elu_fp32)
  TORCH_BINDING_COMMON_EXTENSION(elu_fp16)
  TORCH_BINDING_COMMON_EXTENSION(elu_fp16x2)
  TORCH_BINDING_COMMON_EXTENSION(elu_fp16x8_pack)
}