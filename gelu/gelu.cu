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

#define WARP_SIZE 32
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (*reinterpret_cast<float2 *>(&(value)))


#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)
#define SQRT_2_PI M_SQRT2 *M_2_SQRTPI * 0.5f
#define HALF_1 __float2half(1.0f)
#define HALF_2 __float2half(2.0f)
#define HALF_DIV2 __float2half(0.5f)

#define HALF_SQRT_2_PI                                                         \
  __float2half(M_SQRT2) * __float2half(M_2_SQRTPI) * HALF_DIV2
#define HALF_V_APP __float2half(0.044715f)

#define HALF_GELU_OPS gelu_tanh_approximate
#define GELU_OPS gelu_tanh_approximate

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

__inline__ __device__ half gelu_tanh_approximate(half x) {
  half x_cube = x * x * x;
  // compute mid value : inner = 0.7978845608 * (x + 0.044715 * x * x * x)
  half inner = HALF_SQRT_2_PI * (x + HALF_V_APP * x_cube);
  // compute tanh
  return HALF_DIV2 * x *
         (HALF_1 +
          ((hexp(inner * HALF_2) - HALF_1) / (hexp(inner * HALF_2) + HALF_1)));
}

__inline__ __device__ float gelu_tanh_approximate(float x) {
  return 0.5f * x * (1.0f + tanhf(SQRT_2_PI * (x + 0.044715f * x * x * x)));
}

// fp32 kernel
__global__ void gelu_f32_kernel(float *x, float *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float v = fminf(fmaxf(x[idx], MIN_EXP_F32), MAX_EXP_F32);
    y[idx] = GELU_OPS(v);
  }
}

// fp16
__global__ void gelu_f16_kernel(half *x, half *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    half v = x[idx];
    v = __hmin(__hmax(v, MIN_EXP_F16), MAX_EXP_F16);

    y[idx] = HALF_GELU_OPS(v);
  }
}

// fp16x4 packed

__global__ void gelu_f16x4_pack_kernel(half *x, half *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

  half pack_x[4], pack_y[4];

  // Load 64 bits = 4 half = 2 x half2
  LDST64BITS(pack_x[0]) = LDST64BITS(x[idx]);

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    half v = __hmin(__hmax(pack_x[i], MIN_EXP_F16), MAX_EXP_F16);
    pack_y[i] = HALF_GELU_OPS(v);
  }

  if ((idx + 3) < N) {
    LDST64BITS(y[idx]) = LDST64BITS(pack_y[0]);
  }
}

/*__global__ void gelu_f16x4_pack_kernel(half *x, half *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

  half2 reg_x[4], reg_y[4];
  // Load 8 elements as 4 half2's
  reg_x[0] = HALF2(x[idx + 0]);
  reg_x[1] = HALF2(x[idx + 2]);
  reg_x[2] = HALF2(x[idx + 4]);
  reg_x[3] = HALF2(x[idx + 6]);

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    reg_x[i].x = __hmin(__hmax(reg_x[i].x, MIN_EXP_F16), MAX_EXP_F16);
    reg_x[i].y = __hmin(__hmax(reg_x[i].y, MIN_EXP_F16), MAX_EXP_F16);

    reg_y[i].x = HALF_GELU_OPS(reg_x[i].x);
    reg_y[i].y = HALF_GELU_OPS(reg_x[i].y);
  }

  // Store results safely
  if ((idx + 0) < N) HALF2(y[idx + 0]) = reg_y[0];
  if ((idx + 2) < N) HALF2(y[idx + 2]) = reg_y[1];
  if ((idx + 4) < N) HALF2(y[idx + 4]) = reg_y[2];
  if ((idx + 6) < N) HALF2(y[idx + 6]) = reg_y[3];
}*/

#define TORCH_BINDING_GELU(packed_type, th_type, element_type, n_elements)     \
  void gelu_##packed_type(torch::Tensor x, torch::Tensor y) {                  \
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
      gelu_##packed_type##_kernel<<<grid, block>>>(                            \
          reinterpret_cast<element_type *>(x.data_ptr()),                      \
          reinterpret_cast<element_type *>(y.data_ptr()), N);                  \
    } else {                                                                   \
      const int S = x.size(0);                                                 \
      const int K = x.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        dim3 block(K / (n_elements));                                          \
        dim3 grid(S);                                                          \
        gelu_##packed_type##_kernel<<<grid, block>>>(                          \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= x.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        gelu_##packed_type##_kernel<<<grid, block>>>(                          \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      }                                                                        \
    }                                                                          \
  }

TORCH_BINDING_GELU(f32, torch::kFloat32, float, 1)
TORCH_BINDING_GELU(f16, torch::kHalf, half, 1)
TORCH_BINDING_GELU(f16x4_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(gelu_f32)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f16)
  TORCH_BINDING_COMMON_EXTENSION(gelu_f16x4_pack)
}