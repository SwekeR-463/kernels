#include <cuda.h>
#include <cuda_runtime.h>

__global__ void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;  // batch and head index

    // offset into Q,K,V,O,l,m
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  
    int lm_offset = (bx * gridDim.y * N) + (by * N);  

    // shared memory
    extern __shared__ float sram[];
    int tile_size = Bc * d;
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++) {
        // load Kj, Vj into shared memory (coalesced access)
        int k_idx = qkv_offset + (Bc * j * d) + (tx * d);
        int v_idx = qkv_offset + (Bc * j * d) + (tx * d);
        for (int x = 0; x < d; x++) {
            Kj[tx * d + x] = K[k_idx + x];
            Vj[tx * d + x] = V[v_idx + x];
        }
        __syncthreads();

        for (int i = 0; i < Tr; i++) {
            // load Qi into shared memory
            int q_idx = qkv_offset + (Br * i * d) + (tx * d);
            for (int x = 0; x < d; x++) {
                Qi[tx * d + x] = Q[q_idx + x];
            }

            // load previous l and m from global memory
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // compute S = QK^T using warp parallelism
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0.0f;
                for (int x = 0; x < d; x += 32) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[tx * Bc + y] = sum;

                row_m = fmaxf(row_m, sum);
            }
            __syncthreads();

            // softmax computation
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[tx * Bc + y] = __expf(S[tx * Bc + y] - row_m);
                row_l += S[tx * Bc + y];
            }

            // compute new l and m
            float row_m_new = fmaxf(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // compute weighted sum of V
            for (int x = 0; x < d; x++) {
                float pv = 0.0f;
                for (int y = 0; y < Bc; y++) {
                    pv += S[tx * Bc + y] * Vj[y * d + x];
                }
                int o_idx = qkv_offset + (Br * i * d) + (tx * d) + x;
                O[o_idx] = (1 / row_l_new) * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[o_idx]) + (__expf(row_m - row_m_new) * pv));
            }

            // store updated l and m
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();
    }
}
