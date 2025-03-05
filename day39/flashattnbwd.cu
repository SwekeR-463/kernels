#include <cuda.h>
#include <cuda_runtime.h>

__global__ void backward_kernel(
    const float* Q, const float* K, const float* V, const float* dO,  // input tensors and output gradient
    const float* l, const float* m,  // intermediate results from forward pass
    const int N, const int d,        // sequence length and embedding dimension
    const int Tc, const int Tr,      // number of tiles in columns and rows
    const int Bc, const int Br,      // tile sizes for columns and rows
    const float softmax_scale,       // scaling factor for softmax
    float* dQ, float* dK, float* dV  // gradients to compute
) {
    int tx = threadIdx.x;            // thread index within a block
    int bx = blockIdx.x, by = blockIdx.y;  // block indices for batch and head

    // compute offsets for Q, K, V, dO, dQ, dK, dV, l, and m
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // offset for Q, K, V
    int lm_offset = (bx * gridDim.y * N) + (by * N);           // offset for l and m

    // shared memory
    extern __shared__ float sram[];  
    int tile_size = Bc * d;          // size of a tile in shared memory
    float* Qi = sram;                // tile for Q
    float* Kj = &sram[tile_size];    // tile for K
    float* Vj = &sram[tile_size * 2];// tile for V
    float* S = &sram[tile_size * 3]; // tile for attention scores
    float* dS = &sram[tile_size * 4];// tile for gradients of attention scores

    // loop over column tiles
    for (int j = 0; j < Tc; j++) {
        // load Kj and Vj into shared memory (coalesced access)
        int k_idx = qkv_offset + (Bc * j * d) + (tx * d);  // index for K
        int v_idx = qkv_offset + (Bc * j * d) + (tx * d);  // index for V
        for (int x = 0; x < d; x++) {
            Kj[tx * d + x] = K[k_idx + x];  // load K into shared memory
            Vj[tx * d + x] = V[v_idx + x];  // load V into shared memory
        }
        __syncthreads();  // synchronize threads to ensure shared memory is loaded

        // loop over row tiles 
        for (int i = 0; i < Tr; i++) {
            // load Qi into shared memory
            int q_idx = qkv_offset + (Br * i * d) + (tx * d);  // index for Q
            for (int x = 0; x < d; x++) {
                Qi[tx * d + x] = Q[q_idx + x];  // load Q into shared memory
            }

            // load previous l and m from global memory
            float row_m_prev = m[lm_offset + (Br * i) + tx];  // previous row max
            float row_l_prev = l[lm_offset + (Br * i) + tx];  // previous row sum

            // compute S = QK^T using warp parallelism
            float row_m = -INFINITY;  // initialize row max for softmax
            for (int y = 0; y < Bc; y++) {
                float sum = 0.0f;  // accumulator for dot product
                for (int x = 0; x < d; x += 32) {  // process in chunks of 32 elements
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];  // dot product
                }
                sum *= softmax_scale;  // scale by softmax factor
                S[tx * Bc + y] = sum;  // store attention score
                row_m = fmaxf(row_m, sum);  // update row max
            }
            __syncthreads();  // synchronize threads to ensure S is computed

            // softmax computation
            float row_l = 0;  // initialize row sum for softmax
            for (int y = 0; y < Bc; y++) {
                S[tx * Bc + y] = __expf(S[tx * Bc + y] - row_m);  // exponentiate
                row_l += S[tx * Bc + y];  // accumulate row sum
            }

            // compute new l and m for numerical stability
            float row_m_new = fmaxf(row_m_prev, row_m);  // new row max
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) +
                             (__expf(row_m - row_m_new) * row_l);  // new row sum

            // compute dS 
            for (int y = 0; y < Bc; y++) {
                dS[tx * Bc + y] = 0.0f;  // initialize gradient
                for (int x = 0; x < d; x++) {
                    int o_idx = qkv_offset + (Br * i * d) + (tx * d) + x;  // index for dO
                    dS[tx * Bc + y] += dO[o_idx] * Vj[y * d + x];  // accumulate gradient
                }
                dS[tx * Bc + y] *= S[tx * Bc + y] / row_l_new;  // scale by softmax
            }
            __syncthreads();  // synchronize threads to ensure dS is computed

            // compute dQ
            for (int x = 0; x < d; x++) {
                float sum = 0.0f;  // accumulator for gradient
                for (int y = 0; y < Bc; y++) {
                    sum += dS[tx * Bc + y] * Kj[y * d + x];  // accumulate gradient
                }
                int q_idx = qkv_offset + (Br * i * d) + (tx * d) + x;  // index for dQ
                dQ[q_idx] += sum * softmax_scale;  // store gradient
            }

            // compute dK and dV 
            for (int y = 0; y < Bc; y++) {
                float sum_k = 0.0f;  // accumulator for dK
                float sum_v = 0.0f;  // accumulator for dV
                for (int x = 0; x < d; x++) {
                    sum_k += dS[tx * Bc + y] * Qi[tx * d + x];  // accumulate dK
                    sum_v += dO[qkv_offset + (Br * i * d) + (tx * d) + x] * S[tx * Bc + y];  // accumulate dV
                }
                int k_idx = qkv_offset + (Bc * j * d) + (y * d) + tx;  // index for dK
                int v_idx = qkv_offset + (Bc * j * d) + (y * d) + tx;  // index for dV
                dK[k_idx] += sum_k * softmax_scale;  // store dK
                dV[v_idx] += sum_v;  // store dV
            }
        }
        __syncthreads();  // synchronize threads before moving to the next tile
    }
}