import triton
import triton.language as tl
import torch

@triton.jit
def self_attention_kernel(
    Q_ptr, K_ptr, V_ptr, output_ptr,  # pointers to input/output tensors
    seq_len, d_k, batch_size,         # dimensions
    stride_qb, stride_qs, stride_qd,  # strides for Q (batch, seq, dim)
    stride_kb, stride_ks, stride_kd,  # strides for K
    stride_vb, stride_vs, stride_vd,  # strides for V
    stride_ob, stride_os, stride_od,  # strides for output
    BLOCK_SIZE: tl.constexpr          # tile size for computation
):
    # get batch and sequence indices from program ID
    pid_batch = tl.program_id(0)
    pid_seq = tl.program_id(1)

    # compute offsets for this block
    q_offset = pid_batch * stride_qb + pid_seq * stride_qs
    k_offset = pid_batch * stride_kb
    v_offset = pid_batch * stride_vb
    o_offset = pid_batch * stride_ob + pid_seq * stride_os

    # load Q tile (one row of the sequence)
    q_idxs = q_offset + tl.arange(0, BLOCK_SIZE)
    q_mask = tl.arange(0, BLOCK_SIZE) < d_k
    q = tl.load(Q_ptr + q_idxs, mask=q_mask, other=0.0)

    # accumulate log-sum-exp for numerical stability and output incrementally
    output = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    lse = tl.zeros((1,), dtype=tl.float32)  # log-sum-exp for softmax normalization
    max_score = tl.full((1,), -float('inf'), dtype=tl.float32)

    # compute scores and softmax incrementally
    for k_start in range(0, seq_len, BLOCK_SIZE):
        k_idxs = k_offset + k_start * stride_ks + tl.arange(0, BLOCK_SIZE)
        k_mask = (k_start + tl.arange(0, BLOCK_SIZE)) < seq_len
        k = tl.load(K_ptr + k_idxs, mask=k_mask, other=0.0)

        # compute block-wise scores: Q @ K^T
        scores = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        scores = tl.where(k_mask, tl.sum(q * k, axis=0), scores)

        # scale scores
        d_k_float = d_k.to(tl.float32)
        scores = scores / tl.sqrt(d_k_float)

        # update max and log-sum-exp for softmax
        new_max = tl.maximum(max_score, tl.max(scores, axis=0))
        old_exp = tl.exp(max_score - new_max)
        new_exp = tl.sum(tl.where(k_mask, tl.exp(scores - new_max), 0.0), axis=0)
        lse = lse * old_exp + new_exp
        max_score = new_max

        # load V block and compute partial output
        v_idxs = v_offset + k_start * stride_vs + tl.arange(0, BLOCK_SIZE)
        v_mask = (k_start + tl.arange(0, BLOCK_SIZE)) < seq_len
        v = tl.load(V_ptr + v_idxs, mask=v_mask, other=0.0)
        attn_weights = tl.where(k_mask, tl.exp(scores - max_score) / lse, 0.0)
        output += tl.sum(attn_weights * v, axis=0)

    # store output
    o_idxs = o_offset + tl.arange(0, BLOCK_SIZE)
    o_mask = tl.arange(0, BLOCK_SIZE) < d_k
    tl.store(output_ptr + o_idxs, output, mask=o_mask)

# usage
def run_self_attention(Q, K, V):
    batch_size, seq_len, d_k = Q.shape
    output = torch.empty_like(Q, device='cuda')

    grid = (batch_size, seq_len)

    # kernel
    self_attention_kernel[grid](
        Q, K, V, output,
        seq_len, d_k, batch_size,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_SIZE=32  
    )
    return output

# testing on random data
batch_size, seq_len, d_k = 4, 64, 64
Q = torch.randn(batch_size, seq_len, d_k, device='cuda')
K = torch.randn(batch_size, seq_len, d_k, device='cuda')
V = torch.randn(batch_size, seq_len, d_k, device='cuda')
output = run_self_attention(Q, K, V)
print(output.shape)  # (4, 64, 64)