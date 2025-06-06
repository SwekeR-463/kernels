import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

# Optimized autotune configs
autotune_configs = [
    triton.Config({'BLOCK_SIZE_SEQ': 128, 'BLOCK_SIZE_DIM': 32}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_SEQ': 64, 'BLOCK_SIZE_DIM': 32}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_SEQ': 128, 'BLOCK_SIZE_DIM': 16}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_SEQ': 64, 'BLOCK_SIZE_DIM': 16}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_SEQ': 32, 'BLOCK_SIZE_DIM': 32}, num_stages=5, num_warps=2),
]

@triton.autotune(configs=autotune_configs, key=['seq_len', 'embed_dim', 'batch_size'])
@triton.jit
def _rope_kernel(
    input_ptr, output_ptr, theta_ptr,
    seq_len, embed_dim, batch_size,
    stride_input_b, stride_input_seq, stride_input_dim,
    stride_output_b, stride_output_seq, stride_output_dim,
    # Meta-parameters
    BLOCK_SIZE_SEQ: tl.constexpr, BLOCK_SIZE_DIM: tl.constexpr
):
    pid_seq = tl.program_id(0)  # Sequence position
    pid_dim = tl.program_id(1)  # Dimension pair index
    pid_batch = tl.program_id(2)  # Batch index

    seq_start = pid_seq * BLOCK_SIZE_SEQ
    dim_start = pid_dim * BLOCK_SIZE_DIM
    batch_offset = pid_batch * stride_input_b

    offsets_seq = seq_start + tl.arange(0, BLOCK_SIZE_SEQ)
    offsets_dim = dim_start + tl.arange(0, BLOCK_SIZE_DIM)

    valid_seq_mask = offsets_seq < seq_len
    valid_dim_mask = offsets_dim < (embed_dim // 2)

    theta_offsets = offsets_dim
    theta = tl.load(theta_ptr + theta_offsets, mask=valid_dim_mask, other=1.0)  # Avoid zero division

    input_base = input_ptr + batch_offset
    output_base = output_ptr + batch_offset

    for i in range(BLOCK_SIZE_SEQ):  # Explicit loop to replace static_range
        pos = seq_start + i
        pos_mask = pos < seq_len  # Ensure we stay in bounds

        pos_offset = pos * stride_input_seq
        dim_offsets = 2 * offsets_dim

        idx = pos_offset + dim_offsets * stride_input_dim
        mask = valid_dim_mask & pos_mask

        # Load input pairs with mask
        x0 = tl.load(input_base + idx, mask=mask, other=0.0)
        x1 = tl.load(input_base + idx + 1, mask=mask, other=0.0)  # Use +1 for next dim element

        # Compute RoPE
        angle = pos * theta
        cos_val = tl.cos(angle)
        sin_val = tl.sin(angle)

        rot_x0 = x0 * cos_val - x1 * sin_val
        rot_x1 = x0 * sin_val + x1 * cos_val

        # Store results with mask
        tl.store(output_base + idx, rot_x0, mask=mask)
        tl.store(output_base + idx + 1, rot_x1, mask=mask)

def rope(input, theta):
    assert input.ndim == 3, "Input must be 3D (batch, seq, embed)"
    assert theta.ndim == 1, "Theta must be 1D"
    assert input.dtype == torch.float16, "Input must be fp16"
    assert theta.dtype == torch.float32, "Theta must be fp32"
    assert input.shape[2] % 2 == 0, "Embedding dimension must be even"

    batch_size, seq_len, embed_dim = input.shape
    assert theta.shape[0] == embed_dim // 2, "Theta size must match embed_dim/2"

    output = torch.empty_like(input)

    grid = lambda meta: (
        min(triton.cdiv(seq_len, meta['BLOCK_SIZE_SEQ']), 65535),
        min(triton.cdiv(embed_dim // 2, meta['BLOCK_SIZE_DIM']), 65535),
        min(batch_size, 65535)
    )

    _rope_kernel[grid](
        input, output, theta,
        seq_len, embed_dim, batch_size,
        input.stride(0), input.stride(1), input.stride(2),
        output.stride(0), output.stride(1), output.stride(2)
    )
    return output

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')


def test_rope_kernel(batch_size=2, seq_len=64, embed_dim=128, atol=1e-2, rtol=1e-1, device=DEVICE):
    torch.manual_seed(0)
    
    input = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float16, device=device)
    theta = torch.randn(embed_dim // 2, dtype=torch.float32, device=device)

    output_tri = rope(input, theta)

    output_ref = input.clone()
    for b in range(batch_size):
        for t in range(seq_len):
            for d in range(0, embed_dim, 2):
                angle = t * theta[d//2]
                cos_val = torch.cos(angle)
                sin_val = torch.sin(angle)
                x0, x1 = input[b, t, d], input[b, t, d+1]
                output_ref[b, t, d] = x0 * cos_val - x1 * sin_val
                output_ref[b, t, d+1] = x0 * sin_val + x1 * cos_val

    torch.testing.assert_close(output_tri, output_ref, atol=atol, rtol=rtol)
    print(f'Max difference: {torch.max(torch.abs(output_ref - output_tri))}')
    print("PASSED")

if __name__ == "__main__":
    test_rope_kernel()