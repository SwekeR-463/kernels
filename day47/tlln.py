import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda: {torch.cuda.current_device()}')

@triton.jit
def _layernorm_forward(
    x_ptr, y_ptr,
    w_ptr, b_ptr, 
    mean_ptr, rstd_ptr, 
    stride_M, 
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # use the p_id to move x_ptr & y_ptr to the row of X & Y they will compute
    row = tl.program_id(0)
    x_ptr += row * stride_M
    y_ptr += row * stride_M
    
    # compute mean
    sum_accumulator = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        x_ptrs = tl.load(x_ptr + cols, mask=cols < N, other=0).to(tl.float32)
        sum_accumulator += x_ptrs
    mean = tl.sum(sum_accumulator, axis=0) / N
    
    # compute variance
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        x_ptrs = tl.load(x_ptr + cols, mask=cols < N, other=0.).to(tl.float32)
        diff = tl.where(cols < N, x_ptrs - mean, 0.)
        acc += diff * diff
        
    var = tl.sum(acc, axis=0) / N
    
    rstd = 1 / tl.sqrt(var + eps)
    
    # store the mean & std dev for back pass later
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)
    
    # normalize and apply linear transformation
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w_ptrs = tl.load(w_ptr + cols, mask=mask)
        b_ptrs = tl.load(b_ptr + cols, mask=mask)
        x_ptrs = tl.load(x_ptr + cols, mask=mask)
        
        x_hat = (x_ptrs - mean) * rstd
        y = x_hat * w_ptrs + b_ptrs
        
        # write the output
        tl.store(y_ptr + cols, y, mask=mask)