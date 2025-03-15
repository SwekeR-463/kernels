import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda: {torch.cuda.current_device()}')

@triton.jit
def _batchnorm_forward(
    x_ptr, y_ptr,
    gamma_ptr, beta_ptr, 
    mean_ptr, var_ptr,
    N, eps: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # mean
    x_sum_block = tl.sum(x, axis=0)
    
    global_sum_ptr = mean_ptr
    tl.atomic_add(global_sum_ptr, x_sum_block)
    
    if pid == 0:
        global_sum = tl.load(global_sum_ptr)
        mean = global_sum / N
        tl.store(mean_ptr, mean)
        
    mean = tl.load(mean_ptr)
    
    # variance
    
    diff = x - mean
    sq_diff = diff * diff
    var_sum_block = tl.sum(sq_diff, axis=0)
    
    global_var_sum_ptr = var_ptr
    tl.atomic_add(global_var_sum_ptr, var_sum_block)
    
    var = tl.load(var_ptr)
    rstd = tl.load(var_ptr + 1)
    
    # normalization
    gamma = tl.load(gamma_ptr)
    beta = tl.load(beta_ptr)
    y = gamma * (x - mean) * rstd + beta
    tl.store(y_ptr + offsets, y, mask=mask)