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
        
@triton.jit
def _layernorm_backward_dLdx(
    x_ptr, dLdx_ptr, dLdy_ptr,
    w_ptr,
    dLdw_intermediate_ptr, dLdb_intermediate_ptr,
    mean_ptr, rstd_ptr,
    locks_ptr,
    stride, N,
    GROUP_SIZE: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    PID = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    x_ptr += PID * stride
    dLdx_ptr += PID * stride
    dLdy_ptr += PID * stride
    
    x = tl.load(x_ptr + cols, mask=mask, other=0).to(tl.float32)
    dLdy = tl.load(dLdy_ptr + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
    mean = tl.load(mean_ptr + PID)
    rstd = tl.load(rstd_ptr + PID)
    
    x_normalized = tl.where(mask, (x - mean) * rstd, 0.)        
    dydx_normed = tl.where(mask, w * dLdy, 0.)                 
    c1 = tl.sum(x_normalized * dydx_normed, axis=0) / N        
    c2 = tl.sum(dydx_normed, axis=0) / N                        
    dLdx = (dydx_normed - (x_normalized * c1 + c2)) * rstd 
    
    tl.store(dLdx_ptr + cols, dLdx, mask=mask)

    dLdw_contribution = (dLdy * x_normalized).to(w.dtype)
    dLdb_contribution = (dLdy).to(w.dtype)
    
    lock_id = PID % GROUP_SIZE
    locks_ptr += lock_id
    
    count_ptr = locks_ptr + GROUP_SIZE
    
    dLdw_intermediate_ptrs = dLdw_intermediate_ptr + lock_id * N + cols 
    dLdb_intermediate_ptrs = dLdb_intermediate_ptr + lock_id * N + cols 
    
    while tl.atomic_cas(locks_ptr, 0, 1) == 1:
        pass
        
    
    count = tl.load(count_ptr) 
    if count == 0: 
        tl.atomic_xchg(count_ptr, 1)
    else: 
        dLdw_contribution += tl.load(dLdw_intermediate_ptrs, mask=mask) 
        dLdb_contribution += tl.load(dLdb_intermediate_ptrs, mask=mask) 
        
    tl.store(dLdw_intermediate_ptrs, dLdw_contribution, mask=mask)
    tl.store(dLdb_intermediate_ptrs, dLdb_contribution, mask=mask)

    tl.atomic_xchg(locks_ptr, 0)
    
@triton.jit
def _layernorm_backward_dLdw_dLdb(
    dLdw_intermediate_ptr,  dLdb_intermediate_ptr, 
    dLdw_ptr, dLdb_ptr,                          
    GROUP_SIZE,  N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    PID = tl.program_id(0)
    col_ptrs = PID * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    dLdw_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    dLdb_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(0, GROUP_SIZE, BLOCK_SIZE_M):
        row_ptrs = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (row_ptrs[:, None] < GROUP_SIZE) & (col_ptrs[None, :] < N)
        offsets = row_ptrs[:, None] * N + col_ptrs[None, :]

        dLdw_acc += tl.load(dLdw_intermediate_ptr + offsets, mask=mask, other=0.) 
        dLdb_acc += tl.load(dLdb_intermediate_ptr + offsets, mask=mask, other=0.)


    sum_dLdw = tl.sum(dLdw_acc, axis=0) 
    sum_dLdb = tl.sum(dLdb_acc, axis=0)

    tl.store(dLdw_ptr + col_ptrs, sum_dLdw, mask=col_ptrs < N)
    tl.store(dLdb_ptr + col_ptrs, sum_dLdb, mask=col_ptrs < N)
    

