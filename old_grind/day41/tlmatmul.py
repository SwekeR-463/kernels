import torch
import triton
import triton.language as tl
DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')


# autotuning is just setting up a bunch of different potential meta-parameters configurations that Triton will automatically
# choose from later based on which one performs best on our specific GPU. Triton will figure out for us which one to use. They're 
# all values chosen heuristically, but notice everything is a multiple of 32 in sticking w/ the number of threads in a warp.
autotune_configs = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2)
]

@triton.autotune(configs = autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr, 
    M, N, K, 
    stride_a_M, stride_a_K, 
    stride_b_K, stride_b_N, 
    stride_c_M, stride_c_N, 
    # meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, 
):

    # we start with a 1D launch grid that we will turn into a 2D grid with a complicated "group-wise" ordering
    PID = tl.program_id(axis=0) 
    # defining the size of groups
    num_PID_along_M = tl.cdiv(M, BLOCK_SIZE_M) # the number of blocks along M dimension
    num_PID_along_N = tl.cdiv(N, BLOCK_SIZE_N) # the number of blocks along N dimension
    num_PID_in_group = GROUP_SIZE * num_PID_along_N
    # figurinig out which group this PID is in
    group_id = PID // num_PID_in_group 
    # tells us which row to start at for this group
    first_PID_in_group_along_M = group_id * GROUP_SIZE 
    # this is usually equal to GROUP_SIZE; the alternative case happens when we're at edge of the tensor and 
    #  its dimensions don't cleanly divde into GROUP_SIZE # TODO is this true?
    group_size_adj = min(num_PID_along_M - first_PID_in_group_along_M, GROUP_SIZE) 
    # this is the bulk of the actual mapping of PIDs to group-major ordering
    PID_M = first_PID_in_group_along_M + ((PID % num_PID_in_group) % group_size_adj)
        # (PID % num_PID_in_group) puts the current program id into the context of a group
        # (first_PID_in_group_along_m + ...) shifts the PID into the correct group
        # (... % group_size_adj) removes the column component to get us onto the correct row
    PID_N = (PID % num_PID_in_group) // group_size_adj
        # (... // group_size_adj) removes the row component to get us onto the correct column
    

    # let's create pointer vectors for the first group of blocks of the input matrices
    offsets_M = PID_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_N = PID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_K = tl.arange(0, BLOCK_SIZE_K)
    # in previous lessons the blocks we loaded into SRAM were vectors; here they are matrices
    a_offsets = offsets_M[:, None] * stride_a_M + offsets_K[None, :] * stride_a_K
    b_offsets = offsets_K[:, None] * stride_b_K + offsets_N[None, :] * stride_b_N

    # inputs tensors are fp16 but we accumulate into a block of fp32 values for higher accuracy (we'll revert later)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32) # the full C is shape (M, N)
        # for a demonstration of why accumulation works, check out `./block_wise_matmul.png`
        
    # we'll iterate along the K dimension of both A and B to compute a single block of the C matrix
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # out-of-bounds entries (along K) need to be masked out
        mask = offsets_K < K - k * BLOCK_SIZE_K
            # k * BLOCK_SIZE_K is the current starting index of offsets_k.
            # so this only really activates when k is within BLOCK_SIZE_K entries from K
            # meaning this gets triggered on the last iteration of the loop, and only if K is not a multiple of BLOCK_SIZE_K
        
        # now we load blocks of A and B matrices. If multiple blocks in a group are on the same SM, 
        # they can share these loaded values, which reduces the number of expensive loads from DRAM
        a = tl.load(a_ptr + a_offsets, mask=mask[None, :], other=0.0) # shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_ptr + b_offsets, mask=mask[:, None], other=0.0) # shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
            # fill in any masked-out parts with 0.0's since they don't have any effect on the summation in the next step

        # we accumulate along the K dimension
        accumulator = tl.dot(a, b, acc=accumulator)
            # triton is weird with operation notation; this is actually a tiny matmul not a dot product
            #   shape (BLOCK_SIZE_M, BLOCK_SIZE_K) @ (BLOCK_SIZE_K, BLOCK_SIZE_N) = (BLOCK_SIZE_M, BLOCK_SIZE_N)
            # `acc` tells Triton to write the output of the matmul directly to accumulator, which is more efficient than
            #   accumulator += tl.dot(a, b)

        # advance the pointers to the next block along K
        a_offsets += BLOCK_SIZE_K * stride_a_K
        b_offsets += BLOCK_SIZE_K * stride_b_K

    # and now we reset the data type to the expected output
    accumulator = accumulator.to(tl.float16)

    # write back the block of the output matrix C with masks
    c_offsets = stride_c_M * offsets_M[:, None] + stride_c_N * offsets_N[None, :]
    c_mask = (offsets_M[:, None] < M) & (offsets_N[None, :] < N) # notice the 2D mask
    tl.store(c_ptr + c_offsets, accumulator, mask=c_mask) # shape (BLOCK_SIZE_M, BLOCK_SIZE_N)

# wrapper func
def matmul(a, b):
    # check constraints
    assert a.ndim == b.ndim == 2, "only supports matrices, not vectors or tensors"
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    #assert a.is_contiguous() and b.is_contiguous, "input matrices must be contiguous"
    a, b = a.to(torch.float16), b.to(torch.float16)
    
    # get dimesion lengths
    (M, K), (_, N) = a.shape, b.shape

    # allocates output
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    # cdiv(x, y) = (x + (y - 1)) // y
    # a naive (slow) launch grid might try to separate our axes of parallelizatio into 2 dimensions, one
    #  for cdiv(M, BLOCK_SIZE_M) and the other for cdiv(N, BLOCK_SIZE_N)
    # Here instead we use a 1D launch kernel defined by cdiv(M, BLOCK_SIZE_M) * cdiv(N, BLOCK_SIZE_N)
    # The reasoning behind this is explained inside the kernel
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

def test_matmul_kernel(size: tuple, atol=1e-2, rtol=1e-1, device=DEVICE): 
    
    torch.manual_seed(0)
    assert type(size) == tuple and len(size) == 2
    a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    
    c_tri = matmul(a, b)
    c_ref = torch.matmul(a, b)
    
    torch.testing.assert_close(c_tri, c_ref, atol=atol, rtol=rtol)
    print(f'The maximum difference between torch and triton is {torch.max(torch.abs(c_ref - c_tri))}')
    print("PASSED")
    
if __name__ == "__main__":
    
    test_matmul_kernel(size=(1024, 1024))