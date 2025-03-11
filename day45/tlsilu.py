import triton
import triton.language as tl
import torch

@triton.jit
def silu_fwd_kernel(X_ptr, Y_ptr, Sigmoid_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask)
    sigmoid_x = 1 / (1 + tl.exp(-x))
    y = x * sigmoid_x

    tl.store(Y_ptr + offsets, y, mask=mask)
    tl.store(Sigmoid_ptr + offsets, sigmoid_x, mask=mask)  # store sigmoid for reuse in backward pass

def silu_forward(x):
    N = x.numel()
    y = torch.empty_like(x)
    sigmoid_x = torch.empty_like(x)  # store sigmoid for backward pass
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    silu_fwd_kernel[grid](x, y, sigmoid_x, N, BLOCK_SIZE)
    return y, sigmoid_x  # return both SiLU output and stored sigmoid

@triton.jit
def silu_bwd_kernel(GradY_ptr, X_ptr, Sigmoid_ptr, GradX_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    dy = tl.load(GradY_ptr + offsets, mask=mask)  # gradient from next layer
    x = tl.load(X_ptr + offsets, mask=mask)       # original input
    sigmoid_x = tl.load(Sigmoid_ptr + offsets, mask=mask)  # stored sigmoid value from forward pass

    # compute SiLU gradient
    dx = dy * (sigmoid_x + x * sigmoid_x * (1 - sigmoid_x))

    tl.store(GradX_ptr + offsets, dx, mask=mask)

def silu_backward(grad_y, x, sigmoid_x):
    N = x.numel()
    grad_x = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    silu_bwd_kernel[grid](grad_y, x, sigmoid_x, grad_x, N, BLOCK_SIZE)
    return grad_x

x = torch.randn(4096, device="cuda", requires_grad=True)

# forward pass
y, sigmoid_x = silu_forward(x)

# loss gradient (dL/dy)
grad_y = torch.ones_like(y)  # assume dL/dy = 1 for simplicity

# backward pass
grad_x = silu_backward(grad_y, x, sigmoid_x)

# compare with pytorch autograd
y_torch = torch.nn.functional.silu(x)
y_torch.backward(torch.ones_like(y_torch))  
print(torch.allclose(x.grad, grad_x, atol=1e-6))  # True