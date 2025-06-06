import triton
import triton.language as tl
import torch

@triton.jit
def swiglu_forward_kernel(
    x_ptr,  
    y_ptr,  
    n_elements: tl.constexpr,  
    BLOCK_SIZE: tl.constexpr,  
    feature_dim: tl.constexpr  # feature dimension (size of x_W and x_V)
):
    pid = tl.program_id(axis=0)  
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  

    # load input tensor (split into x_W and x_V)
    x_W = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_V = tl.load(x_ptr + offsets + feature_dim, mask=mask, other=0.0)

    # SwiGLU activation: SiLU(x_W) * x_V
    swiglu = (x_W * tl.sigmoid(x_W)) * x_V

    tl.store(y_ptr + offsets, swiglu, mask=mask)

@triton.jit
def swiglu_backward_kernel(
    dy_ptr,  # pointer to gradient of loss w.r.t. output
    x_ptr,   # pointer to input tensor
    dx_ptr,  # pointer to gradient of loss w.r.t. input
    n_elements: tl.constexpr,  
    BLOCK_SIZE: tl.constexpr,  
    feature_dim: tl.constexpr  # feature dimension (size of x_W and x_V)
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # load inputs (split into x_W and x_V)
    x_W = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_V = tl.load(x_ptr + offsets + feature_dim, mask=mask, other=0.0)
    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0)

    # compute SiLU and its derivative for x_W
    sig = tl.sigmoid(x_W)
    sig_derivative = sig * (1 - sig)
    silu_grad = sig + x_W * sig_derivative

    # compute gradients for x_W and x_V
    dx_W = dy * x_V * silu_grad
    dx_V = dy * (x_W * sig)

    tl.store(dx_ptr + offsets, dx_W, mask=mask)
    tl.store(dx_ptr + offsets + feature_dim, dx_V, mask=mask)

def swiglu_forward(x):
    n_elements = x.numel() // 2  # output has half the feature size
    feature_dim = x.shape[-1] // 2  # split input into two halves
    y = torch.empty_like(x[..., :feature_dim])

    grid = (triton.cdiv(n_elements, 1024),)  
    swiglu_forward_kernel[grid](x, y, n_elements, BLOCK_SIZE=1024, feature_dim=feature_dim, num_warps=4)

    return y

def swiglu_backward(dy, x):
    n_elements = x.numel() // 2  # output has half the feature size
    feature_dim = x.shape[-1] // 2  # split input into two halves
    dx = torch.empty_like(x)

    grid = (triton.cdiv(n_elements, 1024),) 
    swiglu_backward_kernel[grid](dy, x, dx, n_elements, BLOCK_SIZE=1024, feature_dim=feature_dim, num_warps=4)

    return dx

class SwiGLUTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = swiglu_forward(x)
        ctx.save_for_backward(x)  # save input for backward pass
        return y

    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        dx = swiglu_backward(dy, x)
        return dx

class TritonSwiGLULayer(torch.nn.Module):
    def forward(self, x):
        return SwiGLUTriton.apply(x)

# test the implementation
x = torch.randn(4096, device='cuda', requires_grad=True, dtype=torch.float64)
y = SwiGLUTriton.apply(x)

# backward test
dy = torch.ones_like(y)  # assume dL/dy = 1
dx = torch.autograd.grad(y, x, grad_outputs=dy)[0]

print(y)  # forward pass output
print(dx) # backward pass gradients

# gradient check 
test = torch.autograd.gradcheck(SwiGLUTriton.apply, (x,), eps=1e-6, atol=1e-5, nondet_tol=1e-5)
print("Gradient check passed:", test) # True