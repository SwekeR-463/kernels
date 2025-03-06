import torch
import triton
import triton.language as tl
# DEVICE = torch.device(f"cuda: {torch.cuda.current_device()}"")
DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

print(DEVICE)
# print(triton.__version__)

# triton.jit decorator tells triton to compile this function into gpu code
@triton.jit 
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements,
               BLOCK_SIZE: tl.constexpr
               # no of elements each program should process
               # tl.constexpr designates BLOCK_SIZE as a compile time variable
               ):
    pid = tl.program_id(axis=0) # program_id -> 1d output so axis = 0 but for bigger ops we define it as tuple
    # for instance, if we have a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256]
    
    # tell the program to process inputs that are offset from the initial data
    block_start = pid * BLOCK_SIZE
    
    # offset is array of int that act as pointers
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # create a mask to guard memory operations against out-of-bounds accesses
    mask = offsets < n_elements
    
    # load X & Y from VRAM/HBM to SRAM/OCM 
    # the not currently in-use data is stored on DRAM and calculations are done on the data that's in SRAM
    x = tl.load(x_ptr + offsets, mask=mask, other=None)
    y = tl.load(y_ptr + offsets, mask=mask, other=None)
    # mask -> ensures to not access memory beyond the vector's end
    # other -> what value to put in the place of any masked-out values
    
    output = x + y
    
    # write output back to DRAM
    tl.store(output_ptr + offsets, output, mask=mask)
    
# wrapper torch function
def add(x: torch.Tensor, y: torch.Tensor):
    # preallocating the output
    output = torch.empty_like(x)
    
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE,  \
        f'DEVICE: {DEVICE}, x.device: {x.device}, y.device: {y.device}, output.device: {output.device}'
        
    n_elements = output.numel() # numel returns total no of entries in tensor of any shape
    
    # grid -> defines no of kernel instances that run in parallel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # triton.cdiv -> (n_elements + (BLOCK_SIZE - 1)) // BLOCK_SIZE
    # meta -> returns a tuple with no of programs we want to instantiate at once
    
    # kernel launch
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output


def test_add_kernel(size, atol=1e-3, rtol=1e-3, device=DEVICE):
    torch.manual_seed(0)
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    z_tri = add(x, y)
    z_ref = x + y
    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print(f'The maximum difference between torch and triton is {torch.max(torch.abs(z_ref - z_tri))}')
    print("PASSED HURRAY!!!!!!!!!!")
    
if __name__ == "__main__":
    test_add_kernel(size=98432)