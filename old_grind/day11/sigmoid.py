import torch
import numpy as np
import time

def pytorch_sigmoid(x):
    return torch.sigmoid(x)

def cuda_sigmoid(input_array):
    import ctypes
    # load the CUDA shared library
    sigmoid_lib = ctypes.cdll.LoadLibrary('./sigmoid.so')
    
    # prepare inputs
    input_array = input_array.astype(np.float32)
    output_array = np.zeros_like(input_array, dtype=np.float32)

    sigmoid_lib.sigmoid(
        input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(input_array.size),
    )
    return output_array

# input data
size = 1024*1024
input_data = torch.randn(size)

# pytorch timing
start = time.time()
output_torch = pytorch_sigmoid(input_data)
torch_time = time.time() - start
print(f"PyTorch time: {torch_time:.6f} seconds")

# cuda timing
start = time.time()
output_cuda = cuda_sigmoid(input_data.numpy())
cuda_time = time.time() - start
print(f"CUDA time: {cuda_time:.6f} seconds")
# PyTorch time: 0.004542 seconds
# CUDA time: 0.140152 seconds