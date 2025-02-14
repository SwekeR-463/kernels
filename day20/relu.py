import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

N = 1 << 20  # 1M elements
input_tensor = torch.rand(N, device=device) * 2 - 1 

for _ in range(10):
    _ = torch.relu(input_tensor)

start_time = time.time()
output_tensor = torch.relu(input_tensor)
end_time = time.time()

torch.cuda.synchronize()

elapsed_time = (end_time - start_time) * 1000  
print(f"PyTorch ReLU execution time: {elapsed_time:.3f} ms")
# PyTorch ReLU execution time: 0.013 ms