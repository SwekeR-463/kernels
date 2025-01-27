import torch
import time

N = 1024 * 1024

A = torch.rand(N, device='cuda')
B = torch.rand(N, device='cuda')

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

for _ in range(10):
    C = A * B

start_event.record()

C = A * B

end_event.record()

torch.cuda.synchronize()

elapsed_time = start_event.elapsed_time(end_event)

print(f"PyTorch kernel execution time: {elapsed_time:.4f} ms")
# 